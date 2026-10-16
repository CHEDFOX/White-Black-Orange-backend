"""Point-in-time query layer.

Every read used by a historical analysis goes through this module. The rule it
enforces, in one sentence: **a row is visible at instant ``T`` only if
``available_at <= T``, and among the revisions that satisfy that, only the
latest one is returned.**

Why a dedicated layer rather than a convention:

* ``available_at`` is easy to forget in an ad-hoc query, and forgetting it
  produces an analysis that looks plausible and is worthless. Centralising the
  predicate means one audited implementation instead of forty call sites.
* Revision selection is subtle. Filtering on ``available_at <= T`` alone
  returns *every* vintage published before ``T``, so a series would contain
  duplicates. The correct answer picks, per entity-and-period, the row with the
  greatest ``available_at`` not exceeding ``T``.
* ``ingested_at`` must never appear in a filter. A backfill run today writes
  2019 rows with ``ingested_at`` of today; filtering on it would make 2019 data
  invisible to a 2019 analysis.

``AsOf`` wraps the cutoff instant so it cannot be confused with an ordinary
timestamp argument, and carries the disclosures a report must repeat.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, TypeVar

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.orm import Session

from app.core.errors import LookAheadError
from app.core.timeutil import ensure_utc, now_utc
from app.db.models.enums import DataClassification

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AsOf:
    """An information cutoff.

    ``cutoff`` is the instant beyond which nothing may be seen. ``purpose``
    records why the cutoff exists, which ends up in the report so a reader
    knows whether they are looking at a reconstruction or a live view.
    """

    cutoff: dt.datetime
    purpose: str = "historical_reconstruction"
    # Accumulated statements about what this reconstruction cannot guarantee,
    # e.g. "macro series are latest-revision because no FRED key is set".
    disclosures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff", ensure_utc(self.cutoff, field="cutoff"))

    @classmethod
    def now(cls, *, purpose: str = "current_analysis") -> AsOf:
        return cls(cutoff=now_utc(), purpose=purpose)

    @property
    def is_live(self) -> bool:
        """True when the cutoff is effectively the present moment."""
        return (now_utc() - self.cutoff).total_seconds() < 60

    def with_disclosure(self, text: str) -> AsOf:
        if text in self.disclosures:
            return self
        return AsOf(
            cutoff=self.cutoff,
            purpose=self.purpose,
            disclosures=(*self.disclosures, text),
        )

    def assert_not_future(self, observed_at: dt.datetime, *, what: str) -> None:
        """Raise if ``observed_at`` is after the cutoff.

        Used as a belt-and-braces check inside analysis code: if a value ever
        reaches a calculation with a timestamp beyond the cutoff, that is a bug
        and it must fail loudly rather than quietly bias a result.
        """
        observed = ensure_utc(observed_at, field=what)
        if observed > self.cutoff:
            raise LookAheadError(
                f"{what} is timestamped {observed.isoformat()}, which is after the "
                f"information cutoff {self.cutoff.isoformat()}. This would leak "
                "future information into a historical analysis.",
                details={
                    "observed_at": observed.isoformat(),
                    "cutoff": self.cutoff.isoformat(),
                    "what": what,
                },
            )


@dataclass
class PitCoverage:
    """What a point-in-time read actually found, for honest reporting."""

    rows_returned: int = 0
    first_event_time: dt.datetime | None = None
    last_event_time: dt.datetime | None = None
    sources: set[str] = field(default_factory=set)
    used_revisions: dict[int, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def visible_at(model: Any, as_of: AsOf) -> Any:
    """SQL predicate: this row was available at the cutoff and not yet superseded.

    ``superseded_at`` is checked as well as ``available_at`` so that a row
    retired before the cutoff drops out. Both halves are required: the first
    excludes rows published later, the second excludes rows already replaced.
    """
    return and_(
        model.available_at <= as_of.cutoff,
        or_(model.superseded_at.is_(None), model.superseded_at > as_of.cutoff),
    )


def exclude_synthetic(model: Any) -> Any:
    """Predicate keeping synthetic fixture data out of production reads."""
    return model.classification == DataClassification.PRODUCTION


def latest_revision_subquery(
    model: Any,
    as_of: AsOf,
    *,
    partition_by: tuple[Any, ...],
) -> Select:
    """Select the newest visible revision per ``partition_by`` group.

    Implemented with a window function rather than a correlated subquery
    because the latter degrades badly once a series has millions of rows and
    several vintages each.
    """
    ranked = (
        select(
            model.id.label("row_id"),
            func.row_number()
            .over(
                partition_by=partition_by,
                order_by=(model.available_at.desc(), model.revision.desc()),
            )
            .label("rn"),
        )
        .where(visible_at(model, as_of))
        .subquery()
    )
    return select(ranked.c.row_id).where(ranked.c.rn == 1)


def apply_pit(
    stmt: Select,
    model: Any,
    as_of: AsOf,
    *,
    partition_by: tuple[Any, ...] | None = None,
    production_only: bool = True,
) -> Select:
    """Add point-in-time filtering to a query.

    When ``partition_by`` is given, only the newest visible revision per group
    survives. Omit it only for tables whose rows are never revised.
    """
    stmt = stmt.where(visible_at(model, as_of))
    if production_only:
        stmt = stmt.where(exclude_synthetic(model))
    if partition_by:
        newest = latest_revision_subquery(model, as_of, partition_by=partition_by)
        stmt = stmt.where(model.id.in_(newest))
    return stmt


def assert_no_future_rows(
    session: Session, model: Any, as_of: AsOf, *, extra_filter: Any = None
) -> None:
    """Fail if any row visible to a query postdates the cutoff.

    Called by the look-ahead test suite. It asserts the invariant against the
    database itself rather than trusting that every call site remembered the
    predicate.
    """
    stmt = select(func.count()).select_from(model).where(visible_at(model, as_of))
    if extra_filter is not None:
        stmt = stmt.where(extra_filter)
    visible = session.execute(stmt).scalar_one()

    leak_stmt = (
        select(func.count())
        .select_from(model)
        .where(visible_at(model, as_of))
        .where(model.available_at > as_of.cutoff)
    )
    if extra_filter is not None:
        leak_stmt = leak_stmt.where(extra_filter)
    leaked = session.execute(leak_stmt).scalar_one()

    if leaked:
        raise LookAheadError(
            f"{leaked} of {visible} rows visible at {as_of.cutoff.isoformat()} have "
            f"available_at beyond the cutoff in {model.__tablename__}",
            details={"table": model.__tablename__, "leaked": leaked, "visible": visible},
        )


def availability_from_publication(
    published_at: dt.datetime,
    *,
    provider_lag: dt.timedelta = dt.timedelta(0),
) -> dt.datetime:
    """Derive ``available_at`` from a publication instant and provider latency.

    Availability is never earlier than publication. ``provider_lag`` models the
    delay before *this platform* could have read the value, which is the honest
    cutoff for a strategy that would have had to act on our own feed.
    """
    return ensure_utc(published_at, field="published_at") + max(
        provider_lag, dt.timedelta(0)
    )
