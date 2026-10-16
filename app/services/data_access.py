"""Assembles analysis inputs from point-in-time database reads.

This is the only place the analysis engines' ``MethodInput`` is constructed, so
there is exactly one implementation of "load the bars that were knowable at T
and turn them into a feature-engine input". Everything about the load is
reported back in a ``SeriesBundle``: which source supplied it, how many bars,
what gaps and quality issues the window contains, and whether anything was
clamped. A caller that wants to present a number can therefore also present
what the number rests on.

Floats appear here for the first time, deliberately and at one boundary: prices
are stored and read as ``Decimal``, and converted to float only when handed to
NumPy and SciPy. The conversion is one-way -- nothing converts back and writes
a float to the ledger.
"""

from __future__ import annotations

import datetime as dt
import decimal
import uuid
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.analysis.registry import MethodInput
from app.core.config import settings
from app.core.errors import InsufficientDataError, NotFoundError
from app.core.logging import get_logger
from app.db.models.enums import AdjustmentBasis, DataQualitySeverity
from app.db.models.fundamentals import FinancialFact, FiscalPeriod
from app.db.models.instrument import Instrument, InstrumentCoverage
from app.db.models.market_data import DataGap, DataQualityIssue, PriceBar
from app.db.models.provider import DataSource
from app.pit.availability import AsOf
from app.pit.queries import financial_facts_as_of, price_bars_as_of, resolve_symbol_at

log = get_logger(__name__)


@dataclass
class SeriesBundle:
    """A loaded price series plus everything a caller must disclose about it."""

    instrument: Instrument
    resolution: str
    adjustment_basis: str
    timestamps: list[dt.datetime]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float | None]
    currency: str
    as_of: AsOf
    source_keys: list[str] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    quality_issues: list[dict] = field(default_factory=list)
    coverage_notes: list[str] = field(default_factory=list)
    revisions_used: dict[int, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.close)

    @property
    def first_timestamp(self) -> dt.datetime | None:
        return self.timestamps[0] if self.timestamps else None

    @property
    def last_timestamp(self) -> dt.datetime | None:
        return self.timestamps[-1] if self.timestamps else None

    @property
    def has_defects(self) -> bool:
        return bool(self.gaps) or any(
            i["severity"] == DataQualitySeverity.ERROR for i in self.quality_issues
        )

    def to_method_input(
        self,
        *,
        benchmark: SeriesBundle | None = None,
        scalars: dict[str, float] | None = None,
        series: dict[str, list[float]] | None = None,
    ) -> MethodInput:
        """Build a ``MethodInput``, aligning any benchmark by timestamp.

        Alignment is by exact timestamp intersection rather than by position.
        Two series of equal length can still be misaligned -- a different
        holiday calendar is enough -- and a one-bar offset silently destroys a
        beta estimate.
        """
        benchmark_close: tuple[float, ...] = ()
        if benchmark is not None:
            aligned = align_series(self, benchmark)
            if aligned is None:
                log.info(
                    "benchmark_alignment_failed",
                    instrument=self.instrument.symbol,
                    benchmark=benchmark.instrument.symbol,
                )
            else:
                own, bench = aligned
                return MethodInput(
                    timestamps=tuple(own["timestamps"]),
                    open=tuple(own["open"]),
                    high=tuple(own["high"]),
                    low=tuple(own["low"]),
                    close=tuple(own["close"]),
                    volume=tuple(own["volume"]),
                    benchmark_close=tuple(bench),
                    scalars=dict(scalars or {}),
                    series={k: tuple(v) for k, v in (series or {}).items()},
                    resolution=self.resolution,
                    currency=self.currency,
                    trading_days_per_year=settings.trading_days_per_year,
                    as_of=self.as_of.cutoff,
                )

        return MethodInput(
            timestamps=tuple(self.timestamps),
            open=tuple(self.open),
            high=tuple(self.high),
            low=tuple(self.low),
            close=tuple(self.close),
            volume=tuple(self.volume),
            benchmark_close=benchmark_close,
            scalars=dict(scalars or {}),
            series={k: tuple(v) for k, v in (series or {}).items()},
            resolution=self.resolution,
            currency=self.currency,
            trading_days_per_year=settings.trading_days_per_year,
            as_of=self.as_of.cutoff,
        )


def align_series(
    a: SeriesBundle, b: SeriesBundle
) -> tuple[dict[str, list], list[float]] | None:
    """Intersect two series on exact timestamps.

    Returns ``a``'s fields restricted to the shared timestamps, plus ``b``'s
    closes on the same grid. None when the overlap is too small to be useful.
    """
    b_by_time = dict(zip(b.timestamps, b.close, strict=True))
    keep = [i for i, ts in enumerate(a.timestamps) if ts in b_by_time]
    if len(keep) < settings.min_bars_for_analysis:
        return None
    return (
        {
            "timestamps": [a.timestamps[i] for i in keep],
            "open": [a.open[i] for i in keep],
            "high": [a.high[i] for i in keep],
            "low": [a.low[i] for i in keep],
            "close": [a.close[i] for i in keep],
            "volume": [a.volume[i] for i in keep],
        },
        [b_by_time[a.timestamps[i]] for i in keep],
    )


def _f(value: decimal.Decimal | None) -> float | None:
    """Decimal -> float at the single boundary where NumPy needs it."""
    return None if value is None else float(value)


def get_instrument(session: Session, instrument_id: uuid.UUID) -> Instrument:
    found = session.get(Instrument, instrument_id)
    if found is None:
        raise NotFoundError(f"instrument {instrument_id} not found")
    return found


def resolve_instrument(
    session: Session,
    identifier: str,
    *,
    as_of: AsOf | None = None,
    exchange_code: str | None = None,
) -> Instrument:
    """Resolve a UUID or a ticker to an instrument."""
    try:
        return get_instrument(session, uuid.UUID(identifier))
    except (ValueError, AttributeError):
        pass

    found = resolve_symbol_at(
        session, identifier, as_of or AsOf.now(), exchange_code=exchange_code
    )
    if found is None:
        raise NotFoundError(
            f"no instrument matches {identifier!r}"
            + (f" on {exchange_code}" if exchange_code else "")
            + ". Search for it first, or ingest it."
        )
    return found


def load_series(
    session: Session,
    instrument: Instrument,
    *,
    as_of: AsOf,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    resolution: str = "1d",
    adjustment_basis: str = AdjustmentBasis.TOTAL_RETURN,
    min_bars: int | None = None,
    include_quality: bool = True,
) -> SeriesBundle:
    """Load a point-in-time price series with its quality context."""
    result = price_bars_as_of(
        session,
        instrument.id,
        as_of,
        start=start,
        end=end,
        resolution=resolution,
        adjustment_basis=adjustment_basis,
    )
    rows: list[PriceBar] = result.rows

    required = settings.min_bars_for_analysis if min_bars is None else min_bars
    if len(rows) < required:
        raise InsufficientDataError(
            f"{instrument.symbol} has {len(rows)} {resolution} bars on the "
            f"{adjustment_basis} basis available at {as_of.cutoff.isoformat()}, "
            f"but at least {required} are needed. "
            + ("; ".join(result.coverage.notes) if result.coverage.notes else ""),
            details={
                "instrument": instrument.symbol,
                "available": len(rows),
                "required": required,
                "resolution": resolution,
                "adjustment_basis": adjustment_basis,
                "as_of": as_of.cutoff.isoformat(),
                "notes": result.coverage.notes,
            },
        )

    source_ids = {r.data_source_id for r in rows}
    source_keys = list(
        session.execute(
            select(DataSource.key).where(DataSource.id.in_(source_ids))
        ).scalars().all()
    )

    bundle = SeriesBundle(
        instrument=instrument,
        resolution=resolution,
        adjustment_basis=adjustment_basis,
        timestamps=[r.event_time for r in rows],
        open=[_f(r.open) for r in rows],
        high=[_f(r.high) for r in rows],
        low=[_f(r.low) for r in rows],
        close=[_f(r.close) for r in rows],
        volume=[_f(r.volume) for r in rows],
        currency=rows[0].currency,
        as_of=as_of,
        source_keys=sorted(source_keys),
        coverage_notes=list(result.coverage.notes),
        revisions_used=dict(result.coverage.used_revisions),
    )

    if include_quality and rows:
        bundle.gaps = _load_gaps(
            session, instrument.id, resolution, rows[0].event_time, rows[-1].event_time
        )
        bundle.quality_issues = _load_quality_issues(
            session, instrument.id, resolution, rows[0].event_time, rows[-1].event_time
        )
        if bundle.revisions_used and max(bundle.revisions_used) > 0:
            revised = sum(n for rev, n in bundle.revisions_used.items() if rev > 0)
            bundle.coverage_notes.append(
                f"{revised} bar(s) in this window are corrected revisions; the "
                "values shown are those available at the analysis timestamp."
            )
    return bundle


def _load_gaps(
    session: Session,
    instrument_id: uuid.UUID,
    resolution: str,
    start: dt.datetime,
    end: dt.datetime,
) -> list[dict]:
    rows = session.execute(
        select(DataGap)
        .where(
            DataGap.instrument_id == instrument_id,
            DataGap.resolution == resolution,
            DataGap.gap_start >= start,
            DataGap.gap_start <= end,
        )
        .order_by(DataGap.gap_start)
    ).scalars().all()
    return [
        {
            "gap_start": g.gap_start.isoformat(),
            "gap_end": g.gap_end.isoformat(),
            "expected_sessions": g.expected_sessions,
            "reason": g.reason,
            "filled": g.filled_at is not None,
        }
        for g in rows
    ]


def _load_quality_issues(
    session: Session,
    instrument_id: uuid.UUID,
    resolution: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    limit: int = 100,
) -> list[dict]:
    rows = session.execute(
        select(DataQualityIssue)
        .where(
            DataQualityIssue.instrument_id == instrument_id,
            DataQualityIssue.resolution == resolution,
            DataQualityIssue.window_start >= start,
            DataQualityIssue.window_start <= end,
            DataQualityIssue.severity.in_(
                [DataQualitySeverity.WARNING, DataQualitySeverity.ERROR]
            ),
        )
        .order_by(DataQualityIssue.window_start)
        .limit(limit)
    ).scalars().all()
    return [
        {
            "kind": i.kind,
            "severity": i.severity,
            "detail": i.detail,
            "at": i.window_start.isoformat() if i.window_start else None,
            "context": i.context,
        }
        for i in rows
    ]


def load_benchmark(
    session: Session,
    *,
    as_of: AsOf,
    start: dt.datetime | None,
    end: dt.datetime | None,
    resolution: str = "1d",
    ticker: str | None = None,
) -> SeriesBundle | None:
    """Load the configured benchmark, or None when it is not ingested.

    Returning None rather than raising is deliberate: a missing benchmark must
    degrade the analysis (no beta, no relative strength) and be disclosed, not
    fail the whole request.
    """
    symbol = ticker or settings.default_benchmark_ticker
    try:
        instrument = resolve_instrument(session, symbol, as_of=as_of)
        return load_series(
            session,
            instrument,
            as_of=as_of,
            start=start,
            end=end,
            resolution=resolution,
            adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
            include_quality=False,
        )
    except (NotFoundError, InsufficientDataError) as exc:
        log.info("benchmark_unavailable", benchmark=symbol, reason=str(exc))
        return None


@dataclass
class FundamentalSnapshot:
    """Point-in-time fundamentals, flattened for the analysis engines."""

    scalars: dict[str, float] = field(default_factory=dict)
    series: dict[str, list[float]] = field(default_factory=dict)
    period_labels: list[str] = field(default_factory=list)
    latest_period_end: dt.date | None = None
    latest_filed_at: dt.datetime | None = None
    currency: str | None = None
    notes: list[str] = field(default_factory=list)
    restated_count: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.scalars and not self.series


def load_fundamentals(
    session: Session,
    instrument: Instrument,
    *,
    as_of: AsOf,
    annual_only: bool = True,
    max_periods: int = 6,
) -> FundamentalSnapshot:
    """Load fundamentals as reported at the analysis timestamp.

    Builds both the latest-period scalars (for ratios) and per-period series
    (for growth), using only facts filed on or before the cutoff -- so a 2022
    analysis sees FY2021 as originally reported, not as restated in 2024.
    """
    snapshot = FundamentalSnapshot()
    if instrument.issuer_id is None:
        snapshot.notes.append(
            f"{instrument.symbol} has no issuer linked, so no fundamentals are "
            "available. ETFs, indexes and currencies have none by nature."
        )
        return snapshot

    result = financial_facts_as_of(session, instrument.issuer_id, as_of)
    facts: list[FinancialFact] = result.rows
    snapshot.notes.extend(result.coverage.notes)
    if not facts:
        return snapshot

    period_ids = {f.fiscal_period_id for f in facts if f.fiscal_period_id}
    periods: dict[uuid.UUID, FiscalPeriod] = {}
    if period_ids:
        for period in session.execute(
            select(FiscalPeriod).where(FiscalPeriod.id.in_(period_ids))
        ).scalars().all():
            periods[period.id] = period

    # Group by period, keeping only comparable periods (annual with annual).
    by_period: dict[tuple, dict[str, float]] = {}
    for fact in facts:
        period = periods.get(fact.fiscal_period_id) if fact.fiscal_period_id else None
        if period is None:
            continue
        # Filter on period_kind, which is derived from the reported duration.
        # The legacy fiscal_period label comes from the filing and does not
        # reliably describe the period.
        if annual_only and period.period_kind != "FY":
            continue
        key = (period.period_end, period.fiscal_year, period.period_kind)
        if fact.value is not None:
            by_period.setdefault(key, {})[fact.concept] = float(fact.value)
        if fact.currency and snapshot.currency is None:
            snapshot.currency = fact.currency
        if fact.is_restatement or fact.revision > 0:
            snapshot.restated_count += 1

    if not by_period:
        snapshot.notes.append(
            "financial facts exist but none could be attached to a comparable "
            f"{'annual' if annual_only else 'fiscal'} period"
        )
        return snapshot

    ordered = sorted(by_period)[-max_periods:]
    snapshot.period_labels = [f"{fy} {kind}" for _, fy, kind in ordered]
    snapshot.latest_period_end = ordered[-1][0]
    snapshot.scalars = dict(by_period[ordered[-1]])

    concepts = {c for values in by_period.values() for c in values}
    for concept in sorted(concepts):
        values = [by_period[k].get(concept) for k in ordered]
        # Only build a growth series when every period has the concept. A
        # series with holes would make consecutive-period growth compare
        # non-adjacent periods.
        if all(v is not None for v in values) and len(values) >= 2:
            snapshot.series[f"{concept}_by_period"] = [float(v) for v in values]

    latest_filed = max(
        (f.available_at for f in facts if f.available_at), default=None
    )
    snapshot.latest_filed_at = latest_filed

    if snapshot.restated_count:
        snapshot.notes.append(
            f"{snapshot.restated_count} fact(s) are revisions of an earlier "
            "filing; the figures shown are the newest published on or before "
            "the analysis timestamp."
        )
    return snapshot


def coverage_summary(session: Session, instrument_id: uuid.UUID) -> list[dict]:
    """What the platform actually holds for an instrument."""
    rows = session.execute(
        select(InstrumentCoverage, DataSource.key)
        .join(DataSource, DataSource.id == InstrumentCoverage.data_source_id)
        .where(InstrumentCoverage.instrument_id == instrument_id)
    ).all()
    return [
        {
            "data_kind": c.data_kind,
            "resolution": c.resolution,
            "source": source_key,
            "first_observation": (
                c.first_observation.isoformat() if c.first_observation else None
            ),
            "last_observation": (
                c.last_observation.isoformat() if c.last_observation else None
            ),
            "observation_count": c.observation_count,
            "last_refreshed_at": (
                c.last_refreshed_at.isoformat() if c.last_refreshed_at else None
            ),
            "known_gap_count": c.known_gap_count,
            "notes": c.notes,
        }
        for c, source_key in rows
    ]


def data_freshness(session: Session, instrument_id: uuid.UUID) -> dict:
    """How current the stored data is, for the UI's freshness indicator."""
    latest = session.execute(
        select(func.max(PriceBar.event_time)).where(
            PriceBar.instrument_id == instrument_id,
            PriceBar.resolution == "1d",
            PriceBar.superseded_at.is_(None),
        )
    ).scalar_one_or_none()
    if latest is None:
        return {"latest_bar": None, "age_days": None, "is_stale": True,
                "note": "no daily bars are stored for this instrument"}

    from app.core.timeutil import now_utc

    age = (now_utc() - latest).days
    return {
        "latest_bar": latest.isoformat(),
        "age_days": age,
        # Four calendar days covers a weekend plus a holiday.
        "is_stale": age > 4,
        "note": (
            f"the most recent stored daily bar is {age} day(s) old"
            if age > 4
            else "data is current to the last completed session"
        ),
    }
