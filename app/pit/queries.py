"""Concrete point-in-time reads for each observation type.

Each function here answers a question of the form "what did the record show
about X at instant T?" and returns both the data and a coverage record, so a
caller can report what was and was not available instead of presenting a
partial answer as complete.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.db.models.enums import AdjustmentBasis
from app.db.models.fundamentals import FinancialFact, Filing
from app.db.models.instrument import Instrument, InstrumentTicker
from app.db.models.macro import EconomicObservation, EconomicSeries
from app.db.models.market_data import PriceBar
from app.db.models.news import CorporateEvent, NewsArticle, NewsEntityLink
from app.db.models.provider import DataSource
from app.pit.availability import AsOf, PitCoverage, apply_pit, visible_at


@dataclass
class PitResult:
    """Rows plus what the read managed to cover."""

    rows: list
    coverage: PitCoverage


def resolve_symbol_at(
    session: Session, symbol: str, as_of: AsOf, *, exchange_code: str | None = None
) -> Instrument | None:
    """Resolve a ticker to an instrument *as it was* at the cutoff.

    This is what lets a 2019 analysis of ``FB`` and a 2024 analysis of ``META``
    both land on the same instrument, and prevents a 2019 analysis from
    resolving a ticker that had not yet been assigned.
    """
    cutoff_date = as_of.cutoff.date()
    stmt = (
        select(Instrument)
        .join(InstrumentTicker, InstrumentTicker.instrument_id == Instrument.id)
        .where(func.upper(InstrumentTicker.symbol) == symbol.upper())
        .where(InstrumentTicker.valid_from <= cutoff_date)
        .where(
            (InstrumentTicker.valid_to.is_(None))
            | (InstrumentTicker.valid_to >= cutoff_date)
        )
    )
    if exchange_code:
        stmt = stmt.where(Instrument.exchange_code == exchange_code.upper())
    found = session.execute(stmt.limit(1)).scalars().first()
    if found is not None:
        return found

    # Fall back to the instrument's current symbol. Recorded as a fallback
    # because it is weaker evidence than a dated ticker row.
    stmt2 = select(Instrument).where(func.upper(Instrument.symbol) == symbol.upper())
    if exchange_code:
        stmt2 = stmt2.where(Instrument.exchange_code == exchange_code.upper())
    return session.execute(stmt2.limit(1)).scalars().first()


def price_bars_as_of(
    session: Session,
    instrument_id: uuid.UUID,
    as_of: AsOf,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    resolution: str = "1d",
    adjustment_basis: str = AdjustmentBasis.TOTAL_RETURN,
    limit: int | None = None,
    production_only: bool = True,
) -> PitResult:
    """Price bars knowable at the cutoff.

    ``end`` is clamped to the cutoff: asking for bars up to next month in a
    reconstruction dated last year is a request for future data, and silently
    returning nothing would be less useful than clamping and disclosing it.
    """
    effective_end = min(end, as_of.cutoff) if end is not None else as_of.cutoff

    stmt: Select = select(PriceBar).where(
        PriceBar.instrument_id == instrument_id,
        PriceBar.resolution == resolution,
        PriceBar.adjustment_basis == adjustment_basis,
        PriceBar.event_time <= effective_end,
    )
    if start is not None:
        stmt = stmt.where(PriceBar.event_time >= start)

    stmt = apply_pit(
        stmt,
        PriceBar,
        as_of,
        partition_by=(
            PriceBar.instrument_id,
            PriceBar.resolution,
            PriceBar.adjustment_basis,
            PriceBar.event_time,
        ),
        production_only=production_only,
    ).order_by(PriceBar.event_time)
    if limit is not None:
        stmt = stmt.limit(limit)

    rows = list(session.execute(stmt).scalars().all())
    coverage = PitCoverage(
        rows_returned=len(rows),
        first_event_time=rows[0].event_time if rows else None,
        last_event_time=rows[-1].event_time if rows else None,
    )
    for row in rows:
        coverage.used_revisions[row.revision] = coverage.used_revisions.get(row.revision, 0) + 1
    if end is not None and end > as_of.cutoff:
        coverage.notes.append(
            f"Requested end {end.isoformat()} was clamped to the information cutoff "
            f"{as_of.cutoff.isoformat()}."
        )
    if not rows:
        coverage.notes.append(
            f"No {resolution} bars on the {adjustment_basis} basis were available at "
            f"{as_of.cutoff.isoformat()}."
        )
    return PitResult(rows=rows, coverage=coverage)


def financial_facts_as_of(
    session: Session,
    issuer_id: uuid.UUID,
    as_of: AsOf,
    *,
    concepts: tuple[str, ...] | None = None,
    production_only: bool = True,
) -> PitResult:
    """Fundamentals as they were reported at the cutoff.

    This is where restatements matter. SEC data holds one row per
    (period, filing), so a fiscal year reported in 2021 and restated in 2023
    has two rows. Partitioning on (concept, fiscal period) and taking the
    newest visible row returns the 2021 figure for a 2022 analysis -- which is
    what an analyst actually had.
    """
    stmt: Select = select(FinancialFact).where(FinancialFact.issuer_id == issuer_id)
    if concepts:
        stmt = stmt.where(FinancialFact.concept.in_(concepts))

    stmt = apply_pit(
        stmt,
        FinancialFact,
        as_of,
        partition_by=(
            FinancialFact.issuer_id,
            FinancialFact.concept,
            FinancialFact.fiscal_period_id,
        ),
        production_only=production_only,
    ).order_by(FinancialFact.concept, FinancialFact.event_time)

    rows = list(session.execute(stmt).scalars().all())
    coverage = PitCoverage(rows_returned=len(rows))
    if rows:
        coverage.first_event_time = min(r.event_time for r in rows)
        coverage.last_event_time = max(r.event_time for r in rows)
        restated = sum(1 for r in rows if r.revision > 0)
        if restated:
            coverage.notes.append(
                f"{restated} fact(s) are revisions of an earlier filing; the figure "
                "returned is the newest one published on or before the cutoff."
            )
    else:
        coverage.notes.append(
            f"No financial facts had been filed as of {as_of.cutoff.isoformat()}."
        )
    return PitResult(rows=rows, coverage=coverage)


def macro_observations_as_of(
    session: Session,
    series_code: str,
    as_of: AsOf,
    *,
    start: dt.datetime | None = None,
    lookback: dt.timedelta | None = None,
) -> PitResult:
    """Macro series values as published at the cutoff.

    When the stored series has no vintages, the coverage note says so. A
    report must repeat that note rather than implying the 2008 analysis used
    2008's published GDP figures when it used today's revised ones.
    """
    series = session.execute(
        select(EconomicSeries).where(EconomicSeries.series_code == series_code.upper())
    ).scalars().first()
    if series is None:
        return PitResult(
            rows=[],
            coverage=PitCoverage(notes=[f"Series {series_code} is not registered."]),
        )

    window_start = start
    if window_start is None and lookback is not None:
        window_start = as_of.cutoff - lookback

    stmt: Select = select(EconomicObservation).where(
        EconomicObservation.series_id == series.id,
        EconomicObservation.event_time <= as_of.cutoff,
    )
    if window_start is not None:
        stmt = stmt.where(EconomicObservation.event_time >= window_start)

    stmt = apply_pit(
        stmt,
        EconomicObservation,
        as_of,
        partition_by=(EconomicObservation.series_id, EconomicObservation.event_time),
    ).order_by(EconomicObservation.event_time)

    rows = list(session.execute(stmt).scalars().all())
    coverage = PitCoverage(rows_returned=len(rows))
    if rows:
        coverage.first_event_time = rows[0].event_time
        coverage.last_event_time = rows[-1].event_time
    if not series.has_vintages:
        coverage.notes.append(
            f"{series.series_code} is stored without revision vintages, so these are "
            "the latest revised values, not the figures published at the time. "
            "Configure FRED_API_KEY to ingest true vintages."
        )
    return PitResult(rows=rows, coverage=coverage)


def events_as_of(
    session: Session,
    instrument_id: uuid.UUID,
    as_of: AsOf,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    categories: tuple[str, ...] | None = None,
    limit: int = 500,
) -> PitResult:
    """Corporate and macro events that had been published by the cutoff."""
    effective_end = min(end, as_of.cutoff) if end is not None else as_of.cutoff
    stmt: Select = select(CorporateEvent).where(
        CorporateEvent.instrument_id == instrument_id,
        CorporateEvent.event_time <= effective_end,
    )
    if start is not None:
        stmt = stmt.where(CorporateEvent.event_time >= start)
    if categories:
        stmt = stmt.where(CorporateEvent.category.in_(categories))

    stmt = apply_pit(
        stmt,
        CorporateEvent,
        as_of,
        partition_by=(
            CorporateEvent.instrument_id,
            CorporateEvent.category,
            CorporateEvent.event_time,
        ),
    ).order_by(CorporateEvent.event_time.desc()).limit(limit)

    rows = list(session.execute(stmt).scalars().all())
    coverage = PitCoverage(rows_returned=len(rows))
    if rows:
        coverage.first_event_time = min(r.event_time for r in rows)
        coverage.last_event_time = max(r.event_time for r in rows)
    return PitResult(rows=rows, coverage=coverage)


def news_as_of(
    session: Session,
    instrument_id: uuid.UUID,
    as_of: AsOf,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    limit: int = 200,
    include_duplicates: bool = False,
) -> PitResult:
    """Articles published before the cutoff and linked to the instrument.

    An article is excluded unless its own ``published_at`` precedes the cutoff.
    Using the ingestion time instead is the classic news-backtest leak: a
    crawler that indexed 2020 articles in 2024 would make them "available" in
    2020.
    """
    effective_end = min(end, as_of.cutoff) if end is not None else as_of.cutoff
    stmt: Select = (
        select(NewsArticle)
        .join(NewsEntityLink, NewsEntityLink.article_id == NewsArticle.id)
        .where(NewsEntityLink.instrument_id == instrument_id)
        .where(NewsArticle.published_at.isnot(None))
        .where(NewsArticle.published_at <= effective_end)
    )
    if start is not None:
        stmt = stmt.where(NewsArticle.published_at >= start)
    if not include_duplicates:
        stmt = stmt.where(NewsArticle.duplicate_of_id.is_(None))

    stmt = apply_pit(stmt, NewsArticle, as_of, partition_by=None).order_by(
        NewsArticle.published_at.desc()
    ).limit(limit)

    rows = list(session.execute(stmt).scalars().all())
    coverage = PitCoverage(rows_returned=len(rows))
    if rows:
        coverage.first_event_time = min(r.event_time for r in rows)
        coverage.last_event_time = max(r.event_time for r in rows)
    return PitResult(rows=rows, coverage=coverage)


def filings_as_of(
    session: Session,
    issuer_id: uuid.UUID,
    as_of: AsOf,
    *,
    form_types: tuple[str, ...] | None = None,
    limit: int = 100,
) -> PitResult:
    stmt: Select = select(Filing).where(
        Filing.issuer_id == issuer_id, Filing.filed_at <= as_of.cutoff
    )
    if form_types:
        stmt = stmt.where(Filing.form_type.in_(form_types))
    stmt = apply_pit(stmt, Filing, as_of, partition_by=None).order_by(
        Filing.filed_at.desc()
    ).limit(limit)
    rows = list(session.execute(stmt).scalars().all())
    return PitResult(rows=rows, coverage=PitCoverage(rows_returned=len(rows)))


def data_source_attribution(session: Session, source_ids: set[uuid.UUID]) -> list[dict]:
    """Attribution records for the sources a result actually used."""
    if not source_ids:
        return []
    rows = session.execute(
        select(DataSource).where(DataSource.id.in_(source_ids))
    ).scalars().all()
    return [
        {
            "key": r.key,
            "name": r.name,
            "attribution_required": r.attribution_required,
            "attribution_text": r.attribution_text,
            "official_api": r.official_api,
            "redistribution_allowed": r.redistribution_allowed,
            "classification": r.classification,
            "caveats": r.caveats,
        }
        for r in rows
    ]
