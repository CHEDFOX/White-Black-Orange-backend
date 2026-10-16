"""Macro series ingestion.

Writes ``has_vintages`` on each series so the point-in-time layer can disclose,
per series, whether a historical read returns the figure published at the time
or today's revised one. That flag is the difference between an honest macro
backtest and a misleading one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models.enums import JobStatus
from app.db.models.macro import EconomicObservation, EconomicSeries
from app.db.models.provider import DataSource
from app.providers.fred import FredProvider, MacroSeriesSpec

log = get_logger(__name__)


@dataclass
class MacroReport:
    series_code: str = ""
    observations_written: int = 0
    observations_unchanged: int = 0
    revisions_written: int = 0
    has_vintages: bool = False
    status: str = JobStatus.SUCCEEDED
    notes: list[str] = field(default_factory=list)


def upsert_series(
    session: Session,
    *,
    data_source: DataSource,
    spec: MacroSeriesSpec,
    has_vintages: bool,
) -> EconomicSeries:
    existing = session.execute(
        select(EconomicSeries).where(
            EconomicSeries.data_source_id == data_source.id,
            EconomicSeries.series_code == spec.code,
        )
    ).scalars().first()

    notes = (
        f"{spec.notes} Typical publication lag: "
        f"{spec.typical_publication_lag_days} day(s)."
    ).strip()

    if existing is None:
        series = EconomicSeries(
            data_source_id=data_source.id,
            series_code=spec.code,
            title=spec.title,
            frequency=spec.frequency,
            unit=spec.unit,
            seasonal_adjustment=spec.seasonal_adjustment,
            country=spec.country,
            category=spec.category,
            has_vintages=has_vintages,
            notes=notes,
        )
        session.add(series)
        session.flush()
        return series

    existing.title = spec.title
    existing.has_vintages = has_vintages
    existing.notes = notes
    session.flush()
    return existing


def ingest_series(
    session: Session,
    *,
    provider: FredProvider,
    data_source: DataSource,
    series_code: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> MacroReport:
    """Ingest one macro series, preserving vintages when they are available."""
    spec = provider.series_spec(series_code)
    report = MacroReport(series_code=spec.code, has_vintages=provider.has_vintages)
    series = upsert_series(
        session, data_source=data_source, spec=spec, has_vintages=provider.has_vintages
    )

    observations = provider.fetch_observations(spec.code, start=start, end=end)
    if not observations:
        report.status = JobStatus.PARTIAL
        report.notes.append(f"FRED returned no observations for {spec.code}")
        return report

    existing_rows = session.execute(
        select(EconomicObservation).where(EconomicObservation.series_id == series.id)
    ).scalars().all()
    existing_by_key = {(r.event_time, r.revision): r for r in existing_rows}

    for observation in observations:
        key = (observation.event_time, observation.revision)
        prior = existing_by_key.get(key)
        if prior is not None:
            if prior.value == observation.value:
                report.observations_unchanged += 1
                continue
            # A changed value at the same revision number means the provider
            # restated without advancing its own vintage; store it as a new
            # revision rather than overwriting.
            next_revision = (
                max(
                    (r.revision for r in existing_rows if r.event_time == observation.event_time),
                    default=observation.revision,
                )
                + 1
            )
            prior.superseded_at = observation.available_at
        else:
            next_revision = observation.revision

        session.add(
            EconomicObservation(
                series_id=series.id,
                value=observation.value,
                unit=observation.unit,
                event_time=observation.event_time,
                published_at=observation.published_at,
                available_at=observation.available_at,
                revision=next_revision,
                data_source_id=data_source.id,
                raw_payload={
                    "publication_known": observation.publication_known,
                    "series_code": spec.code,
                },
            )
        )
        if next_revision > 0:
            report.revisions_written += 1
        report.observations_written += 1

    session.flush()

    if not provider.has_vintages:
        report.notes.append(
            f"{spec.code} was ingested without revision vintages because no "
            "FRED_API_KEY is configured. Availability timestamps are estimated "
            f"from the series' documented {spec.typical_publication_lag_days}-day "
            "publication lag, and the values are the latest revisions rather than "
            "the figures published at the time."
        )
    return report


def ingest_curated_series(
    session: Session,
    *,
    provider: FredProvider,
    data_source: DataSource,
    codes: tuple[str, ...] | None = None,
    start: dt.date | None = None,
) -> list[MacroReport]:
    """Ingest the curated macro set."""
    from app.providers.fred import CURATED_SERIES

    selected = codes or tuple(CURATED_SERIES)
    out: list[MacroReport] = []
    for code in selected:
        try:
            out.append(
                ingest_series(
                    session,
                    provider=provider,
                    data_source=data_source,
                    series_code=code,
                    start=start,
                )
            )
        except Exception as exc:
            log.warning("macro_ingest_failed", series=code, error=str(exc))
            out.append(
                MacroReport(
                    series_code=code,
                    status=JobStatus.FAILED,
                    notes=[f"{type(exc).__name__}: {exc}"],
                )
            )
    return out
