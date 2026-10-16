"""Price ingestion: idempotent, revision-aware, validation-gated.

Idempotency matters because ingestion is retried. Re-running a backfill must
not double a volume figure or create a second vintage of an unchanged bar. The
rule applied here:

* A bar that does not exist is inserted.
* A bar that exists with identical values is left alone (a no-op, not a new
  revision).
* A bar that exists with *different* values becomes a new revision: the old row
  gets ``superseded_at`` set to the new row's ``available_at``, and the new row
  is inserted with ``revision + 1``. A point-in-time query dated before the
  correction still returns the original figure, which is the whole point.

Nothing is ever updated in place. A corrected price is new information with its
own availability timestamp, not an edit to history.
"""

from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.calendars import get_calendar
from app.core.logging import get_logger
from app.core.timeutil import now_utc
from app.db.models.enums import (
    DataClassification,
    DataQualityKind,
    DataQualitySeverity,
    JobStatus,
)
from app.db.models.instrument import Instrument
from app.db.models.market_data import DataGap, DataQualityIssue, PriceBar
from app.db.models.provider import DataSource, IngestionRun
from app.ingestion.instruments import update_coverage
from app.ingestion.validation import (
    ValidationFinding,
    ValidationResult,
    find_session_gaps,
    validate_bars,
)
from app.pit.availability import availability_from_publication
from app.providers.base import BarBatch, MarketDataProvider, ProviderBar, ProviderFetchRequest

log = get_logger(__name__)


@dataclass
class IngestionReport:
    """Everything that happened in one ingestion call."""

    instrument_id: uuid.UUID | None = None
    resolution: str = "1d"
    adjustment_basis: str = ""
    provider_key: str = ""
    bars_received: int = 0
    bars_inserted: int = 0
    bars_unchanged: int = 0
    bars_revised: int = 0
    bars_rejected: int = 0
    gaps_recorded: int = 0
    issues_recorded: int = 0
    findings: list[ValidationFinding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    status: str = JobStatus.SUCCEEDED

    @property
    def bars_written(self) -> int:
        return self.bars_inserted + self.bars_revised


def _bar_fingerprint(bar: ProviderBar) -> str:
    """Stable hash of a bar's economic content, for change detection."""
    parts = [
        str(bar.open), str(bar.high), str(bar.low), str(bar.close),
        "null" if bar.volume is None else str(bar.volume),
        bar.currency, bar.resolution, bar.adjustment_basis,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _values_equal(row: PriceBar, bar: ProviderBar) -> bool:
    def eq(a: decimal.Decimal | None, b: decimal.Decimal | None) -> bool:
        if a is None or b is None:
            return a is None and b is None
        return a == b

    return (
        eq(row.open, bar.open)
        and eq(row.high, bar.high)
        and eq(row.low, bar.low)
        and eq(row.close, bar.close)
        and eq(row.volume, bar.volume)
    )


def ingest_bar_batch(
    session: Session,
    *,
    instrument: Instrument,
    data_source: DataSource,
    batch: BarBatch,
    record_gaps: bool = True,
    outlier_threshold: float | None = None,
) -> IngestionReport:
    """Validate and store a batch of bars."""
    report = IngestionReport(
        instrument_id=instrument.id,
        resolution=batch.resolution,
        adjustment_basis=batch.adjustment_basis,
        provider_key=data_source.key,
        bars_received=len(batch.bars),
    )
    report.notes.extend(batch.notes)

    calendar = get_calendar(instrument.calendar_code)
    gap_window = (
        (batch.requested_start, batch.requested_end)
        if record_gaps and batch.requested_start and batch.requested_end
        else None
    )
    validation: ValidationResult = validate_bars(
        list(batch.bars),
        resolution=batch.resolution,
        calendar=calendar,
        check_gaps_between=gap_window,
        **({"outlier_threshold": outlier_threshold} if outlier_threshold is not None else {}),
    )
    report.findings = list(validation.findings)
    report.bars_rejected = validation.rejected_count

    classification = (
        DataClassification.SYNTHETIC
        if data_source.classification == DataClassification.SYNTHETIC
        else DataClassification.PRODUCTION
    )

    # Load the existing newest revision for each timestamp in one query rather
    # than one query per bar.
    times = [b.event_time for b in validation.accepted]
    existing_rows: dict[dt.datetime, PriceBar] = {}
    if times:
        rows = session.execute(
            select(PriceBar)
            .where(
                PriceBar.instrument_id == instrument.id,
                PriceBar.resolution == batch.resolution,
                PriceBar.adjustment_basis == batch.adjustment_basis,
                PriceBar.data_source_id == data_source.id,
                PriceBar.event_time.in_(times),
                PriceBar.superseded_at.is_(None),
            )
            .order_by(PriceBar.event_time, PriceBar.revision)
        ).scalars().all()
        for row in rows:
            existing_rows[row.event_time] = row

    for bar in validation.accepted:
        # A bar becomes available once its own session has closed plus the
        # provider's delivery lag -- never at its open, which would make the
        # close knowable before it happened.
        available_at = availability_from_publication(
            bar.bar_close_time, provider_lag=batch.availability_lag
        )
        prior = existing_rows.get(bar.event_time)

        if prior is None:
            session.add(
                PriceBar(
                    instrument_id=instrument.id,
                    resolution=batch.resolution,
                    adjustment_basis=batch.adjustment_basis,
                    data_source_id=data_source.id,
                    event_time=bar.event_time,
                    bar_close_time=bar.bar_close_time,
                    published_at=bar.bar_close_time,
                    available_at=available_at,
                    revision=0,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    vwap=bar.vwap,
                    trade_count=bar.trade_count,
                    currency=bar.currency,
                    is_partial=bar.is_partial,
                    provider_timestamp=bar.provider_timestamp,
                    classification=classification,
                    raw_payload=bar.raw,
                )
            )
            report.bars_inserted += 1
            continue

        if _values_equal(prior, bar):
            # Idempotent re-ingestion: identical data is not a new vintage.
            report.bars_unchanged += 1
            continue

        # Genuine correction. Retire the old row *as of the new row's
        # availability*, so queries dated earlier still see the original.
        revision_available_at = max(available_at, now_utc())
        prior.superseded_at = revision_available_at
        session.add(
            PriceBar(
                instrument_id=instrument.id,
                resolution=batch.resolution,
                adjustment_basis=batch.adjustment_basis,
                data_source_id=data_source.id,
                event_time=bar.event_time,
                bar_close_time=bar.bar_close_time,
                published_at=bar.bar_close_time,
                available_at=revision_available_at,
                revision=prior.revision + 1,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                vwap=bar.vwap,
                trade_count=bar.trade_count,
                currency=bar.currency,
                is_partial=bar.is_partial,
                provider_timestamp=bar.provider_timestamp,
                classification=classification,
                raw_payload=bar.raw,
            )
        )
        report.bars_revised += 1
        report.notes.append(
            f"Bar {bar.event_time.isoformat()} was revised: close {prior.close} -> "
            f"{bar.close}. The earlier value remains queryable before "
            f"{revision_available_at.isoformat()}."
        )

    # Persist findings so a report can disclose them later.
    detected_at = now_utc()
    for finding in validation.findings:
        session.add(
            DataQualityIssue(
                instrument_id=instrument.id,
                data_source_id=data_source.id,
                kind=finding.kind,
                severity=finding.severity,
                resolution=batch.resolution,
                window_start=finding.window_start or finding.event_time,
                window_end=finding.window_end or finding.event_time,
                detected_at=detected_at,
                detail=finding.detail,
                context=finding.context,
                classification=classification,
            )
        )
        report.issues_recorded += 1

    if record_gaps and batch.resolution == "1d" and batch.requested_start and batch.requested_end:
        for gap in find_session_gaps(
            validation.accepted,
            calendar,
            start=batch.requested_start,
            end=batch.requested_end,
        ):
            exists = session.execute(
                select(DataGap).where(
                    DataGap.instrument_id == instrument.id,
                    DataGap.resolution == batch.resolution,
                    DataGap.gap_start == gap.gap_start,
                )
            ).scalars().first()
            if exists is not None:
                continue
            session.add(
                DataGap(
                    instrument_id=instrument.id,
                    resolution=batch.resolution,
                    gap_start=gap.gap_start,
                    gap_end=gap.gap_end,
                    expected_sessions=gap.expected_sessions,
                    detected_at=detected_at,
                    reason=(
                        f"{gap.expected_sessions} session(s) expected by the "
                        f"{calendar.name} calendar have no bar from {data_source.key}."
                    ),
                )
            )
            report.gaps_recorded += 1

    session.flush()

    if validation.accepted:
        update_coverage(
            session,
            instrument.id,
            data_source.id,
            data_kind="prices",
            resolution=batch.resolution,
            first_observation=validation.accepted[0].event_time,
            last_observation=validation.accepted[-1].event_time,
            observation_count=_count_bars(session, instrument.id, data_source.id, batch),
            refreshed_at=detected_at,
            known_gap_count=report.gaps_recorded,
            notes="; ".join(batch.notes) or None,
        )

    if report.bars_rejected and not validation.accepted:
        report.status = JobStatus.FAILED
    elif report.bars_rejected:
        report.status = JobStatus.PARTIAL

    return report


def _count_bars(
    session: Session, instrument_id: uuid.UUID, data_source_id: uuid.UUID, batch: BarBatch
) -> int:
    from sqlalchemy import func as sa_func

    return session.execute(
        select(sa_func.count())
        .select_from(PriceBar)
        .where(
            PriceBar.instrument_id == instrument_id,
            PriceBar.data_source_id == data_source_id,
            PriceBar.resolution == batch.resolution,
            PriceBar.adjustment_basis == batch.adjustment_basis,
            PriceBar.superseded_at.is_(None),
        )
    ).scalar_one()


def backfill_prices(
    session: Session,
    *,
    provider: MarketDataProvider,
    data_source: DataSource,
    instrument: Instrument,
    provider_symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    resolution: str = "1d",
    adjustment_basis: str,
    chunk_days: int | None = None,
) -> IngestionReport:
    """Fetch and store a date range, chunking long windows.

    Chunking exists because providers cap a single response; without it a
    ten-year request silently returns the most recent slice and looks complete.
    """
    run = _start_run(
        session,
        data_source=data_source,
        job_name="backfill_prices",
        params={
            "instrument_id": str(instrument.id),
            "provider_symbol": provider_symbol,
            "resolution": resolution,
            "adjustment_basis": adjustment_basis,
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
    )

    combined = IngestionReport(
        instrument_id=instrument.id,
        resolution=resolution,
        adjustment_basis=adjustment_basis,
        provider_key=data_source.key,
    )
    try:
        for window_start, window_end in _chunks(start, end, resolution, chunk_days):
            batch = provider.fetch_bars(
                ProviderFetchRequest(
                    symbol=provider_symbol,
                    start=window_start,
                    end=window_end,
                    resolution=resolution,
                    adjustment_basis=adjustment_basis,
                )
            )
            part = ingest_bar_batch(
                session,
                instrument=instrument,
                data_source=data_source,
                batch=batch,
            )
            _merge(combined, part)
    except Exception as exc:
        run.status = JobStatus.FAILED
        run.error = f"{type(exc).__name__}: {exc}"[:2000]
        run.finished_at = now_utc()
        session.flush()
        raise

    run.status = combined.status
    run.rows_written = combined.bars_written
    run.rows_rejected = combined.bars_rejected
    run.finished_at = now_utc()
    run.stats = {
        "bars_received": combined.bars_received,
        "bars_inserted": combined.bars_inserted,
        "bars_unchanged": combined.bars_unchanged,
        "bars_revised": combined.bars_revised,
        "gaps_recorded": combined.gaps_recorded,
        "findings": {k: v for k, v in _finding_counts(combined).items()},
    }
    session.flush()
    return combined


def incremental_update(
    session: Session,
    *,
    provider: MarketDataProvider,
    data_source: DataSource,
    instrument: Instrument,
    provider_symbol: str,
    resolution: str = "1d",
    adjustment_basis: str,
    overlap_days: int = 5,
) -> IngestionReport:
    """Fetch only what is newer than what we hold.

    The window deliberately overlaps the last stored bar: providers correct
    recent sessions, and the overlap is how a correction is picked up as a
    revision instead of being missed forever.
    """
    last = session.execute(
        select(PriceBar.event_time)
        .where(
            PriceBar.instrument_id == instrument.id,
            PriceBar.resolution == resolution,
            PriceBar.adjustment_basis == adjustment_basis,
            PriceBar.data_source_id == data_source.id,
            PriceBar.superseded_at.is_(None),
        )
        .order_by(PriceBar.event_time.desc())
        .limit(1)
    ).scalars().first()

    end = now_utc()
    if last is None:
        start = end - dt.timedelta(days=365)
    else:
        start = last - dt.timedelta(days=overlap_days)

    return backfill_prices(
        session,
        provider=provider,
        data_source=data_source,
        instrument=instrument,
        provider_symbol=provider_symbol,
        start=start,
        end=end,
        resolution=resolution,
        adjustment_basis=adjustment_basis,
    )


def _chunks(
    start: dt.datetime, end: dt.datetime, resolution: str, chunk_days: int | None
) -> list[tuple[dt.datetime, dt.datetime]]:
    if chunk_days is None:
        chunk_days = {"1m": 5, "5m": 30, "15m": 45, "30m": 45, "1h": 365}.get(
            resolution, 3650
        )
    span = dt.timedelta(days=chunk_days)
    out: list[tuple[dt.datetime, dt.datetime]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + span, end)
        out.append((cursor, stop))
        cursor = stop
    return out or [(start, end)]


def _merge(target: IngestionReport, part: IngestionReport) -> None:
    target.bars_received += part.bars_received
    target.bars_inserted += part.bars_inserted
    target.bars_unchanged += part.bars_unchanged
    target.bars_revised += part.bars_revised
    target.bars_rejected += part.bars_rejected
    target.gaps_recorded += part.gaps_recorded
    target.issues_recorded += part.issues_recorded
    target.findings.extend(part.findings)
    for note in part.notes:
        if note not in target.notes:
            target.notes.append(note)
    if part.status == JobStatus.FAILED:
        target.status = JobStatus.FAILED
    elif part.status == JobStatus.PARTIAL and target.status != JobStatus.FAILED:
        target.status = JobStatus.PARTIAL


def _finding_counts(report: IngestionReport) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in report.findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    return counts


def _start_run(
    session: Session, *, data_source: DataSource, job_name: str, params: dict
) -> IngestionRun:
    """Create an ingestion-run row keyed for idempotency."""
    key_material = f"{job_name}|{data_source.key}|" + "|".join(
        f"{k}={params[k]}" for k in sorted(params)
    )
    idempotency_key = hashlib.sha256(key_material.encode()).hexdigest()

    existing = session.execute(
        select(IngestionRun).where(IngestionRun.idempotency_key == idempotency_key)
    ).scalars().first()
    if existing is not None:
        existing.status = JobStatus.RUNNING
        existing.started_at = now_utc()
        existing.finished_at = None
        existing.error = None
        session.flush()
        return existing

    run = IngestionRun(
        data_source_id=data_source.id,
        job_name=job_name,
        idempotency_key=idempotency_key,
        status=JobStatus.RUNNING,
        started_at=now_utc(),
        request_params=params,
    )
    session.add(run)
    session.flush()
    return run
