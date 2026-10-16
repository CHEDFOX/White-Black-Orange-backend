"""Celery application and scheduled jobs.

The scheduled work is deliberately modest and all of it is idempotent, because
a retried ingestion must not duplicate a bar or invent a revision:

* **Incremental price refresh** re-fetches a short overlapping window so a
  provider's correction to a recent session is picked up as a revision rather
  than missed forever.
* **Gap detection** re-checks recent sessions against the trading calendar.
* **Forecast evaluation** scores forecasts whose horizon has closed. This is
  the job that makes "historical forecasts are evaluated without hindsight"
  true in a running system rather than only in tests.
* **Provider health** records reachability so the data-source endpoint reports
  a measured state rather than an assumption.

Nothing here is required for the API to work. Without a worker the platform is
read-only over whatever has been seeded, which is a legitimate deployment.
"""

from __future__ import annotations

import datetime as dt

from celery import Celery
from celery.schedules import crontab

from app.core.config import settings
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)

celery_app = Celery(
    "stock_intelligence",
    broker=settings.broker_url,
    backend=settings.result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # A provider call can be slow; a hard limit well above the HTTP timeout
    # stops a hung socket from occupying a worker indefinitely.
    task_time_limit=900,
    task_soft_time_limit=840,
    worker_max_tasks_per_child=200,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    beat_schedule={
        "refresh-prices": {
            "task": "app.jobs.celery_app.refresh_all_prices",
            # After the US close, allowing for the consolidated tape to settle.
            "schedule": crontab(hour=22, minute=30),
        },
        "evaluate-forecasts": {
            "task": "app.jobs.celery_app.evaluate_due_forecasts",
            "schedule": crontab(hour=23, minute=15),
        },
        "detect-gaps": {
            "task": "app.jobs.celery_app.detect_recent_gaps",
            "schedule": crontab(hour=23, minute=30),
        },
        "provider-health": {
            "task": "app.jobs.celery_app.record_provider_health",
            "schedule": crontab(minute="*/30"),
        },
        "refresh-macro": {
            "task": "app.jobs.celery_app.refresh_macro",
            "schedule": crontab(hour=12, minute=15),
        },
    },
)


@celery_app.on_after_configure.connect
def _configure(**_: object) -> None:  # pragma: no cover - startup hook
    configure_logging()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
def refresh_instrument_prices(self, instrument_id: str, adjustment_basis: str) -> dict:
    """Fetch any bars newer than what is stored, with a deliberate overlap.

    The overlap is the point: providers correct recent sessions, and re-reading
    them is how a correction becomes a stored revision instead of being lost.
    """
    import uuid

    from app.db.models.enums import JobStatus
    from app.db.session import session_scope
    from app.ingestion.prices import incremental_update
    from app.ingestion.sources import get_data_source
    from app.providers.registry import active_provider_keys, get_provider

    with session_scope() as session:
        from app.services.data_access import get_instrument

        instrument = get_instrument(session, uuid.UUID(instrument_id))
        for provider_key in active_provider_keys():
            source = get_data_source(session, provider_key)
            if source is None:
                continue
            provider = get_provider(provider_key)
            if not provider.capabilities.supports("prices"):
                continue

            # Use the symbol this provider knows the instrument by, not our own.
            identifier = next(
                (
                    i.identifier
                    for i in instrument.identifiers
                    if i.data_source_id == source.id
                ),
                None,
            )
            if identifier is None:
                continue

            try:
                report = incremental_update(
                    session,
                    provider=provider,
                    data_source=source,
                    instrument=instrument,
                    provider_symbol=identifier,
                    adjustment_basis=adjustment_basis,
                )
            except Exception as exc:
                log.warning(
                    "price_refresh_failed",
                    instrument=instrument.symbol,
                    provider=provider_key,
                    error=str(exc),
                )
                raise self.retry(exc=exc) from exc

            return {
                "instrument": instrument.symbol,
                "provider": provider_key,
                "inserted": report.bars_inserted,
                "revised": report.bars_revised,
                "unchanged": report.bars_unchanged,
                "status": report.status,
            }

    return {"instrument": instrument_id, "status": JobStatus.PARTIAL,
            "detail": "no active provider holds an identifier for this instrument"}


@celery_app.task
def refresh_all_prices() -> dict:
    """Queue a refresh for every instrument with stored price coverage."""
    from sqlalchemy import select

    from app.db.models.enums import AdjustmentBasis
    from app.db.models.instrument import InstrumentCoverage
    from app.db.session import session_scope

    queued = 0
    with session_scope() as session:
        instrument_ids = session.execute(
            select(InstrumentCoverage.instrument_id)
            .where(InstrumentCoverage.data_kind == "prices")
            .distinct()
        ).scalars().all()

    for instrument_id in instrument_ids:
        for basis in (AdjustmentBasis.RAW, AdjustmentBasis.TOTAL_RETURN):
            refresh_instrument_prices.delay(str(instrument_id), str(basis))
            queued += 1

    log.info("price_refresh_queued", instruments=len(instrument_ids), tasks=queued)
    return {"instruments": len(instrument_ids), "tasks_queued": queued}


@celery_app.task
def evaluate_due_forecasts(limit: int = 500) -> dict:
    """Score forecasts whose horizon has closed.

    Reads the realised outcome from the stored series using the same
    point-in-time layer the rest of the platform uses, so the evaluation sees
    the same data a reader would.
    """
    import math

    from app.core.timeutil import now_utc
    from app.db.models.enums import AdjustmentBasis, ForecastTarget
    from app.db.session import session_scope
    from app.forecasting.store import due_for_evaluation, record_outcome
    from app.pit.availability import AsOf
    from app.pit.queries import price_bars_as_of

    evaluated = 0
    skipped = 0
    with session_scope() as session:
        for forecast in due_for_evaluation(session, limit=limit):
            if forecast.target != ForecastTarget.LOG_RETURN:
                skipped += 1
                continue

            result = price_bars_as_of(
                session,
                forecast.instrument_id,
                AsOf.now(purpose="forecast_evaluation"),
                start=forecast.information_cutoff,
                end=forecast.resolves_at,
                resolution=forecast.resolution,
                adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
            )
            bars = result.rows
            if len(bars) < 2:
                # The horizon has closed on the calendar but the data has not
                # arrived. Leaving it unevaluated is correct; scoring it against
                # a partial window would flatter or punish the model unfairly.
                skipped += 1
                continue

            base = float(bars[0].close)
            final = float(bars[-1].close)
            if base <= 0 or final <= 0:
                skipped += 1
                continue

            try:
                record_outcome(
                    session,
                    forecast,
                    realized_value=math.log(final / base),
                    realized_at=bars[-1].event_time,
                    notes=(
                        f"scored from {len(bars)} stored bars between "
                        f"{bars[0].event_time.date()} and {bars[-1].event_time.date()}"
                    ),
                )
            except Exception as exc:
                log.warning(
                    "forecast_evaluation_failed",
                    forecast_id=str(forecast.id),
                    error=str(exc),
                )
                skipped += 1
                continue
            evaluated += 1

    log.info("forecasts_evaluated", evaluated=evaluated, skipped=skipped)
    return {"evaluated": evaluated, "skipped": skipped, "at": now_utc().isoformat()}


@celery_app.task
def detect_recent_gaps(lookback_days: int = 30) -> dict:
    """Re-check recent sessions against the trading calendar."""
    from sqlalchemy import select

    from app.core.calendars import get_calendar
    from app.core.timeutil import now_utc
    from app.db.models.enums import AdjustmentBasis
    from app.db.models.instrument import Instrument
    from app.db.models.market_data import DataGap, PriceBar
    from app.db.session import session_scope
    from app.ingestion.validation import expected_sessions_in_window

    recorded = 0
    with session_scope() as session:
        end = now_utc()
        start = end - dt.timedelta(days=lookback_days)
        instruments = session.execute(select(Instrument)).scalars().all()

        for instrument in instruments:
            calendar = get_calendar(instrument.calendar_code)
            expected = expected_sessions_in_window(calendar, start, end)
            if not expected:
                continue

            stored = session.execute(
                select(PriceBar.event_time).where(
                    PriceBar.instrument_id == instrument.id,
                    PriceBar.resolution == "1d",
                    PriceBar.adjustment_basis == AdjustmentBasis.TOTAL_RETURN,
                    PriceBar.event_time >= start,
                    PriceBar.superseded_at.is_(None),
                )
            ).scalars().all()
            have = {ts.astimezone(calendar.tz).date() for ts in stored}
            missing = [d for d in expected if d not in have]
            if not missing:
                continue

            for day in missing:
                session_window = calendar.session(day)
                if session_window is None:
                    continue
                exists = session.execute(
                    select(DataGap).where(
                        DataGap.instrument_id == instrument.id,
                        DataGap.resolution == "1d",
                        DataGap.gap_start == session_window.open_utc,
                    )
                ).scalars().first()
                if exists is not None:
                    continue
                session.add(
                    DataGap(
                        instrument_id=instrument.id,
                        resolution="1d",
                        gap_start=session_window.open_utc,
                        gap_end=session_window.close_utc,
                        expected_sessions=1,
                        detected_at=now_utc(),
                        reason=(
                            f"the {calendar.name} calendar expects a session on "
                            f"{day} but no bar is stored"
                        ),
                    )
                )
                recorded += 1

    log.info("gap_detection_complete", gaps_recorded=recorded)
    return {"gaps_recorded": recorded}


@celery_app.task
def record_provider_health() -> dict:
    """Probe each active provider and store the result."""
    from app.db.models.provider import ProviderHealthCheck
    from app.db.session import session_scope
    from app.ingestion.sources import get_data_source
    from app.providers.registry import active_provider_keys, get_provider

    results: dict[str, bool] = {}
    with session_scope() as session:
        for provider_key in active_provider_keys():
            source = get_data_source(session, provider_key)
            if source is None:
                continue
            try:
                status = get_provider(provider_key).health_check()
            except Exception as exc:
                log.warning("health_check_failed", provider=provider_key, error=str(exc))
                continue
            session.add(
                ProviderHealthCheck(
                    data_source_id=source.id,
                    checked_at=status.checked_at,
                    healthy=status.healthy,
                    latency_ms=status.latency_ms,
                    status_code=status.status_code,
                    detail=status.detail[:2000] if status.detail else None,
                )
            )
            results[provider_key] = status.healthy

    log.info("provider_health_recorded", results=results)
    return results


@celery_app.task
def refresh_macro() -> dict:
    """Re-fetch the curated macro series."""
    from app.db.session import session_scope
    from app.ingestion.macro import ingest_curated_series
    from app.ingestion.sources import get_data_source
    from app.providers.fred import FredProvider

    with session_scope() as session:
        source = get_data_source(session, "fred")
        if source is None:
            return {"status": "skipped", "reason": "the FRED data source is not registered"}
        provider = FredProvider()
        try:
            reports = ingest_curated_series(
                session, provider=provider, data_source=source
            )
        finally:
            provider.close()

    return {
        "series": len(reports),
        "observations_written": sum(r.observations_written for r in reports),
        "revisions_written": sum(r.revisions_written for r in reports),
        "has_vintages": any(r.has_vintages for r in reports),
    }
