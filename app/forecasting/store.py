"""Forecast persistence with enforced immutability.

A forecast that can be edited after the outcome is known is not a forecast. The
guarantee is enforced in three independent places, because a single convention
would eventually be bypassed:

1. **No update path.** This module offers ``store_forecast`` and
   ``record_outcome``; there is no function that modifies a stored prediction.
2. **An ORM event listener** raises ``ImmutabilityError`` on any attempt to
   flush a change to a ``Forecast`` row, including from unrelated code.
3. **A test** asserts both, so removing the listener breaks the build.

Re-running the same model at the same cutoff is idempotent by uniqueness on
(instrument, model, target, horizon, cutoff): the second call returns the
existing row rather than creating a second vintage of the same prediction.
"""

from __future__ import annotations

import datetime as dt
import decimal
import uuid

from sqlalchemy import event, select
from sqlalchemy.orm import Session

from app.core.errors import ImmutabilityError, NotFoundError
from app.core.logging import get_logger
from app.core.timeutil import now_utc
from app.db.models.enums import DataClassification
from app.db.models.research import Forecast, ForecastOutcome, ModelDefinition
from app.forecasting.base import ForecastContext, ForecastDistribution, Forecaster
from app.forecasting.evaluation import crps_from_quantiles, interval_hits, pit_value

log = get_logger(__name__)

# Attributes a resolved forecast may have set once, by record_outcome.
# ``evaluated`` is a one-way latch; ``outcome`` is the relationship to the
# score row, which carries no predictive content of its own.
_ALLOWED_POST_WRITE_FIELDS = frozenset({"evaluated", "outcome"})


@event.listens_for(Session, "before_flush")
def _block_forecast_mutation(session: Session, flush_context, instances) -> None:
    """Refuse any in-place change to a stored forecast.

    ``evaluated`` is the single exception: it is a one-way latch set when the
    outcome is recorded, and it carries no predictive content. Everything else
    -- the distribution, the cutoff, the assumptions -- is frozen.
    """
    for obj in session.dirty:
        if not isinstance(obj, Forecast):
            continue
        state = obj._sa_instance_state
        changed = {
            attr.key
            for attr in state.attrs
            if attr.history.has_changes()
        }
        illegal = changed - _ALLOWED_POST_WRITE_FIELDS
        if illegal:
            raise ImmutabilityError(
                f"forecast {obj.id} cannot be modified after creation; attempted to "
                f"change {sorted(illegal)}. A prediction that can be edited after "
                "the outcome is known is not a prediction.",
                details={"forecast_id": str(obj.id), "fields": sorted(illegal)},
            )

    for obj in session.deleted:
        if isinstance(obj, Forecast):
            raise ImmutabilityError(
                f"forecast {obj.id} cannot be deleted; the prediction record is "
                "append-only so that historical performance cannot be curated.",
                details={"forecast_id": str(obj.id)},
            )


def ensure_model_definition(session: Session, model: Forecaster) -> ModelDefinition:
    """Create or refresh the durable record of a model version."""
    existing = session.execute(
        select(ModelDefinition).where(
            ModelDefinition.key == model.spec.key,
            ModelDefinition.version == model.spec.version,
        )
    ).scalars().first()

    fields = {
        "family": model.spec.family,
        "display_name": model.spec.display_name,
        "description": model.spec.description,
        "target": model.spec.target,
        "supported_horizons": list(model.spec.supported_horizons),
        "hyperparameters": model.spec.hyperparameters,
        "is_baseline": model.spec.is_baseline,
        "lifecycle": model.spec.lifecycle,
        "lifecycle_reason": model.spec.lifecycle_reason,
        "limitations": list(model.spec.limitations),
        "min_observations": model.spec.min_observations,
    }
    if existing is None:
        definition = ModelDefinition(
            key=model.spec.key, version=model.spec.version, **fields
        )
        session.add(definition)
        session.flush()
        return definition

    for key, value in fields.items():
        setattr(existing, key, value)
    session.flush()
    return existing


def find_forecast(
    session: Session,
    *,
    instrument_id: uuid.UUID,
    model_definition_id: uuid.UUID,
    target: str,
    horizon_label: str,
    information_cutoff: dt.datetime,
) -> Forecast | None:
    return session.execute(
        select(Forecast).where(
            Forecast.instrument_id == instrument_id,
            Forecast.model_definition_id == model_definition_id,
            Forecast.target == target,
            Forecast.horizon_label == horizon_label,
            Forecast.information_cutoff == information_cutoff,
        )
    ).scalars().first()


def store_forecast(
    session: Session,
    *,
    instrument_id: uuid.UUID,
    model: Forecaster,
    context: ForecastContext,
    distribution: ForecastDistribution,
    regime_label: str | None = None,
    dataset_version: str | None = None,
    classification: str = DataClassification.PRODUCTION,
) -> Forecast:
    """Persist a forecast, or return the identical one already stored.

    Validates the distribution before writing. A malformed distribution that
    reached storage would be permanently uncorrectable, since nothing may
    later edit it.
    """
    distribution.validate()

    if context.information_cutoff is None:
        raise ValueError(
            "a forecast cannot be stored without an information cutoff; without it "
            "the prediction is unfalsifiable"
        )

    definition = ensure_model_definition(session, model)
    existing = find_forecast(
        session,
        instrument_id=instrument_id,
        model_definition_id=definition.id,
        target=distribution.target,
        horizon_label=distribution.horizon_label,
        information_cutoff=context.information_cutoff,
    )
    if existing is not None:
        # Idempotent: the same model at the same cutoff is the same prediction.
        log.info(
            "forecast_already_stored",
            forecast_id=str(existing.id),
            model=model.spec.key,
            cutoff=context.information_cutoff.isoformat(),
        )
        return existing

    base_price = (
        decimal.Decimal(repr(context.close[-1])) if context.close else None
    )
    forecast = Forecast(
        instrument_id=instrument_id,
        model_definition_id=definition.id,
        created_at=now_utc(),
        information_cutoff=context.information_cutoff,
        resolves_at=model.resolves_at(context, distribution.horizon_label),
        target=distribution.target,
        horizon_label=distribution.horizon_label,
        horizon_bars=distribution.horizon_bars,
        resolution=context.resolution,
        point_estimate=distribution.point_estimate,
        quantiles={str(q): v for q, v in sorted(distribution.quantiles.items())},
        prob_positive=distribution.prob_positive,
        predicted_volatility=distribution.predicted_volatility,
        distribution_family=distribution.distribution_family,
        distribution_params=distribution.distribution_params,
        base_price=base_price,
        contributing_factors=distribution.contributing_factors,
        assumptions=list(model.spec.assumptions),
        invalidation_conditions=list(model.spec.invalidation_conditions),
        limitations=list(model.spec.limitations) + distribution.warnings,
        observations_used=distribution.observations_used,
        input_fingerprint=context.fingerprint(),
        data_quality_flags=list(context.data_quality_flags),
        dataset_version=dataset_version,
        regime_at_cutoff=regime_label or context.regime_label,
        evaluated=False,
        classification=classification,
    )
    session.add(forecast)
    session.flush()
    return forecast


def record_outcome(
    session: Session,
    forecast: Forecast,
    *,
    realized_value: float,
    realized_at: dt.datetime,
    benchmark_errors: dict | None = None,
    notes: str | None = None,
) -> ForecastOutcome:
    """Score a forecast against its realised outcome.

    Refuses to write before ``resolves_at``: scoring a forecast whose horizon
    has not closed would compare it to an incomplete outcome and flatter it.
    """
    if forecast.outcome is not None:
        raise ImmutabilityError(
            f"forecast {forecast.id} has already been evaluated; re-scoring it "
            "would allow historical performance to be rewritten.",
            details={"forecast_id": str(forecast.id)},
        )

    now = now_utc()
    if realized_at < forecast.resolves_at:
        raise ValueError(
            f"the outcome is timestamped {realized_at.isoformat()} but the forecast "
            f"does not resolve until {forecast.resolves_at.isoformat()}; scoring it "
            "now would use an incomplete horizon"
        )

    quantiles = {float(k): float(v) for k, v in (forecast.quantiles or {}).items()}
    reconstructed = ForecastDistribution(
        target=forecast.target,
        horizon_label=forecast.horizon_label,
        horizon_bars=forecast.horizon_bars,
        point_estimate=forecast.point_estimate or 0.0,
        quantiles=quantiles,
        predicted_volatility=forecast.predicted_volatility,
        prob_positive=forecast.prob_positive,
        distribution_family=forecast.distribution_family or "empirical",
        distribution_params=forecast.distribution_params or {},
    )

    error = realized_value - (forecast.point_estimate or 0.0)
    direction = None
    if forecast.point_estimate and abs(forecast.point_estimate) > 1e-12:
        direction = (forecast.point_estimate > 0) == (realized_value > 0)

    outcome = ForecastOutcome(
        forecast_id=forecast.id,
        evaluated_at=now,
        realized_value=realized_value,
        realized_at=realized_at,
        error=error,
        absolute_error=abs(error),
        squared_error=error**2,
        crps=crps_from_quantiles(quantiles, realized_value) if quantiles else None,
        pit_value=pit_value(reconstructed, realized_value),
        interval_coverage=interval_hits(reconstructed, realized_value),
        direction_correct=direction,
        benchmark_errors=benchmark_errors or {},
        notes=notes,
    )
    session.add(outcome)
    # Assign the relationship, not just the foreign key. Reading
    # ``forecast.outcome`` above lazy-loads and caches None; without this the
    # cached None survives the flush and a second call to this function would
    # sail past the guard and fail on the unique constraint instead, leaving
    # the transaction unusable.
    forecast.outcome = outcome
    # The one permitted post-write change: a one-way latch with no predictive
    # content, so the evaluation sweep need not re-scan scored forecasts.
    forecast.evaluated = True
    session.flush()
    return outcome


def due_for_evaluation(
    session: Session, *, as_of: dt.datetime | None = None, limit: int = 500
) -> list[Forecast]:
    """Forecasts whose horizon has closed and which have no outcome yet."""
    cutoff = as_of or now_utc()
    return list(
        session.execute(
            select(Forecast)
            .where(Forecast.resolves_at <= cutoff, Forecast.evaluated.is_(False))
            .order_by(Forecast.resolves_at)
            .limit(limit)
        ).scalars().all()
    )


def forecast_history(
    session: Session,
    *,
    instrument_id: uuid.UUID,
    model_key: str | None = None,
    horizon_label: str | None = None,
    include_unresolved: bool = True,
    limit: int = 200,
) -> list[Forecast]:
    """Past forecasts for an instrument, newest first."""
    stmt = select(Forecast).where(Forecast.instrument_id == instrument_id)
    if model_key:
        stmt = stmt.join(ModelDefinition).where(ModelDefinition.key == model_key)
    if horizon_label:
        stmt = stmt.where(Forecast.horizon_label == horizon_label)
    if not include_unresolved:
        stmt = stmt.where(Forecast.evaluated.is_(True))
    stmt = stmt.order_by(Forecast.information_cutoff.desc()).limit(limit)
    return list(session.execute(stmt).scalars().all())


def get_forecast(session: Session, forecast_id: uuid.UUID) -> Forecast:
    found = session.get(Forecast, forecast_id)
    if found is None:
        raise NotFoundError(f"forecast {forecast_id} not found")
    return found
