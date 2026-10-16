"""Forecast creation, retrieval and validation endpoints."""

from __future__ import annotations

import datetime as dt
import uuid

from fastapi import APIRouter, Body, Depends, Query
from pydantic import Field
from sqlalchemy.orm import Session

from app.api.deps import CommonGuards, get_db, parse_as_of
from app.api.schemas.common import ApiModel
from app.core.errors import UnsupportedError
from app.db.models.enums import AdjustmentBasis
from app.forecasting.base import (
    HORIZONS,
    ForecastContext,
    all_models,
    get_model,
    model_registry_snapshot,
)
from app.forecasting.evaluation import WalkForwardConfig, walk_forward
from app.forecasting.store import forecast_history, get_forecast, store_forecast
from app.pit.availability import AsOf
from app.services.data_access import load_series, resolve_instrument

router = APIRouter(prefix="/forecasts", tags=["forecasts"], dependencies=CommonGuards)


class ModelInfo(ApiModel):
    key: str
    version: str
    family: str
    display_name: str
    description: str
    target: str
    supported_horizons: list[str]
    min_observations: int
    is_baseline: bool
    lifecycle: str
    lifecycle_reason: str
    limitations: list[str]
    assumptions: list[str]
    invalidation_conditions: list[str]
    hyperparameters: dict


class ForecastOut(ApiModel):
    id: uuid.UUID | None = None
    instrument_id: uuid.UUID
    symbol: str
    model_key: str
    model_version: str
    is_baseline: bool
    lifecycle: str
    target: str
    horizon_label: str
    horizon_bars: int
    information_cutoff: str
    resolves_at: str
    point_estimate: float | None
    quantiles: dict[str, float]
    price_levels: dict[str, float] = Field(default_factory=dict)
    base_price: float | None = None
    prob_positive: float | None = None
    predicted_volatility: float | None = None
    distribution_family: str
    contributing_factors: list[dict] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    observations_used: int
    input_fingerprint: str
    stored: bool
    # Present when the model has been evaluated out of sample.
    validation: dict | None = None
    disclaimer: str


FORECAST_DISCLAIMER = (
    "A forecast is a probability distribution produced by a stated model under "
    "stated assumptions, not a prediction of what will happen. The interval "
    "matters more than the central estimate. No model here has demonstrated "
    "profitable predictive performance, and several are reference baselines "
    "whose purpose is to be beaten."
)


@router.get("/models", response_model=list[ModelInfo], summary="Available models")
def models() -> list[ModelInfo]:
    """Registered forecasting models with their assumptions and limitations."""
    return [ModelInfo(**m) for m in model_registry_snapshot()]


@router.get("/horizons", summary="Supported horizons")
def horizons() -> dict:
    return {
        "horizons": {
            label: {
                "bars": bars,
                "note": (
                    "A model fitted or validated for one horizon does not transfer "
                    "to another; each model declares the horizons it supports."
                ),
            }
            for label, bars in HORIZONS.items()
        }
    }


class ForecastRequest(ApiModel):
    model_key: str = "random_walk"
    horizon: str = "5d"
    lookback_days: int = 1095
    store: bool = Field(
        default=False,
        description=(
            "Persist the forecast immutably so it can be evaluated later. A "
            "stored forecast can never be edited or deleted."
        ),
    )
    include_validation: bool = Field(
        default=True,
        description="Run a walk-forward evaluation of this model on this instrument.",
    )


@router.post("/{identifier}", response_model=ForecastOut, summary="Create a forecast")
def create(
    identifier: str,
    request: ForecastRequest = Body(),
    as_of: AsOf = Depends(parse_as_of),
    db: Session = Depends(get_db),
) -> ForecastOut:
    """Produce a forecast distribution for a horizon.

    When ``include_validation`` is set, the response also carries this model's
    out-of-sample walk-forward performance on this instrument, so the forecast
    is never shown without evidence about whether the model works.
    """
    if request.horizon not in HORIZONS:
        raise UnsupportedError(
            f"unknown horizon {request.horizon!r}; supported: {sorted(HORIZONS)}"
        )
    model = get_model(request.model_key)
    instrument = resolve_instrument(db, identifier, as_of=as_of)

    start = as_of.cutoff - dt.timedelta(days=request.lookback_days)
    series = load_series(
        db, instrument, as_of=as_of, start=start, end=as_of.cutoff,
        resolution="1d", adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
        min_bars=model.spec.min_observations,
    )

    quality_flags = [f"{i['kind']}@{str(i['at'])[:10]}" for i in series.quality_issues]
    if series.gaps:
        quality_flags.append(
            f"{sum(g['expected_sessions'] for g in series.gaps)}_missing_sessions"
        )

    context = ForecastContext(
        timestamps=tuple(series.timestamps),
        close=tuple(series.close),
        high=tuple(series.high),
        low=tuple(series.low),
        volume=tuple(series.volume),
        # The cutoff is the last observation, not "now": a model must not be
        # credited with seeing a session that has not closed.
        information_cutoff=series.timestamps[-1],
        resolution="1d",
        data_quality_flags=tuple(quality_flags),
    )

    distribution = model.forecast(context, request.horizon)
    base_price = series.close[-1]

    stored_id: uuid.UUID | None = None
    if request.store:
        record = store_forecast(
            db, instrument_id=instrument.id, model=model,
            context=context, distribution=distribution,
        )
        db.commit()
        stored_id = record.id

    validation: dict | None = None
    if request.include_validation:
        validation = _validation_for(series, model, request.horizon)

    return ForecastOut(
        id=stored_id,
        instrument_id=instrument.id,
        symbol=instrument.symbol,
        model_key=model.spec.key,
        model_version=model.spec.version,
        is_baseline=model.spec.is_baseline,
        lifecycle=model.spec.lifecycle,
        target=distribution.target,
        horizon_label=distribution.horizon_label,
        horizon_bars=distribution.horizon_bars,
        information_cutoff=context.information_cutoff.isoformat(),
        resolves_at=model.resolves_at(context, request.horizon).isoformat(),
        point_estimate=distribution.point_estimate,
        quantiles={str(q): v for q, v in sorted(distribution.quantiles.items())},
        price_levels=distribution.as_price_levels(base_price),
        base_price=base_price,
        prob_positive=distribution.prob_positive,
        predicted_volatility=distribution.predicted_volatility,
        distribution_family=distribution.distribution_family,
        contributing_factors=distribution.contributing_factors,
        assumptions=list(model.spec.assumptions),
        invalidation_conditions=list(model.spec.invalidation_conditions),
        limitations=list(model.spec.limitations),
        warnings=distribution.warnings + series.coverage_notes,
        observations_used=distribution.observations_used,
        input_fingerprint=context.fingerprint(),
        stored=stored_id is not None,
        validation=validation,
        disclaimer=FORECAST_DISCLAIMER,
    )


def _validation_for(series, model, horizon: str) -> dict:
    """Walk-forward this model on this instrument, against the random walk."""
    baseline = get_model("random_walk")
    config = WalkForwardConfig(
        min_train_bars=max(model.spec.min_observations, 250),
        step_bars=max(HORIZONS[horizon] // 2, 1),
        horizon_label=horizon,
        max_forecasts=200,
    )
    try:
        result = walk_forward(
            model,
            timestamps=series.timestamps,
            close=series.close,
            config=config,
            baseline=None if model.spec.key == baseline.spec.key else baseline,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": (
                f"this model could not be validated on {series.instrument.symbol}'s "
                f"history: {exc}"
            ),
        }
    return {
        "available": True,
        "metrics": result.metrics.as_dict(),
        "execution_assumptions": result.execution_assumptions,
        "limitations": result.limitations,
        "scheme": "walk_forward",
        "note": (
            "These metrics come from forecasts made with the same information "
            "cutoff discipline as the live forecast, scored only after their "
            "horizon closed."
        ),
    }


class CompareResponse(ApiModel):
    instrument_id: uuid.UUID
    symbol: str
    horizon: str
    as_of: str
    results: list[dict]
    note: str


@router.get("/{identifier}/compare", response_model=CompareResponse, summary="Compare models")
def compare(
    identifier: str,
    horizon: str = Query(default="5d"),
    lookback_days: int = Query(default=1825, ge=365, le=7300),
    as_of: AsOf = Depends(parse_as_of),
    db: Session = Depends(get_db),
) -> CompareResponse:
    """Walk-forward every applicable model on this instrument, side by side.

    This is the endpoint that answers "does any of this work?" -- and for most
    instruments and horizons the honest answer is that nothing beats the random
    walk by a meaningful margin.
    """
    if horizon not in HORIZONS:
        raise UnsupportedError(f"unknown horizon {horizon!r}")

    instrument = resolve_instrument(db, identifier, as_of=as_of)
    start = as_of.cutoff - dt.timedelta(days=lookback_days)
    series = load_series(
        db, instrument, as_of=as_of, start=start, end=as_of.cutoff,
        resolution="1d", adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
    )

    baseline = get_model("random_walk")
    out: list[dict] = []
    for model in all_models():
        if horizon not in model.spec.supported_horizons:
            out.append({
                "model_key": model.spec.key,
                "available": False,
                "reason": f"this model does not support the {horizon} horizon",
            })
            continue
        config = WalkForwardConfig(
            min_train_bars=max(model.spec.min_observations, 250),
            step_bars=max(HORIZONS[horizon] // 2, 1),
            horizon_label=horizon,
            max_forecasts=300,
        )
        try:
            result = walk_forward(
                model,
                timestamps=series.timestamps,
                close=series.close,
                config=config,
                baseline=None if model.spec.key == baseline.spec.key else baseline,
            )
        except Exception as exc:
            out.append({
                "model_key": model.spec.key,
                "available": False,
                "reason": str(exc),
            })
            continue
        out.append({
            "model_key": model.spec.key,
            "model_version": model.spec.version,
            "is_baseline": model.spec.is_baseline,
            "lifecycle": model.spec.lifecycle,
            "available": True,
            "metrics": result.metrics.as_dict(),
            "execution_assumptions": result.execution_assumptions,
            "limitations": result.limitations,
        })

    return CompareResponse(
        instrument_id=instrument.id,
        symbol=instrument.symbol,
        horizon=horizon,
        as_of=as_of.cutoff.isoformat(),
        results=out,
        note=(
            "Models are ranked by CRPS skill against the random-walk baseline. A "
            "skill score at or below zero means the model has no demonstrated "
            "advantage over assuming no drift. All results are gross of execution "
            "costs unless the assumptions say otherwise."
        ),
    )


class ForecastRecordOut(ApiModel):
    id: uuid.UUID
    model_key: str
    model_version: str
    target: str
    horizon_label: str
    information_cutoff: str
    resolves_at: str
    created_at: str
    point_estimate: float | None
    quantiles: dict[str, float]
    prob_positive: float | None
    evaluated: bool
    outcome: dict | None = None


@router.get(
    "/{identifier}/history",
    response_model=list[ForecastRecordOut],
    summary="Stored forecasts and their outcomes",
)
def history(
    identifier: str,
    model_key: str | None = Query(default=None),
    horizon: str | None = Query(default=None),
    resolved_only: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[ForecastRecordOut]:
    """Past forecasts with their realised outcomes.

    This is the audit trail: each entry was stored before its horizon closed
    and scored afterwards, with no opportunity to revise it in between.
    """
    instrument = resolve_instrument(db, identifier)
    records = forecast_history(
        db,
        instrument_id=instrument.id,
        model_key=model_key,
        horizon_label=horizon,
        include_unresolved=not resolved_only,
        limit=limit,
    )
    out: list[ForecastRecordOut] = []
    for record in records:
        outcome = None
        if record.outcome is not None:
            o = record.outcome
            outcome = {
                "realized_value": o.realized_value,
                "realized_at": o.realized_at.isoformat() if o.realized_at else None,
                "error": o.error,
                "absolute_error": o.absolute_error,
                "crps": o.crps,
                "pit_value": o.pit_value,
                "interval_coverage": o.interval_coverage,
                "direction_correct": o.direction_correct,
                "evaluated_at": o.evaluated_at.isoformat(),
            }
        out.append(
            ForecastRecordOut(
                id=record.id,
                model_key=record.model.key,
                model_version=record.model.version,
                target=record.target,
                horizon_label=record.horizon_label,
                information_cutoff=record.information_cutoff.isoformat(),
                resolves_at=record.resolves_at.isoformat(),
                created_at=record.created_at.isoformat(),
                point_estimate=record.point_estimate,
                quantiles={k: float(v) for k, v in (record.quantiles or {}).items()},
                prob_positive=record.prob_positive,
                evaluated=record.evaluated,
                outcome=outcome,
            )
        )
    return out


@router.get("/record/{forecast_id}", response_model=ForecastRecordOut, summary="One forecast")
def record(forecast_id: uuid.UUID, db: Session = Depends(get_db)) -> ForecastRecordOut:
    found = get_forecast(db, forecast_id)
    outcome = None
    if found.outcome is not None:
        o = found.outcome
        outcome = {
            "realized_value": o.realized_value,
            "crps": o.crps,
            "pit_value": o.pit_value,
            "absolute_error": o.absolute_error,
            "direction_correct": o.direction_correct,
            "interval_coverage": o.interval_coverage,
        }
    return ForecastRecordOut(
        id=found.id,
        model_key=found.model.key,
        model_version=found.model.version,
        target=found.target,
        horizon_label=found.horizon_label,
        information_cutoff=found.information_cutoff.isoformat(),
        resolves_at=found.resolves_at.isoformat(),
        created_at=found.created_at.isoformat(),
        point_estimate=found.point_estimate,
        quantiles={k: float(v) for k, v in (found.quantiles or {}).items()},
        prob_positive=found.prob_positive,
        evaluated=found.evaluated,
        outcome=outcome,
    )
