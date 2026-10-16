"""Forecasting interfaces.

Design decisions that the rest of the engine depends on:

* **A forecast is a distribution, not a number.** Every model returns
  quantiles. A point estimate alone cannot be scored honestly, cannot express
  uncertainty, and invites the reader to treat it as a prediction of fact.
* **The information cutoff is part of the forecast.** A model receives a
  ``ForecastContext`` whose data was already filtered as-of that instant, and
  the cutoff is stored with the result. Without it, later evaluation is
  unfalsifiable.
* **Baselines are first-class models, not a benchmark afterthought.** A model
  that cannot beat a random walk has no demonstrated value, so the random walk
  is registered, run and scored exactly like everything else.
* **Models declare the conditions that invalidate them.** Those conditions are
  stored on the forecast so a reader can check whether it still applies.
"""

from __future__ import annotations

import abc
import datetime as dt
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import InsufficientDataError
from app.db.models.enums import ForecastTarget, ModelLifecycle

# Horizon label -> number of bars at the forecast's resolution.
HORIZONS: dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "1mo": 21,
    "3mo": 63,
    "6mo": 126,
    "1y": 252,
}

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    version: str
    family: str
    display_name: str
    description: str
    target: str
    supported_horizons: tuple[str, ...]
    min_observations: int
    limitations: tuple[str, ...]
    invalidation_conditions: tuple[str, ...]
    assumptions: tuple[str, ...]
    is_baseline: bool = False
    lifecycle: str = ModelLifecycle.EXPERIMENTAL
    lifecycle_reason: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    supported_resolutions: tuple[str, ...] = ("1d",)


@dataclass
class ForecastContext:
    """Everything a model may see.

    Already point-in-time filtered by the caller. ``information_cutoff`` is
    recorded so a model cannot be handed future data without the forecast
    saying it was.
    """

    timestamps: tuple[dt.datetime, ...]
    close: tuple[float, ...]
    high: tuple[float, ...] = ()
    low: tuple[float, ...] = ()
    volume: tuple[float | None, ...] = ()
    benchmark_close: tuple[float, ...] = ()
    information_cutoff: dt.datetime | None = None
    resolution: str = "1d"
    trading_days_per_year: int = 252
    data_quality_flags: tuple[str, ...] = ()
    regime_label: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.close)

    def fingerprint(self) -> str:
        payload = {
            "n": len(self.close),
            "first": self.timestamps[0].isoformat() if self.timestamps else None,
            "last": self.timestamps[-1].isoformat() if self.timestamps else None,
            "close": [f"{c:.10g}" for c in self.close],
            "resolution": self.resolution,
            "cutoff": (
                self.information_cutoff.isoformat() if self.information_cutoff else None
            ),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:32]


@dataclass
class ForecastDistribution:
    """A model's output: a distribution over the target, plus its reasoning."""

    target: str
    horizon_label: str
    horizon_bars: int
    point_estimate: float
    quantiles: dict[float, float]
    predicted_volatility: float | None = None
    prob_positive: float | None = None
    distribution_family: str = "empirical"
    distribution_params: dict[str, float] = field(default_factory=dict)
    contributing_factors: list[dict] = field(default_factory=list)
    observations_used: int = 0
    warnings: list[str] = field(default_factory=list)

    def quantile(self, q: float) -> float | None:
        return self.quantiles.get(q)

    def as_price_levels(self, base_price: float) -> dict[str, float]:
        """Translate log-return quantiles to price levels for display."""
        if self.target != ForecastTarget.LOG_RETURN:
            return {}
        return {str(q): base_price * math.exp(v) for q, v in sorted(self.quantiles.items())}

    def validate(self) -> None:
        """Reject a malformed distribution before it is ever stored.

        A non-monotonic quantile function is not a distribution. Storing one
        would make every downstream interval and probability meaningless, and
        the error would be invisible in a chart.
        """
        if not self.quantiles:
            raise ValueError("a forecast must provide quantiles, not only a point estimate")
        ordered = sorted(self.quantiles.items())
        for (q1, v1), (q2, v2) in itertools.pairwise(ordered):
            if v2 < v1:
                raise ValueError(
                    f"quantiles are not monotonic: q={q1} gives {v1} but q={q2} "
                    f"gives {v2}. A non-monotonic quantile function is not a "
                    "distribution and cannot be scored."
                )
        if self.prob_positive is not None and not 0.0 <= self.prob_positive <= 1.0:
            raise ValueError(f"prob_positive must be in [0, 1], got {self.prob_positive}")


class Forecaster(abc.ABC):
    """Base class for every forecasting model."""

    spec: ModelSpec

    @property
    def key(self) -> str:
        return self.spec.key

    def check_context(self, context: ForecastContext, horizon: str) -> int:
        """Validate the context, returning the horizon in bars."""
        if horizon not in self.spec.supported_horizons:
            raise InsufficientDataError(
                f"{self.spec.display_name} does not support the {horizon} horizon; "
                f"supported: {', '.join(self.spec.supported_horizons)}. A model "
                "fitted for one horizon does not transfer to another.",
                details={"model": self.spec.key, "horizon": horizon},
            )
        bars = HORIZONS[horizon]
        required = self.spec.min_observations
        if len(context) < required:
            raise InsufficientDataError(
                f"{self.spec.display_name} needs {required} observations for a "
                f"{horizon} forecast but received {len(context)}",
                details={
                    "model": self.spec.key,
                    "required": required,
                    "available": len(context),
                },
            )
        if context.information_cutoff is not None and context.timestamps:
            last = context.timestamps[-1]
            if last > context.information_cutoff:
                raise InsufficientDataError(
                    f"the context's last observation {last.isoformat()} is after its "
                    f"information cutoff {context.information_cutoff.isoformat()}; "
                    "the model would be fitted on data it could not have seen",
                    details={"model": self.spec.key},
                )
        return bars

    @abc.abstractmethod
    def forecast(self, context: ForecastContext, horizon: str) -> ForecastDistribution:
        """Produce a distribution for the horizon."""

    def resolves_at(self, context: ForecastContext, horizon: str) -> dt.datetime:
        """When the outcome becomes observable.

        Calendar days are used rather than bar counts because the resolution
        instant must be a wall-clock time an evaluator can wait for.
        """
        bars = HORIZONS[horizon]
        last = context.timestamps[-1] if context.timestamps else context.information_cutoff
        assert last is not None
        calendar_days = math.ceil(bars * 7 / 5) + 1
        return last + dt.timedelta(days=calendar_days)


_MODELS: dict[str, Forecaster] = {}


def register_model(model: Forecaster) -> Forecaster:
    _MODELS[model.spec.key] = model
    return model


def get_model(key: str) -> Forecaster:
    _ensure_models_loaded()
    from app.core.errors import NotFoundError

    try:
        return _MODELS[key]
    except KeyError as exc:
        raise NotFoundError(
            f"unknown forecasting model {key!r}; registered: {sorted(_MODELS)}"
        ) from exc


def all_models() -> list[Forecaster]:
    _ensure_models_loaded()
    return sorted(_MODELS.values(), key=lambda m: (not m.spec.is_baseline, m.spec.key))


def baseline_models() -> list[Forecaster]:
    return [m for m in all_models() if m.spec.is_baseline]


_MODELS_LOADED = False


def _ensure_models_loaded() -> None:
    global _MODELS_LOADED
    if _MODELS_LOADED:
        return
    _MODELS_LOADED = True
    from app.forecasting import baselines  # noqa: F401


def model_registry_snapshot() -> list[dict]:
    return [
        {
            "key": m.spec.key,
            "version": m.spec.version,
            "family": m.spec.family,
            "display_name": m.spec.display_name,
            "description": m.spec.description,
            "target": m.spec.target,
            "supported_horizons": list(m.spec.supported_horizons),
            "min_observations": m.spec.min_observations,
            "is_baseline": m.spec.is_baseline,
            "lifecycle": m.spec.lifecycle,
            "lifecycle_reason": m.spec.lifecycle_reason,
            "limitations": list(m.spec.limitations),
            "assumptions": list(m.spec.assumptions),
            "invalidation_conditions": list(m.spec.invalidation_conditions),
            "hyperparameters": m.spec.hyperparameters,
        }
        for m in all_models()
    ]
