"""Forecast scoring and walk-forward evaluation.

Scoring a probabilistic forecast by directional accuracy is the error this
module exists to prevent. A model that says "+0.1% with a 20% standard
deviation" is not making a directional call, and marking it right or wrong on
the sign of the outcome measures almost nothing.

What is measured instead:

``CRPS``
    Continuous Ranked Probability Score: the integral of the squared difference
    between the forecast CDF and the step function at the outcome. It rewards
    both sharpness and calibration, reduces to absolute error for a point
    forecast, and is the primary metric here.
``PIT``
    Probability Integral Transform: the forecast CDF evaluated at the outcome.
    Over many forecasts these must be uniform on [0,1]. A histogram piled at
    the edges means the intervals are too narrow -- the most common and most
    dangerous miscalibration, because it makes a model look confident.
``Interval coverage``
    The realised hit rate of each nominal interval. An 80% interval containing
    the outcome 55% of the time is a specific, reportable failure.
``Skill score``
    Improvement in CRPS over a named baseline. A positive raw metric means
    nothing; beating the random walk means something.

Walk-forward is the only evaluation scheme offered. A single train/test split
invites tuning on the test set, and k-fold cross-validation on a time series
trains on the future, which is simply invalid.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats

from app.core.errors import InsufficientDataError
from app.core.logging import get_logger
from app.forecasting.base import (
    HORIZONS,
    ForecastContext,
    ForecastDistribution,
    Forecaster,
)

log = get_logger(__name__)

NOMINAL_INTERVALS: tuple[tuple[float, float], ...] = (
    (0.25, 0.75),
    (0.10, 0.90),
    (0.05, 0.95),
)


def crps_from_quantiles(quantiles: dict[float, float], outcome: float) -> float:
    """CRPS approximated from a discrete quantile set.

    Uses the quantile-loss (pinball) representation: CRPS equals twice the
    integral of the pinball loss over the quantile levels. With a finite
    quantile set this is a trapezoidal approximation, which is why the result
    is comparable across models only when they report the same levels -- and
    they do, via ``DEFAULT_QUANTILES``.
    """
    if not quantiles:
        raise ValueError("CRPS needs at least one quantile")
    levels = sorted(quantiles)
    losses = []
    for q in levels:
        predicted = quantiles[q]
        if outcome >= predicted:
            losses.append(q * (outcome - predicted))
        else:
            losses.append((1.0 - q) * (predicted - outcome))

    if len(levels) == 1:
        return 2.0 * losses[0]

    # Trapezoidal integration over the quantile levels.
    total = 0.0
    for i in range(len(levels) - 1):
        width = levels[i + 1] - levels[i]
        total += 0.5 * (losses[i] + losses[i + 1]) * width
    # Scale so the result is on the same footing as mean absolute error when
    # the forecast is a point mass.
    span = levels[-1] - levels[0]
    return 2.0 * total / span if span > 0 else 2.0 * statistics.fmean(losses)


def pit_value(distribution: ForecastDistribution, outcome: float) -> float | None:
    """Forecast CDF evaluated at the outcome.

    Uses the parametric CDF where the family is known, because interpolating
    between seven quantiles cannot resolve the tails -- and the tails are
    exactly where miscalibration matters.
    """
    family = distribution.distribution_family
    params = distribution.distribution_params

    if family == "normal" and "sigma" in params:
        sigma = params["sigma"]
        if sigma <= 0:
            return None
        return float(scipy_stats.norm.cdf(outcome, loc=params.get("mu", 0.0), scale=sigma))

    if family == "student_t" and {"df", "scale"} <= set(params):
        if params["scale"] <= 0:
            return None
        return float(
            scipy_stats.t.cdf(
                outcome, params["df"], loc=params.get("loc", 0.0), scale=params["scale"]
            )
        )

    # Empirical fallback: linear interpolation across the quantile set, with
    # the tails capped rather than extrapolated.
    levels = sorted(distribution.quantiles)
    values = [distribution.quantiles[q] for q in levels]
    if outcome <= values[0]:
        return levels[0] / 2.0
    if outcome >= values[-1]:
        return 1.0 - (1.0 - levels[-1]) / 2.0
    return float(np.interp(outcome, values, levels))


def interval_hits(
    distribution: ForecastDistribution, outcome: float
) -> dict[str, bool]:
    """Whether the outcome fell inside each nominal interval."""
    out: dict[str, bool] = {}
    for low_q, high_q in NOMINAL_INTERVALS:
        low = distribution.quantiles.get(low_q)
        high = distribution.quantiles.get(high_q)
        if low is None or high is None:
            continue
        nominal = round((high_q - low_q) * 100)
        out[f"{nominal}%"] = bool(low <= outcome <= high)
    return out


@dataclass
class ScoredForecast:
    """One forecast and its realised outcome."""

    information_cutoff: dt.datetime
    resolved_at: dt.datetime
    horizon_label: str
    predicted: ForecastDistribution
    outcome: float
    crps: float
    absolute_error: float
    squared_error: float
    pit: float | None
    interval_hits: dict[str, bool]
    direction_correct: bool | None
    regime_label: str | None = None


@dataclass
class EvaluationMetrics:
    """Aggregate scores over a set of resolved forecasts."""

    model_key: str
    model_version: str
    horizon_label: str
    n_forecasts: int
    mean_crps: float | None = None
    mean_absolute_error: float | None = None
    root_mean_squared_error: float | None = None
    directional_accuracy: float | None = None
    interval_coverage: dict[str, float] = field(default_factory=dict)
    pit_uniformity_p: float | None = None
    pit_histogram: list[int] = field(default_factory=list)
    calibration_verdict: str = "not_assessed"
    skill_vs_baseline: float | None = None
    baseline_key: str | None = None
    per_regime: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "model_version": self.model_version,
            "horizon_label": self.horizon_label,
            "n_forecasts": self.n_forecasts,
            "mean_crps": self.mean_crps,
            "mean_absolute_error": self.mean_absolute_error,
            "root_mean_squared_error": self.root_mean_squared_error,
            "directional_accuracy": self.directional_accuracy,
            "interval_coverage": self.interval_coverage,
            "pit_uniformity_p": self.pit_uniformity_p,
            "pit_histogram": self.pit_histogram,
            "calibration_verdict": self.calibration_verdict,
            "skill_vs_baseline": self.skill_vs_baseline,
            "baseline_key": self.baseline_key,
            "per_regime": self.per_regime,
            "warnings": self.warnings,
        }


def score_forecast(
    distribution: ForecastDistribution,
    outcome: float,
    *,
    information_cutoff: dt.datetime,
    resolved_at: dt.datetime,
    regime_label: str | None = None,
) -> ScoredForecast:
    error = outcome - distribution.point_estimate

    # Direction is scored only when the model actually made a directional
    # call. A model whose point estimate is exactly zero did not, and marking
    # it right or wrong on the outcome's sign would be measuring noise.
    direction = None
    if abs(distribution.point_estimate) > 1e-12:
        direction = (distribution.point_estimate > 0) == (outcome > 0)

    return ScoredForecast(
        information_cutoff=information_cutoff,
        resolved_at=resolved_at,
        horizon_label=distribution.horizon_label,
        predicted=distribution,
        outcome=outcome,
        crps=crps_from_quantiles(distribution.quantiles, outcome),
        absolute_error=abs(error),
        squared_error=error**2,
        pit=pit_value(distribution, outcome),
        interval_hits=interval_hits(distribution, outcome),
        direction_correct=direction,
        regime_label=regime_label,
    )


def aggregate_metrics(
    scored: Sequence[ScoredForecast],
    *,
    model_key: str,
    model_version: str,
    horizon_label: str,
    baseline_scored: Sequence[ScoredForecast] | None = None,
    baseline_key: str | None = None,
) -> EvaluationMetrics:
    """Aggregate scored forecasts into reportable metrics."""
    metrics = EvaluationMetrics(
        model_key=model_key,
        model_version=model_version,
        horizon_label=horizon_label,
        n_forecasts=len(scored),
    )
    if not scored:
        metrics.warnings.append("no forecasts resolved, so no metrics are reported")
        return metrics

    metrics.mean_crps = statistics.fmean(s.crps for s in scored)
    metrics.mean_absolute_error = statistics.fmean(s.absolute_error for s in scored)
    metrics.root_mean_squared_error = math.sqrt(
        statistics.fmean(s.squared_error for s in scored)
    )

    directional = [s.direction_correct for s in scored if s.direction_correct is not None]
    if directional:
        metrics.directional_accuracy = sum(directional) / len(directional)
    else:
        metrics.warnings.append(
            "no directional accuracy is reported: this model makes no directional "
            "call, so scoring the sign of the outcome would measure nothing"
        )

    # Interval coverage.
    all_labels = {label for s in scored for label in s.interval_hits}
    for label in sorted(all_labels):
        hits = [s.interval_hits[label] for s in scored if label in s.interval_hits]
        if hits:
            metrics.interval_coverage[label] = sum(hits) / len(hits)

    # PIT uniformity: the sharpest calibration diagnostic available.
    pits = [s.pit for s in scored if s.pit is not None]
    if len(pits) >= 20:
        ks = scipy_stats.kstest(pits, "uniform")
        metrics.pit_uniformity_p = float(ks.pvalue)
        counts, _ = np.histogram(pits, bins=10, range=(0.0, 1.0))
        metrics.pit_histogram = [int(c) for c in counts]
        metrics.calibration_verdict = _calibration_verdict(
            metrics.pit_uniformity_p, metrics.interval_coverage, metrics.warnings
        )
    else:
        metrics.warnings.append(
            f"only {len(pits)} resolved forecasts carry a PIT value; at least 20 are "
            "needed before calibration can be assessed, so no verdict is given"
        )

    # Skill score against the baseline. CRPS is scale-dependent, so a raw
    # value is uninterpretable without this comparison.
    if baseline_scored:
        baseline_crps = statistics.fmean(s.crps for s in baseline_scored)
        if baseline_crps > 0:
            metrics.skill_vs_baseline = 1.0 - (metrics.mean_crps / baseline_crps)
            metrics.baseline_key = baseline_key
            if metrics.skill_vs_baseline <= 0:
                metrics.warnings.append(
                    f"this model does not beat the {baseline_key} baseline "
                    f"(CRPS skill {metrics.skill_vs_baseline:+.3f}); it has no "
                    "demonstrated advantage over it"
                )

    # Per-regime breakdown: an average over a trending and a crashing market
    # describes neither.
    by_regime: dict[str, list[ScoredForecast]] = {}
    for s in scored:
        if s.regime_label:
            by_regime.setdefault(s.regime_label, []).append(s)
    for label, group in sorted(by_regime.items()):
        metrics.per_regime[label] = {
            "n": len(group),
            "mean_crps": statistics.fmean(s.crps for s in group),
            "mean_absolute_error": statistics.fmean(s.absolute_error for s in group),
            "coverage_90": _coverage_for(group, "80%") or _coverage_for(group, "90%"),
        }

    if len(scored) < 30:
        metrics.warnings.append(
            f"metrics are computed from {len(scored)} forecasts; with fewer than "
            "about 30 they are too noisy to rank models"
        )
    return metrics


def _coverage_for(group: Sequence[ScoredForecast], label: str) -> float | None:
    hits = [s.interval_hits[label] for s in group if label in s.interval_hits]
    return (sum(hits) / len(hits)) if hits else None


def _calibration_verdict(
    pit_p: float | None, coverage: dict[str, float], warnings: list[str]
) -> str:
    """Name the specific miscalibration rather than just passing or failing."""
    if pit_p is None:
        return "not_assessed"

    worst_label = None
    worst_gap = 0.0
    for label, realised in coverage.items():
        nominal = int(label.rstrip("%")) / 100.0
        gap = realised - nominal
        if abs(gap) > abs(worst_gap):
            worst_gap = gap
            worst_label = label

    if pit_p >= 0.05 and abs(worst_gap) < 0.1:
        return "well_calibrated"

    if worst_label and worst_gap < -0.1:
        warnings.append(
            f"the nominal {worst_label} interval contained the outcome only "
            f"{coverage[worst_label]:.0%} of the time: the intervals are too "
            "narrow and the model is overconfident"
        )
        return "overconfident"

    if worst_label and worst_gap > 0.1:
        warnings.append(
            f"the nominal {worst_label} interval contained the outcome "
            f"{coverage[worst_label]:.0%} of the time: the intervals are wider "
            "than needed and the model is underconfident"
        )
        return "underconfident"

    warnings.append(
        f"a PIT uniformity test rejects calibration (p={pit_p:.4f}) even though "
        "interval coverage looks reasonable; the distribution's shape is wrong"
    )
    return "miscalibrated_shape"


@dataclass
class WalkForwardConfig:
    """Execution assumptions for a walk-forward run.

    ``cost_bps_per_trade`` and ``slippage_bps`` default to zero and must be set
    deliberately. A strategy result reported at zero cost is not a result, so
    ``ExecutionAssumptions.describe`` makes the choice visible in the output.
    """

    min_train_bars: int = 250
    step_bars: int = 5
    horizon_label: str = "5d"
    cost_bps_per_trade: float = 0.0
    slippage_bps: float = 0.0
    fill_rule: str = "next_bar_open"
    max_forecasts: int | None = None

    def describe(self) -> dict:
        return {
            "min_train_bars": self.min_train_bars,
            "step_bars": self.step_bars,
            "horizon_label": self.horizon_label,
            "cost_bps_per_trade": self.cost_bps_per_trade,
            "slippage_bps": self.slippage_bps,
            "fill_rule": self.fill_rule,
            "zero_cost_warning": (
                "costs and slippage are both zero: these results are gross of "
                "execution and are not achievable returns"
                if self.cost_bps_per_trade == 0 and self.slippage_bps == 0
                else None
            ),
        }


@dataclass
class WalkForwardResult:
    model_key: str
    model_version: str
    config: WalkForwardConfig
    scored: list[ScoredForecast]
    metrics: EvaluationMetrics
    skipped: list[str] = field(default_factory=list)
    execution_assumptions: dict = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)


def walk_forward(
    model: Forecaster,
    *,
    timestamps: Sequence[dt.datetime],
    close: Sequence[float],
    config: WalkForwardConfig,
    high: Sequence[float] | None = None,
    low: Sequence[float] | None = None,
    volume: Sequence[float | None] | None = None,
    regime_fn: Callable[[Sequence[float]], str | None] | None = None,
    baseline: Forecaster | None = None,
) -> WalkForwardResult:
    """Evaluate a model chronologically, out of sample.

    At each step the model sees only ``close[:i]`` and is scored against
    ``close[i + horizon]``. The context's ``information_cutoff`` is set to the
    last visible timestamp and the model's own ``check_context`` raises if it
    is ever handed data past it -- so a leak fails the run rather than
    producing a flattering number.
    """
    n = len(close)
    horizon_bars = HORIZONS[config.horizon_label]
    if n < config.min_train_bars + horizon_bars + 1:
        raise InsufficientDataError(
            f"walk-forward over a {config.horizon_label} horizon needs at least "
            f"{config.min_train_bars + horizon_bars + 1} bars but received {n}",
            details={
                "required": config.min_train_bars + horizon_bars + 1,
                "available": n,
            },
        )

    scored: list[ScoredForecast] = []
    baseline_scored: list[ScoredForecast] = []
    skipped: list[str] = []

    i = config.min_train_bars
    while i + horizon_bars < n:
        if config.max_forecasts is not None and len(scored) >= config.max_forecasts:
            break

        train_close = tuple(float(c) for c in close[:i])
        train_ts = tuple(timestamps[:i])
        context = ForecastContext(
            timestamps=train_ts,
            close=train_close,
            high=tuple(float(h) for h in high[:i]) if high else (),
            low=tuple(float(x) for x in low[:i]) if low else (),
            volume=tuple(volume[:i]) if volume else (),
            # The contract that makes this out-of-sample: nothing past here.
            information_cutoff=train_ts[-1],
        )

        realised = math.log(float(close[i + horizon_bars]) / float(close[i - 1]))
        regime = regime_fn(train_close) if regime_fn else None

        try:
            distribution = model.forecast(context, config.horizon_label)
        except InsufficientDataError as exc:
            skipped.append(f"step at index {i}: {exc.message}")
            i += config.step_bars
            continue

        scored.append(
            score_forecast(
                distribution,
                realised,
                information_cutoff=train_ts[-1],
                resolved_at=timestamps[i + horizon_bars],
                regime_label=regime,
            )
        )

        if baseline is not None:
            try:
                base_dist = baseline.forecast(context, config.horizon_label)
            except InsufficientDataError:
                pass
            else:
                baseline_scored.append(
                    score_forecast(
                        base_dist,
                        realised,
                        information_cutoff=train_ts[-1],
                        resolved_at=timestamps[i + horizon_bars],
                        regime_label=regime,
                    )
                )

        i += config.step_bars

    metrics = aggregate_metrics(
        scored,
        model_key=model.spec.key,
        model_version=model.spec.version,
        horizon_label=config.horizon_label,
        baseline_scored=baseline_scored or None,
        baseline_key=baseline.spec.key if baseline else None,
    )

    limitations = [
        "Forecast windows overlap when step_bars is smaller than the horizon, so "
        "the scored forecasts are not independent and the effective sample size "
        "is smaller than the count reported.",
        "The run uses a single instrument's history; results do not generalise "
        "to other instruments without re-evaluation.",
        "The model's hyperparameters were not re-tuned at each step, so this "
        "measures a fixed configuration rather than an adaptive process.",
    ]
    if config.step_bars < horizon_bars:
        effective = len(scored) * config.step_bars / horizon_bars
        limitations.append(
            f"with step {config.step_bars} and horizon {horizon_bars} bars, the "
            f"{len(scored)} forecasts carry roughly {effective:.0f} independent "
            "observations' worth of information"
        )
    if config.cost_bps_per_trade == 0 and config.slippage_bps == 0:
        limitations.append(
            "Execution costs are zero in this configuration. Any implied trading "
            "performance is gross and not achievable."
        )

    return WalkForwardResult(
        model_key=model.spec.key,
        model_version=model.spec.version,
        config=config,
        scored=scored,
        metrics=metrics,
        skipped=skipped,
        execution_assumptions=config.describe(),
        limitations=limitations,
    )


def compare_models(
    models: Sequence[Forecaster],
    *,
    timestamps: Sequence[dt.datetime],
    close: Sequence[float],
    config: WalkForwardConfig,
    baseline_key: str = "random_walk",
) -> dict[str, WalkForwardResult]:
    """Walk-forward every model against the same data and baseline."""
    from app.forecasting.base import get_model

    baseline = get_model(baseline_key)
    out: dict[str, WalkForwardResult] = {}
    for model in models:
        try:
            out[model.spec.key] = walk_forward(
                model,
                timestamps=timestamps,
                close=close,
                config=config,
                baseline=None if model.spec.key == baseline_key else baseline,
            )
        except InsufficientDataError as exc:
            log.info("walk_forward_skipped", model=model.spec.key, reason=exc.message)
    return out
