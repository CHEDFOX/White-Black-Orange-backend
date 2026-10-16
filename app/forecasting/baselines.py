"""Baseline forecasting models.

These exist to be beaten. A complex model is only worth running if it
outperforms the random walk below on out-of-sample, cost-aware evaluation, and
the platform refuses to present any model's numbers without the baseline's
alongside.

Each model is deliberately simple enough to audit by hand. In order of
increasing structure:

``random_walk``
    Zero expected return; uncertainty from historical volatility scaled by the
    square root of the horizon. For most equities over most horizons this is
    genuinely hard to beat, which is the point.
``historical_drift``
    The random walk plus the window's own mean return. Documents the trap of
    extrapolating a sample mean: the standard error of a mean return over 250
    days is roughly the same size as the mean itself.
``ewma_volatility``
    Zero drift, but volatility from an exponentially weighted moving average,
    which responds to the volatility clustering the ARCH test detects.
``student_t``
    Zero drift with a fitted Student-t, so the tails are not Gaussian. This is
    the most defensible of the four for risk purposes.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as scipy_stats

from app.db.models.enums import ForecastTarget, ModelLifecycle
from app.forecasting.base import (
    DEFAULT_QUANTILES,
    ForecastContext,
    ForecastDistribution,
    Forecaster,
    ModelSpec,
    register_model,
)

ALL_HORIZONS = ("1d", "5d", "1mo", "3mo", "6mo", "1y")


def _log_returns(close: tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(close, dtype=float)
    return np.diff(np.log(arr))


def _horizon_warnings(horizon_bars: int, observations: int) -> list[str]:
    """Warn when the horizon is long relative to the evidence behind it."""
    out: list[str] = []
    if horizon_bars > observations / 4:
        out.append(
            f"the {horizon_bars}-bar horizon is long relative to the "
            f"{observations} observations used to estimate it; the resulting "
            "interval is poorly determined"
        )
    if horizon_bars >= 126:
        out.append(
            "square-root-of-time scaling assumes returns are serially "
            "uncorrelated and volatility is constant over the horizon; over "
            "six months or more both assumptions typically fail"
        )
    return out


class RandomWalkForecaster(Forecaster):
    spec = ModelSpec(
        key="random_walk",
        version="1.0.0",
        family="baseline",
        display_name="Random walk (zero drift)",
        description=(
            "Expected log return of zero; uncertainty from the realised "
            "volatility of the lookback window, scaled by the square root of "
            "the horizon. Quantiles are Gaussian."
        ),
        target=ForecastTarget.LOG_RETURN,
        supported_horizons=ALL_HORIZONS,
        min_observations=60,
        is_baseline=True,
        lifecycle=ModelLifecycle.VALIDATED,
        lifecycle_reason=(
            "A reference baseline whose behaviour is fully specified; it is "
            "validated as a benchmark, not as a profitable strategy."
        ),
        assumptions=(
            "Expected excess return over the horizon is zero.",
            "Log returns are independent and identically distributed.",
            "Volatility over the horizon equals the lookback window's volatility.",
            "Returns are Gaussian, so the quantiles have Gaussian tails.",
        ),
        limitations=(
            "By construction this model has no directional information. Its "
            "value is as the benchmark any other model must beat.",
            "Gaussian quantiles understate tail risk for essentially every "
            "equity; use the Student-t baseline for tail estimates.",
            "It cannot anticipate a volatility regime change.",
        ),
        invalidation_conditions=(
            "Realised volatility over the horizon exceeds the lookback "
            "estimate by more than half.",
            "A corporate action, halt or index reconstitution occurs within "
            "the horizon.",
            "A structural break is detected in the lookback window.",
        ),
        hyperparameters={"lookback_bars": 252},
    )

    def forecast(self, context: ForecastContext, horizon: str) -> ForecastDistribution:
        bars = self.check_context(context, horizon)
        lookback = int(self.spec.hyperparameters["lookback_bars"])
        rets = _log_returns(context.close)[-lookback:]

        sigma_per_bar = float(np.std(rets, ddof=1))
        sigma_horizon = sigma_per_bar * math.sqrt(bars)

        quantiles = {
            q: float(scipy_stats.norm.ppf(q, loc=0.0, scale=sigma_horizon))
            for q in DEFAULT_QUANTILES
        }
        dist = ForecastDistribution(
            target=self.spec.target,
            horizon_label=horizon,
            horizon_bars=bars,
            point_estimate=0.0,
            quantiles=quantiles,
            predicted_volatility=sigma_horizon,
            # Exactly 0.5 by construction. Stated so no reader mistakes it for
            # an estimated edge.
            prob_positive=0.5,
            distribution_family="normal",
            distribution_params={"mu": 0.0, "sigma": sigma_horizon},
            contributing_factors=[
                {
                    "factor": "realised volatility",
                    "value": sigma_per_bar,
                    "description": (
                        f"per-bar volatility of {sigma_per_bar:.4f} over the last "
                        f"{len(rets)} bars, scaled by sqrt({bars})"
                    ),
                }
            ],
            observations_used=len(rets),
            warnings=_horizon_warnings(bars, len(rets)),
        )
        dist.validate()
        return dist


class HistoricalDriftForecaster(Forecaster):
    spec = ModelSpec(
        key="historical_drift",
        version="1.0.0",
        family="baseline",
        display_name="Historical drift",
        description=(
            "Extrapolates the lookback window's mean log return over the "
            "horizon, with Gaussian uncertainty. Reports the standard error of "
            "that mean so the reader can see how weakly determined it is."
        ),
        target=ForecastTarget.LOG_RETURN,
        supported_horizons=ALL_HORIZONS,
        min_observations=120,
        is_baseline=True,
        lifecycle=ModelLifecycle.EXPERIMENTAL,
        lifecycle_reason=(
            "Kept as an instructive baseline. Extrapolating a sample mean "
            "return is a documented error, and the model reports the standard "
            "error that demonstrates why."
        ),
        assumptions=(
            "The lookback window's mean return is an unbiased estimate of the "
            "expected return over the horizon.",
            "That expected return is stable across the horizon.",
        ),
        limitations=(
            "The standard error of a mean daily return over one year is "
            "typically as large as the mean itself, so the drift term is "
            "statistically indistinguishable from zero in most windows.",
            "Extrapolating recent drift is momentum by another name, and "
            "performs poorly out of sample at longer horizons.",
            "A window ending in a bubble extrapolates the bubble.",
        ),
        invalidation_conditions=(
            "The drift's 95% confidence interval contains zero, which it "
            "usually does.",
            "A regime change makes the lookback window unrepresentative.",
        ),
        hyperparameters={"lookback_bars": 252},
    )

    def forecast(self, context: ForecastContext, horizon: str) -> ForecastDistribution:
        bars = self.check_context(context, horizon)
        lookback = int(self.spec.hyperparameters["lookback_bars"])
        rets = _log_returns(context.close)[-lookback:]
        n = len(rets)

        mu_per_bar = float(np.mean(rets))
        sigma_per_bar = float(np.std(rets, ddof=1))
        mu_horizon = mu_per_bar * bars
        sigma_horizon = sigma_per_bar * math.sqrt(bars)

        # Standard error of the mean, scaled to the horizon. This is the number
        # that shows the drift is noise.
        se_mu_horizon = (sigma_per_bar / math.sqrt(n)) * bars
        ci_low = mu_horizon - 1.96 * se_mu_horizon
        ci_high = mu_horizon + 1.96 * se_mu_horizon
        drift_indistinguishable = ci_low <= 0.0 <= ci_high

        warnings = _horizon_warnings(bars, n)
        if drift_indistinguishable:
            warnings.append(
                f"the estimated drift of {mu_horizon:+.4f} over the horizon has a "
                f"95% confidence interval of [{ci_low:+.4f}, {ci_high:+.4f}], which "
                "contains zero: this drift is not statistically distinguishable "
                "from no drift at all"
            )

        quantiles = {
            q: float(scipy_stats.norm.ppf(q, loc=mu_horizon, scale=sigma_horizon))
            for q in DEFAULT_QUANTILES
        }
        dist = ForecastDistribution(
            target=self.spec.target,
            horizon_label=horizon,
            horizon_bars=bars,
            point_estimate=mu_horizon,
            quantiles=quantiles,
            predicted_volatility=sigma_horizon,
            prob_positive=float(
                1.0 - scipy_stats.norm.cdf(0.0, loc=mu_horizon, scale=sigma_horizon)
            ),
            distribution_family="normal",
            distribution_params={"mu": mu_horizon, "sigma": sigma_horizon},
            contributing_factors=[
                {
                    "factor": "historical drift",
                    "value": mu_per_bar,
                    "description": (
                        f"mean log return of {mu_per_bar:+.6f} per bar over {n} bars"
                    ),
                },
                {
                    "factor": "drift standard error",
                    "value": se_mu_horizon,
                    "description": (
                        f"95% CI for the horizon drift: [{ci_low:+.4f}, {ci_high:+.4f}]"
                        + (" — contains zero" if drift_indistinguishable else "")
                    ),
                },
            ],
            observations_used=n,
            warnings=warnings,
        )
        dist.validate()
        return dist


class EwmaVolatilityForecaster(Forecaster):
    spec = ModelSpec(
        key="ewma_volatility",
        version="1.0.0",
        family="baseline",
        display_name="EWMA volatility (zero drift)",
        description=(
            "Zero expected return with volatility from an exponentially "
            "weighted moving average of squared returns (RiskMetrics-style, "
            "lambda = 0.94)."
        ),
        target=ForecastTarget.LOG_RETURN,
        supported_horizons=ALL_HORIZONS,
        min_observations=100,
        is_baseline=True,
        lifecycle=ModelLifecycle.VALIDATED,
        lifecycle_reason=(
            "Volatility clustering is a robust empirical regularity, so an "
            "EWMA estimate is a defensible improvement over a flat window "
            "standard deviation for short horizons."
        ),
        assumptions=(
            "Expected return over the horizon is zero.",
            "Recent volatility predicts near-term volatility (decay 0.94).",
            "Volatility is constant across the horizon once estimated.",
        ),
        limitations=(
            "The decay factor is a convention, not fitted to this instrument.",
            "EWMA has no mean reversion in volatility, so it over-extrapolates "
            "a spike at longer horizons. Prefer it for horizons under a month.",
            "Quantiles remain Gaussian, so tails are still understated.",
        ),
        invalidation_conditions=(
            "A volatility spike within the horizon exceeding the estimate by "
            "more than half.",
            "Use at a horizon beyond one month, where the no-mean-reversion "
            "assumption is clearly violated.",
        ),
        hyperparameters={"lambda": 0.94, "lookback_bars": 252},
    )

    def forecast(self, context: ForecastContext, horizon: str) -> ForecastDistribution:
        bars = self.check_context(context, horizon)
        lam = float(self.spec.hyperparameters["lambda"])
        lookback = int(self.spec.hyperparameters["lookback_bars"])
        rets = _log_returns(context.close)[-lookback:]

        # Seed with the sample variance, then apply the EWMA recursion.
        variance = float(np.var(rets, ddof=1))
        for r in rets:
            variance = lam * variance + (1 - lam) * (float(r) ** 2)
        sigma_per_bar = math.sqrt(variance)
        sigma_horizon = sigma_per_bar * math.sqrt(bars)

        simple_sigma = float(np.std(rets, ddof=1))
        warnings = _horizon_warnings(bars, len(rets))
        if bars > 21:
            warnings.append(
                "EWMA volatility has no mean reversion, so at horizons beyond "
                "about a month it over-extrapolates the current volatility level"
            )

        quantiles = {
            q: float(scipy_stats.norm.ppf(q, loc=0.0, scale=sigma_horizon))
            for q in DEFAULT_QUANTILES
        }
        dist = ForecastDistribution(
            target=self.spec.target,
            horizon_label=horizon,
            horizon_bars=bars,
            point_estimate=0.0,
            quantiles=quantiles,
            predicted_volatility=sigma_horizon,
            prob_positive=0.5,
            distribution_family="normal",
            distribution_params={"mu": 0.0, "sigma": sigma_horizon, "lambda": lam},
            contributing_factors=[
                {
                    "factor": "EWMA volatility",
                    "value": sigma_per_bar,
                    "description": (
                        f"EWMA per-bar volatility {sigma_per_bar:.4f} versus a flat "
                        f"window estimate of {simple_sigma:.4f}"
                    ),
                }
            ],
            observations_used=len(rets),
            warnings=warnings,
        )
        dist.validate()
        return dist


class StudentTForecaster(Forecaster):
    spec = ModelSpec(
        key="student_t",
        version="1.0.0",
        family="baseline",
        display_name="Student-t (zero drift, fat tails)",
        description=(
            "Zero expected return with a Student-t fitted to the lookback "
            "window's standardised returns, so the quantiles have fat tails."
        ),
        target=ForecastTarget.LOG_RETURN,
        supported_horizons=ALL_HORIZONS,
        min_observations=250,
        is_baseline=True,
        lifecycle=ModelLifecycle.VALIDATED,
        lifecycle_reason=(
            "Equity returns are reliably fat-tailed, so a Student-t gives "
            "better-calibrated tail quantiles than a Gaussian. Validated for "
            "risk framing, not for direction."
        ),
        assumptions=(
            "Expected return over the horizon is zero.",
            "Returns follow a Student-t with the fitted degrees of freedom.",
            "The horizon distribution is the per-bar distribution scaled by "
            "the square root of the horizon.",
        ),
        limitations=(
            "Square-root scaling of a Student-t is an approximation: a sum of "
            "t-distributed variables is not itself t-distributed.",
            "The degrees-of-freedom estimate is unstable on fewer than a few "
            "hundred observations.",
            "Fat tails describe the distribution's shape, not when a tail "
            "event will occur.",
        ),
        invalidation_conditions=(
            "A realised move beyond the 1st or 99th percentile, which should "
            "occur about twice a year by construction.",
            "A detected structural break in the lookback window.",
        ),
        hyperparameters={"lookback_bars": 504},
    )

    def forecast(self, context: ForecastContext, horizon: str) -> ForecastDistribution:
        bars = self.check_context(context, horizon)
        lookback = int(self.spec.hyperparameters["lookback_bars"])
        rets = _log_returns(context.close)[-lookback:]

        # floc=0 pins the location at zero, so the returned loc is discarded.
        df, _loc, scale = scipy_stats.t.fit(rets, floc=0.0)
        sigma_per_bar = float(np.std(rets, ddof=1))
        sigma_horizon = sigma_per_bar * math.sqrt(bars)
        scale_horizon = float(scale) * math.sqrt(bars)

        warnings = _horizon_warnings(bars, len(rets))
        if df > 30:
            warnings.append(
                f"the fitted degrees of freedom ({df:.1f}) are high, so the "
                "distribution is close to Gaussian and this model adds little "
                "over the random-walk baseline for this window"
            )
        if df < 2.5:
            warnings.append(
                f"the fitted degrees of freedom ({df:.1f}) are very low; the "
                "variance is barely defined and the quantiles are unstable"
            )
        warnings.append(
            "square-root-of-time scaling of a Student-t is an approximation: a "
            "sum of t variables is not t-distributed"
        )

        quantiles = {
            q: float(scipy_stats.t.ppf(q, df, loc=0.0, scale=scale_horizon))
            for q in DEFAULT_QUANTILES
        }
        dist = ForecastDistribution(
            target=self.spec.target,
            horizon_label=horizon,
            horizon_bars=bars,
            point_estimate=0.0,
            quantiles=quantiles,
            predicted_volatility=sigma_horizon,
            prob_positive=0.5,
            distribution_family="student_t",
            distribution_params={
                "df": float(df),
                "loc": 0.0,
                "scale": scale_horizon,
            },
            contributing_factors=[
                {
                    "factor": "fitted tail thickness",
                    "value": float(df),
                    "description": (
                        f"Student-t degrees of freedom {df:.2f} "
                        f"(lower means fatter tails; 30+ is effectively Gaussian)"
                    ),
                }
            ],
            observations_used=len(rets),
            warnings=warnings,
        )
        dist.validate()
        return dist


class SeasonalNaiveForecaster(Forecaster):
    spec = ModelSpec(
        key="seasonal_naive",
        version="1.0.0",
        family="baseline",
        display_name="Seasonal naive (same period last year)",
        description=(
            "Predicts the return observed over the same calendar span one year "
            "earlier. A deliberately weak baseline for detecting spurious "
            "seasonality claims."
        ),
        target=ForecastTarget.LOG_RETURN,
        supported_horizons=("1mo", "3mo", "6mo"),
        min_observations=400,
        is_baseline=True,
        lifecycle=ModelLifecycle.EXPERIMENTAL,
        lifecycle_reason=(
            "Included as a control. If a seasonality-based model cannot beat "
            "this, its seasonal signal is not real."
        ),
        assumptions=(
            "The same calendar span repeats its return year over year.",
        ),
        limitations=(
            "A single prior year is one observation; this is the weakest "
            "baseline in the set and is expected to perform badly.",
            "It carries no uncertainty information of its own; the interval "
            "comes from historical volatility.",
        ),
        invalidation_conditions=(
            "Any structural change in the business or the market since the "
            "prior year.",
        ),
        hyperparameters={"season_bars": 252},
    )

    def forecast(self, context: ForecastContext, horizon: str) -> ForecastDistribution:
        bars = self.check_context(context, horizon)
        season = int(self.spec.hyperparameters["season_bars"])
        close = np.asarray(context.close, dtype=float)

        start_idx = len(close) - 1 - season
        end_idx = start_idx + bars
        if start_idx < 0 or end_idx >= len(close):
            from app.core.errors import InsufficientDataError

            raise InsufficientDataError(
                f"a {horizon} seasonal-naive forecast needs {season + bars + 1} bars "
                f"but only {len(close)} are available",
                details={"model": self.spec.key},
            )

        analogue_return = float(math.log(close[end_idx] / close[start_idx]))
        rets = np.diff(np.log(close))[-season:]
        sigma_horizon = float(np.std(rets, ddof=1)) * math.sqrt(bars)

        quantiles = {
            q: float(scipy_stats.norm.ppf(q, loc=analogue_return, scale=sigma_horizon))
            for q in DEFAULT_QUANTILES
        }
        dist = ForecastDistribution(
            target=self.spec.target,
            horizon_label=horizon,
            horizon_bars=bars,
            point_estimate=analogue_return,
            quantiles=quantiles,
            predicted_volatility=sigma_horizon,
            prob_positive=float(
                1.0 - scipy_stats.norm.cdf(0.0, loc=analogue_return, scale=sigma_horizon)
            ),
            distribution_family="normal",
            distribution_params={"mu": analogue_return, "sigma": sigma_horizon},
            contributing_factors=[
                {
                    "factor": "prior-year analogue",
                    "value": analogue_return,
                    "description": (
                        f"the same {bars}-bar span one year earlier returned "
                        f"{analogue_return:+.4f} in log terms — a single observation"
                    ),
                }
            ],
            observations_used=len(close),
            warnings=[
                "this point estimate rests on one historical observation and is "
                "included as a control, not as a recommendation"
            ],
        )
        dist.validate()
        return dist


register_model(RandomWalkForecaster())
register_model(HistoricalDriftForecaster())
register_model(EwmaVolatilityForecaster())
register_model(StudentTForecaster())
register_model(SeasonalNaiveForecaster())
