"""Forecasting model and evaluation tests.

The scoring tests use closed-form or hand-computable cases so a wrong CRPS or
PIT cannot hide behind plausible-looking numbers. The calibration tests feed
deliberately overconfident and underconfident models through the evaluator and
assert it names the specific failure.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pytest
from scipy import stats as scipy_stats

from app.core.errors import InsufficientDataError
from app.core.timeutil import UTC
from app.db.models.enums import ForecastTarget
from app.forecasting.base import (
    DEFAULT_QUANTILES,
    HORIZONS,
    ForecastContext,
    ForecastDistribution,
    all_models,
    get_model,
)
from app.forecasting.evaluation import (
    WalkForwardConfig,
    aggregate_metrics,
    crps_from_quantiles,
    interval_hits,
    pit_value,
    score_forecast,
    walk_forward,
)


def walk_series(n, *, seed=1, sigma=0.012, mu=0.0, start=100.0):
    rng = random.Random(seed)
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * math.exp(rng.gauss(mu, sigma)))
    return out


def ctx(closes, *, cutoff_offset=0):
    n = len(closes)
    ts = tuple(dt.datetime(2020, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(n))
    return ForecastContext(
        timestamps=ts,
        close=tuple(float(c) for c in closes),
        information_cutoff=ts[-1] + dt.timedelta(days=cutoff_offset),
    )


def normal_distribution(mu, sigma, horizon="5d"):
    return ForecastDistribution(
        target=ForecastTarget.LOG_RETURN,
        horizon_label=horizon,
        horizon_bars=HORIZONS[horizon],
        point_estimate=mu,
        quantiles={q: float(scipy_stats.norm.ppf(q, mu, sigma)) for q in DEFAULT_QUANTILES},
        predicted_volatility=sigma,
        distribution_family="normal",
        distribution_params={"mu": mu, "sigma": sigma},
    )


class TestDistributionValidation:
    def test_a_non_monotonic_quantile_set_is_rejected(self):
        """A non-monotonic quantile function is not a distribution.

        Storing one would make every interval and probability downstream
        meaningless, and the defect would be invisible in a chart.
        """
        bad = ForecastDistribution(
            target=ForecastTarget.LOG_RETURN,
            horizon_label="5d",
            horizon_bars=5,
            point_estimate=0.0,
            quantiles={0.25: 0.05, 0.50: 0.01, 0.75: 0.09},
        )
        with pytest.raises(ValueError, match="not monotonic"):
            bad.validate()

    def test_a_distribution_without_quantiles_is_rejected(self):
        bare = ForecastDistribution(
            target=ForecastTarget.LOG_RETURN, horizon_label="5d",
            horizon_bars=5, point_estimate=0.01, quantiles={},
        )
        with pytest.raises(ValueError, match="must provide quantiles"):
            bare.validate()

    def test_an_out_of_range_probability_is_rejected(self):
        bad = ForecastDistribution(
            target=ForecastTarget.LOG_RETURN, horizon_label="5d", horizon_bars=5,
            point_estimate=0.0, quantiles={0.5: 0.0}, prob_positive=1.4,
        )
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            bad.validate()

    def test_price_levels_are_derived_from_log_return_quantiles(self):
        dist = normal_distribution(0.0, 0.05)
        levels = dist.as_price_levels(100.0)
        assert levels["0.5"] == pytest.approx(100.0)
        assert levels["0.95"] > levels["0.05"]


class TestBaselineModels:
    def test_every_registered_model_is_a_baseline_and_declares_its_terms(self):
        for model in all_models():
            assert model.spec.limitations, f"{model.spec.key} must declare limitations"
            assert model.spec.assumptions, f"{model.spec.key} must declare assumptions"
            assert model.spec.invalidation_conditions, (
                f"{model.spec.key} must declare what would invalidate it"
            )

    def test_random_walk_has_zero_drift_and_half_probability(self):
        dist = get_model("random_walk").forecast(ctx(walk_series(300, seed=2)), "5d")
        assert dist.point_estimate == 0.0
        assert dist.prob_positive == 0.5
        assert dist.quantiles[0.5] == pytest.approx(0.0, abs=1e-9)

    def test_random_walk_uncertainty_scales_with_the_square_root_of_horizon(self):
        context = ctx(walk_series(400, seed=3))
        one_day = get_model("random_walk").forecast(context, "1d")
        twenty_one = get_model("random_walk").forecast(context, "1mo")
        ratio = twenty_one.predicted_volatility / one_day.predicted_volatility
        assert ratio == pytest.approx(math.sqrt(21), rel=1e-6)

    def test_random_walk_quantiles_are_symmetric(self):
        dist = get_model("random_walk").forecast(ctx(walk_series(300, seed=4)), "5d")
        assert dist.quantiles[0.05] == pytest.approx(-dist.quantiles[0.95])
        assert dist.quantiles[0.25] == pytest.approx(-dist.quantiles[0.75])

    def test_historical_drift_reports_that_its_drift_is_indistinguishable_from_zero(self):
        """The model exists to demonstrate this, so it must say it."""
        dist = get_model("historical_drift").forecast(ctx(walk_series(400, seed=5)), "1mo")
        assert any("not statistically distinguishable" in w for w in dist.warnings)
        factors = {f["factor"] for f in dist.contributing_factors}
        assert "drift standard error" in factors

    def test_historical_drift_extrapolates_a_strong_trend(self):
        closes = [100.0 * (1.002**i) for i in range(400)]
        dist = get_model("historical_drift").forecast(ctx(closes), "1mo")
        assert dist.point_estimate > 0
        assert dist.prob_positive > 0.5

    def test_ewma_reacts_to_a_volatility_spike_more_than_a_flat_window(self):
        calm = walk_series(300, seed=6, sigma=0.004)
        rng = random.Random(7)
        loud = [calm[-1]]
        for _ in range(20):
            loud.append(loud[-1] * math.exp(rng.gauss(0, 0.05)))
        context = ctx(calm + loud[1:])
        ewma = get_model("ewma_volatility").forecast(context, "5d")
        flat = get_model("random_walk").forecast(context, "5d")
        assert ewma.predicted_volatility > flat.predicted_volatility

    def test_ewma_warns_when_used_beyond_a_month(self):
        dist = get_model("ewma_volatility").forecast(ctx(walk_series(400, seed=8)), "3mo")
        assert any("no mean reversion" in w for w in dist.warnings)

    def test_student_t_fits_fat_tails_and_dominates_the_gaussian_deep_in_the_tail(self):
        """Where a fitted Student-t actually beats a Gaussian.

        Not at the 5% quantile. A fitted t has a smaller scale parameter than
        the sample standard deviation (its variance is scale^2 * df/(df-2)), so
        in the near tail it is *less* extreme than a Gaussian matched on
        standard deviation. The crossover for a typical equity fit sits around
        the 1% quantile, and the fat-tail advantage shows up beyond it -- which
        is exactly the region a risk estimate cares about.
        """
        rng = random.Random(9)
        closes = [100.0]
        for i in range(700):
            shock = rng.gauss(0, 0.008) if i % 30 else rng.gauss(0, 0.06)
            closes.append(closes[-1] * math.exp(shock))
        context = ctx(closes)
        t_dist = get_model("student_t").forecast(context, "5d")

        df = t_dist.distribution_params["df"]
        assert df < 10, "the generator is strongly fat-tailed, so the fit must be too"

        # Compare the 0.1% quantile from each model's own stored parameters, in
        # units of its own predicted volatility.
        t_deep = (
            scipy_stats.t.ppf(0.001, df, loc=0.0, scale=t_dist.distribution_params["scale"])
            / t_dist.predicted_volatility
        )
        normal_deep = scipy_stats.norm.ppf(0.001)
        assert t_deep < normal_deep * 1.5, (
            "deep in the tail the fitted t must be substantially more extreme "
            "than a Gaussian"
        )

    def test_the_near_tail_crossover_is_a_known_property_not_a_defect(self):
        """Documents the behaviour the previous test explains, so it is not
        mistaken for a bug later."""
        rng = random.Random(9)
        closes = [100.0]
        for i in range(700):
            shock = rng.gauss(0, 0.008) if i % 30 else rng.gauss(0, 0.06)
            closes.append(closes[-1] * math.exp(shock))
        context = ctx(closes)
        t_dist = get_model("student_t").forecast(context, "5d")
        normal = get_model("random_walk").forecast(context, "5d")
        t_near = t_dist.quantiles[0.05] / t_dist.predicted_volatility
        normal_near = normal.quantiles[0.05] / normal.predicted_volatility
        assert t_near > normal_near, (
            "in the near tail the fitted t is less extreme than a volatility-"
            "matched Gaussian; this is expected, not a miscalibration"
        )

    def test_student_t_discloses_the_scaling_approximation(self):
        dist = get_model("student_t").forecast(ctx(walk_series(600, seed=10)), "1mo")
        assert any("sum of t variables is not t-distributed" in w for w in dist.warnings)

    def test_an_unsupported_horizon_is_refused(self):
        with pytest.raises(InsufficientDataError, match="does not support"):
            get_model("seasonal_naive").forecast(ctx(walk_series(600, seed=11)), "1d")

    def test_too_little_history_is_refused(self):
        with pytest.raises(InsufficientDataError, match="needs 60 observations"):
            get_model("random_walk").forecast(ctx(walk_series(30, seed=12)), "5d")

    def test_a_context_past_its_cutoff_is_refused(self):
        """Belt and braces against a leak reaching a model."""
        context = ctx(walk_series(300, seed=13), cutoff_offset=-10)
        with pytest.raises(InsufficientDataError, match="after its information cutoff"):
            get_model("random_walk").forecast(context, "5d")


class TestCRPS:
    def test_crps_is_zero_for_a_perfect_point_forecast(self):
        quantiles = dict.fromkeys(DEFAULT_QUANTILES, 0.05)
        assert crps_from_quantiles(quantiles, 0.05) == pytest.approx(0.0)

    def test_crps_grows_with_the_error_of_a_point_forecast(self):
        quantiles = dict.fromkeys(DEFAULT_QUANTILES, 0.0)
        near = crps_from_quantiles(quantiles, 0.01)
        far = crps_from_quantiles(quantiles, 0.10)
        assert far > near

    def test_crps_reduces_to_absolute_error_for_a_point_mass(self):
        """A degenerate forecast must score as mean absolute error."""
        quantiles = dict.fromkeys(DEFAULT_QUANTILES, 0.0)
        for outcome in (0.01, 0.05, -0.03):
            assert crps_from_quantiles(quantiles, outcome) == pytest.approx(abs(outcome))

    def test_a_sharper_correct_forecast_beats_a_vague_one(self):
        sharp = normal_distribution(0.0, 0.01)
        vague = normal_distribution(0.0, 0.10)
        outcome = 0.002
        assert crps_from_quantiles(sharp.quantiles, outcome) < crps_from_quantiles(
            vague.quantiles, outcome
        )

    def test_a_vague_forecast_beats_a_confidently_wrong_one(self):
        """The property that makes CRPS a proper score.

        Over-narrow intervals must be punished when the outcome lands outside
        them, otherwise a model is rewarded for false confidence.
        """
        confident_wrong = normal_distribution(0.0, 0.005)
        vague = normal_distribution(0.0, 0.10)
        outcome = 0.15
        assert crps_from_quantiles(vague.quantiles, outcome) < crps_from_quantiles(
            confident_wrong.quantiles, outcome
        )

    def test_crps_requires_quantiles(self):
        with pytest.raises(ValueError, match="at least one quantile"):
            crps_from_quantiles({}, 0.0)


class TestPIT:
    def test_pit_of_the_median_is_one_half(self):
        dist = normal_distribution(0.01, 0.05)
        assert pit_value(dist, 0.01) == pytest.approx(0.5)

    def test_pit_uses_the_parametric_cdf_where_available(self):
        dist = normal_distribution(0.0, 0.05)
        expected = float(scipy_stats.norm.cdf(0.03, 0.0, 0.05))
        assert pit_value(dist, 0.03) == pytest.approx(expected)

    def test_pit_is_bounded(self):
        dist = normal_distribution(0.0, 0.05)
        assert 0.0 <= pit_value(dist, -10.0) <= 1.0
        assert 0.0 <= pit_value(dist, 10.0) <= 1.0

    def test_an_empirical_distribution_still_yields_a_pit(self):
        dist = ForecastDistribution(
            target=ForecastTarget.LOG_RETURN, horizon_label="5d", horizon_bars=5,
            point_estimate=0.0,
            quantiles={0.05: -0.08, 0.5: 0.0, 0.95: 0.08},
            distribution_family="empirical",
        )
        assert pit_value(dist, 0.0) == pytest.approx(0.5, abs=0.05)


class TestIntervalHits:
    def test_an_outcome_inside_the_interval_is_a_hit(self):
        hits = interval_hits(normal_distribution(0.0, 0.05), 0.01)
        assert hits["50%"] is True
        assert hits["90%"] is True

    def test_an_outcome_in_the_tail_misses_the_narrow_interval(self):
        dist = normal_distribution(0.0, 0.05)
        hits = interval_hits(dist, dist.quantiles[0.95] + 0.01)
        assert hits["50%"] is False
        assert hits["90%"] is False


class TestDirectionalScoring:
    def test_a_zero_drift_model_gets_no_directional_score(self):
        """Scoring a non-directional model's sign measures nothing."""
        scored = score_forecast(
            normal_distribution(0.0, 0.05), 0.02,
            information_cutoff=dt.datetime(2024, 1, 1, tzinfo=UTC),
            resolved_at=dt.datetime(2024, 1, 8, tzinfo=UTC),
        )
        assert scored.direction_correct is None

    def test_a_directional_model_is_scored_on_direction(self):
        scored = score_forecast(
            normal_distribution(0.02, 0.05), 0.03,
            information_cutoff=dt.datetime(2024, 1, 1, tzinfo=UTC),
            resolved_at=dt.datetime(2024, 1, 8, tzinfo=UTC),
        )
        assert scored.direction_correct is True

    def test_a_wrong_directional_call_is_recorded(self):
        scored = score_forecast(
            normal_distribution(0.02, 0.05), -0.03,
            information_cutoff=dt.datetime(2024, 1, 1, tzinfo=UTC),
            resolved_at=dt.datetime(2024, 1, 8, tzinfo=UTC),
        )
        assert scored.direction_correct is False


class TestCalibrationDetection:
    @staticmethod
    def scored_set(model_sigma, true_sigma, n=200, seed=1):
        rng = random.Random(seed)
        out = []
        base = dt.datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(n):
            outcome = rng.gauss(0.0, true_sigma)
            out.append(
                score_forecast(
                    normal_distribution(0.0, model_sigma),
                    outcome,
                    information_cutoff=base + dt.timedelta(days=i),
                    resolved_at=base + dt.timedelta(days=i + 5),
                )
            )
        return out

    def test_a_well_calibrated_model_is_recognised(self):
        metrics = aggregate_metrics(
            self.scored_set(0.05, 0.05, seed=2),
            model_key="test", model_version="1.0.0", horizon_label="5d",
        )
        assert metrics.calibration_verdict == "well_calibrated"
        assert metrics.interval_coverage["90%"] == pytest.approx(0.9, abs=0.08)

    def test_an_overconfident_model_is_named_as_such(self):
        """The dangerous failure: narrow intervals look like a sharp model."""
        metrics = aggregate_metrics(
            self.scored_set(0.01, 0.05, seed=3),
            model_key="test", model_version="1.0.0", horizon_label="5d",
        )
        assert metrics.calibration_verdict == "overconfident"
        assert metrics.interval_coverage["90%"] < 0.6
        assert any("too narrow" in w for w in metrics.warnings)

    def test_an_underconfident_model_is_named_as_such(self):
        metrics = aggregate_metrics(
            self.scored_set(0.20, 0.05, seed=4),
            model_key="test", model_version="1.0.0", horizon_label="5d",
        )
        assert metrics.calibration_verdict == "underconfident"
        assert metrics.interval_coverage["50%"] > 0.8

    def test_calibration_is_not_assessed_on_too_few_forecasts(self):
        metrics = aggregate_metrics(
            self.scored_set(0.05, 0.05, n=10, seed=5),
            model_key="test", model_version="1.0.0", horizon_label="5d",
        )
        assert metrics.calibration_verdict == "not_assessed"
        assert any("at least 20" in w for w in metrics.warnings)

    def test_a_small_sample_is_flagged_as_too_noisy_to_rank(self):
        metrics = aggregate_metrics(
            self.scored_set(0.05, 0.05, n=25, seed=6),
            model_key="test", model_version="1.0.0", horizon_label="5d",
        )
        assert any("too noisy to rank" in w for w in metrics.warnings)

    def test_no_forecasts_yields_no_metrics_rather_than_zeros(self):
        metrics = aggregate_metrics(
            [], model_key="test", model_version="1.0.0", horizon_label="5d"
        )
        assert metrics.mean_crps is None
        assert any("no forecasts resolved" in w for w in metrics.warnings)


class TestSkillScore:
    def test_a_better_model_has_positive_skill(self):
        good = TestCalibrationDetection.scored_set(0.05, 0.05, seed=7)
        bad = TestCalibrationDetection.scored_set(0.30, 0.05, seed=7)
        metrics = aggregate_metrics(
            good, model_key="good", model_version="1.0.0", horizon_label="5d",
            baseline_scored=bad, baseline_key="worse_baseline",
        )
        assert metrics.skill_vs_baseline > 0

    def test_failing_to_beat_the_baseline_is_stated_plainly(self):
        bad = TestCalibrationDetection.scored_set(0.30, 0.05, seed=8)
        good = TestCalibrationDetection.scored_set(0.05, 0.05, seed=8)
        metrics = aggregate_metrics(
            bad, model_key="bad", model_version="1.0.0", horizon_label="5d",
            baseline_scored=good, baseline_key="random_walk",
        )
        assert metrics.skill_vs_baseline < 0
        assert any("does not beat the random_walk baseline" in w for w in metrics.warnings)


class TestWalkForward:
    def test_walk_forward_produces_out_of_sample_forecasts(self):
        closes = walk_series(800, seed=21)
        ts = [dt.datetime(2020, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(800)]
        result = walk_forward(
            get_model("random_walk"),
            timestamps=ts, close=closes,
            config=WalkForwardConfig(min_train_bars=250, step_bars=10, horizon_label="5d"),
        )
        assert result.metrics.n_forecasts > 30
        assert result.metrics.mean_crps is not None

    def test_every_forecast_cutoff_precedes_its_resolution(self):
        """The invariant that makes the evaluation out-of-sample."""
        closes = walk_series(600, seed=22)
        ts = [dt.datetime(2020, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(600)]
        result = walk_forward(
            get_model("random_walk"),
            timestamps=ts, close=closes,
            config=WalkForwardConfig(min_train_bars=250, step_bars=20, horizon_label="5d"),
        )
        assert result.scored
        for s in result.scored:
            assert s.information_cutoff < s.resolved_at

    def test_a_random_walk_is_well_calibrated_on_gaussian_data(self):
        """The sanity check on the whole pipeline.

        Gaussian data fed to a Gaussian model must come back calibrated. If it
        does not, the scoring or the walk-forward loop is wrong.
        """
        closes = walk_series(1500, seed=23, sigma=0.01)
        ts = [dt.datetime(2015, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(1500)]
        result = walk_forward(
            get_model("random_walk"),
            timestamps=ts, close=closes,
            config=WalkForwardConfig(min_train_bars=250, step_bars=5, horizon_label="5d"),
        )
        coverage = result.metrics.interval_coverage["90%"]
        assert 0.80 <= coverage <= 0.98, f"90% interval covered {coverage:.1%}"

    def test_zero_cost_configuration_is_flagged_as_not_achievable(self):
        closes = walk_series(600, seed=24)
        ts = [dt.datetime(2020, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(600)]
        result = walk_forward(
            get_model("random_walk"),
            timestamps=ts, close=closes,
            config=WalkForwardConfig(min_train_bars=250, step_bars=20),
        )
        assert result.execution_assumptions["zero_cost_warning"] is not None
        assert any("gross and not achievable" in lim for lim in result.limitations)

    def test_overlapping_windows_are_disclosed_with_an_effective_sample_size(self):
        closes = walk_series(900, seed=25)
        ts = [dt.datetime(2020, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(900)]
        result = walk_forward(
            get_model("random_walk"),
            timestamps=ts, close=closes,
            config=WalkForwardConfig(min_train_bars=250, step_bars=1, horizon_label="1mo"),
        )
        assert any("independent observations" in lim for lim in result.limitations)
        assert any("overlap" in lim for lim in result.limitations)

    def test_a_baseline_comparison_produces_a_skill_score(self):
        closes = walk_series(900, seed=26)
        ts = [dt.datetime(2020, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(900)]
        result = walk_forward(
            get_model("historical_drift"),
            timestamps=ts, close=closes,
            config=WalkForwardConfig(min_train_bars=300, step_bars=10, horizon_label="5d"),
            baseline=get_model("random_walk"),
        )
        assert result.metrics.skill_vs_baseline is not None
        assert result.metrics.baseline_key == "random_walk"

    def test_drift_extrapolation_does_not_beat_the_random_walk_on_a_random_walk(self):
        """The documented lesson, asserted rather than asserted-at.

        On genuinely driftless data, extrapolating the sample mean must not
        win. If it did, the evaluation would be leaking.
        """
        closes = walk_series(1500, seed=27, mu=0.0, sigma=0.012)
        ts = [dt.datetime(2015, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(1500)]
        result = walk_forward(
            get_model("historical_drift"),
            timestamps=ts, close=closes,
            config=WalkForwardConfig(min_train_bars=300, step_bars=5, horizon_label="1mo"),
            baseline=get_model("random_walk"),
        )
        assert result.metrics.skill_vs_baseline is not None
        assert result.metrics.skill_vs_baseline < 0.05, (
            "extrapolated drift should give no meaningful edge on driftless data"
        )

    def test_too_little_data_is_refused(self):
        closes = walk_series(100, seed=28)
        ts = [dt.datetime(2020, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(100)]
        with pytest.raises(InsufficientDataError, match="needs at least"):
            walk_forward(
                get_model("random_walk"),
                timestamps=ts, close=closes,
                config=WalkForwardConfig(min_train_bars=250, horizon_label="5d"),
            )

    def test_per_regime_metrics_are_broken_out(self):
        closes = walk_series(900, seed=29)
        ts = [dt.datetime(2020, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(900)]
        result = walk_forward(
            get_model("random_walk"),
            timestamps=ts, close=closes,
            config=WalkForwardConfig(min_train_bars=300, step_bars=10),
            regime_fn=lambda c: "rising" if c[-1] > c[-50] else "falling",
        )
        assert set(result.metrics.per_regime) <= {"rising", "falling"}
        assert result.metrics.per_regime
        for stats in result.metrics.per_regime.values():
            assert stats["n"] > 0
