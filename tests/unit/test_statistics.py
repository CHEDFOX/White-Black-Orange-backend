"""Statistical method tests.

Several of these assert that a method *declines* to find an effect in random
data. That matters more than the positive cases: a statistical engine that
finds structure in noise is worse than no engine.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pytest

from app.analysis.registry import MethodInput, run_method
from app.core.errors import InsufficientDataError
from app.core.timeutil import UTC


def series_input(closes, *, series=None, benchmark=None, scalars=None, resolution="1d"):
    n = len(closes)
    ts = tuple(
        dt.datetime(2020, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(n)
    )
    return MethodInput(
        timestamps=ts,
        open=tuple(float(c) for c in closes),
        high=tuple(float(c) * 1.01 for c in closes),
        low=tuple(float(c) * 0.99 for c in closes),
        close=tuple(float(c) for c in closes),
        volume=tuple(1_000_000.0 for _ in closes),
        benchmark_close=tuple(float(b) for b in benchmark) if benchmark else (),
        series={k: tuple(float(x) for x in v) for k, v in (series or {}).items()},
        scalars=dict(scalars or {}),
        resolution=resolution,
    )


def random_walk(n: int, *, seed: int = 1, sigma: float = 0.015, mu: float = 0.0):
    rng = random.Random(seed)
    out = [100.0]
    for _ in range(n - 1):
        out.append(out[-1] * math.exp(rng.gauss(mu, sigma)))
    return out


class TestReturnDistribution:
    def test_moments_of_a_normal_series_are_near_zero(self):
        closes = random_walk(2000, seed=11)
        v = run_method("return_distribution", series_input(closes)).values
        assert abs(v["skewness"]) < 0.3
        assert abs(v["excess_kurtosis"]) < 0.5

    def test_normality_is_rejected_for_a_fat_tailed_series(self):
        rng = random.Random(5)
        closes = [100.0]
        for i in range(600):
            # Student-t-like: occasional large shocks.
            shock = rng.gauss(0, 0.01) if i % 50 else rng.gauss(0, 0.12)
            closes.append(closes[-1] * math.exp(shock))
        v = run_method("return_distribution", series_input(closes)).values
        assert v["excess_kurtosis"] > 1.0
        assert v["normality_rejected_at_5pct"] is True

    def test_quantiles_are_monotonic(self):
        v = run_method("return_distribution", series_input(random_walk(500, seed=3))).values
        qs = v["quantiles"]
        values = [qs[k] for k in sorted(qs, key=float)]
        assert values == sorted(values)

    def test_small_sample_moment_warning_is_emitted(self):
        result = run_method("return_distribution", series_input(random_walk(40, seed=2)))
        assert any("several hundred" in w for w in result.warnings)


class TestAutocorrelation:
    def test_a_random_walk_shows_no_significant_autocorrelation(self):
        """The negative result that matters.

        If this finds structure, every momentum claim built on the engine is
        suspect.
        """
        v = run_method("autocorrelation", series_input(random_walk(1500, seed=21))).values
        assert v["joint_independence_rejected_at_5pct"] is False
        assert abs(v["coefficients"]["1"]) < v["confidence_band"]

    def test_strong_mean_reversion_is_detected(self):
        # Alternating returns give lag-1 autocorrelation near -1.
        closes = [100.0]
        for i in range(400):
            closes.append(closes[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
        v = run_method("autocorrelation", series_input(closes)).values
        assert v["coefficients"]["1"] < -0.5
        assert v["joint_independence_rejected_at_5pct"] is True

    def test_confidence_band_shrinks_with_sample_size(self):
        small = run_method("autocorrelation", series_input(random_walk(100, seed=4))).values
        large = run_method("autocorrelation", series_input(random_walk(1000, seed=4))).values
        assert large["confidence_band"] < small["confidence_band"]


class TestStationarity:
    def test_returns_are_stationary_and_log_price_is_not(self):
        closes = random_walk(500, seed=31, mu=0.0003)
        v = run_method("stationarity", series_input(closes)).values
        assert v["returns_stationary"] is True
        assert v["price_has_unit_root"] is True


class TestVolatilityClustering:
    def test_constant_volatility_shows_no_arch_effects(self):
        v = run_method(
            "volatility_clustering", series_input(random_walk(800, seed=41))
        ).values
        assert v["arch_effects_present_at_5pct"] is False

    def test_clustered_volatility_is_detected(self):
        """A genuine GARCH(1,1) process must trip the ARCH test.

        The variance has to depend on the previous *squared return*, not on an
        independent draw: a process whose volatility wanders without feeding
        back from realised returns has persistent sigma but no detectable ARCH
        effect, which is exactly what the test is supposed to distinguish.
        """
        rng = random.Random(7)
        omega, alpha, beta = 1e-6, 0.12, 0.85
        var = omega / (1 - alpha - beta)
        closes = [100.0]
        prev_shock = 0.0
        for _ in range(1500):
            var = omega + alpha * (prev_shock**2) + beta * var
            prev_shock = rng.gauss(0, math.sqrt(var))
            closes.append(closes[-1] * math.exp(prev_shock))
        v = run_method("volatility_clustering", series_input(closes)).values
        assert v["arch_effects_present_at_5pct"] is True
        assert v["abs_return_autocorr_lag1"] > 0.05


class TestChangePoint:
    def test_a_volatility_regime_shift_is_located(self):
        calm = random_walk(200, seed=51, sigma=0.005)
        rng = random.Random(52)
        turbulent = [calm[-1]]
        for _ in range(200):
            turbulent.append(turbulent[-1] * math.exp(rng.gauss(0, 0.04)))
        closes = calm + turbulent[1:]
        v = run_method("change_point", series_input(closes)).values
        assert v["change_point_index"] is not None
        # Should land near the true break at index ~200.
        assert 150 < v["change_point_index"] < 250
        assert v["volatility_after"] > v["volatility_before"] * 3

    def test_the_corrected_p_value_is_reported_and_larger(self):
        closes = random_walk(300, seed=53)
        result = run_method("change_point", series_input(closes))
        v = result.values
        if v["f_test_p"] is not None:
            assert v["f_test_p_bonferroni"] >= v["f_test_p"]
            assert v["candidate_splits_searched"] > 0
            assert any("candidate split" in w for w in result.warnings)

    def test_a_homogeneous_series_is_not_declared_a_regime_change(self):
        closes = random_walk(400, seed=54)
        v = run_method("change_point", series_input(closes)).values
        # The scan always returns its best split; the corrected p-value is what
        # decides whether it means anything.
        assert v["f_test_p_bonferroni"] > 0.05


class TestExtremeValue:
    def test_historical_var_cannot_exceed_the_worst_observed_loss(self):
        closes = random_walk(500, seed=61)
        result = run_method("extreme_value", series_input(closes))
        v = result.values
        assert v["historical_var"]["0.99"] >= v["worst_observed_return"]
        assert any("worst observed return" in w for w in result.warnings)

    def test_expected_shortfall_is_worse_than_var(self):
        v = run_method("extreme_value", series_input(random_walk(600, seed=62))).values
        for c in ("0.95", "0.99"):
            assert v["historical_expected_shortfall"][c] <= v["historical_var"][c]

    def test_var_at_99_is_worse_than_at_95(self):
        v = run_method("extreme_value", series_input(random_walk(600, seed=63))).values
        assert v["historical_var"]["0.99"] < v["historical_var"]["0.95"]

    def test_a_normal_model_misses_the_extreme_tail(self):
        """The real tail-risk failure, stated correctly.

        A naive version of this test asserts that normal VaR at 99% is
        shallower than historical VaR. That is false for a contaminated-normal
        mixture: frequent large shocks inflate the standard deviation enough
        that the normal 99% quantile overshoots the empirical one. The failure
        a normal model actually has is further out -- the worst realised loss
        is a far larger multiple of its VaR than Gaussian tails allow.
        """
        rng = random.Random(64)
        fat = [100.0]
        for i in range(2000):
            shock = rng.gauss(0, 0.01) if i % 200 else rng.gauss(0, 0.09)
            fat.append(fat[-1] * math.exp(shock))

        fat_v = run_method("extreme_value", series_input(fat)).values
        normal_v = run_method("extreme_value", series_input(random_walk(2000, seed=66))).values

        fat_ratio = fat_v["worst_observed_return"] / fat_v["normal_var"]["0.99"]
        normal_ratio = normal_v["worst_observed_return"] / normal_v["normal_var"]["0.99"]
        assert fat_ratio > normal_ratio * 1.5, (
            "the worst loss in a fat-tailed series must be a much larger multiple "
            "of its normal-model VaR than in a Gaussian one"
        )
        assert fat_v["excess_kurtosis"] > normal_v["excess_kurtosis"] + 2

    def test_cornish_fisher_is_flagged_as_unreliable_at_extreme_moments(self):
        """The expansion breaks down for large skew/kurtosis, and must say so."""
        rng = random.Random(64)
        closes = [100.0]
        for i in range(800):
            shock = rng.gauss(0, 0.008) if i % 40 else rng.gauss(0, 0.10)
            closes.append(closes[-1] * math.exp(shock))
        result = run_method("extreme_value", series_input(closes))
        assert result.values["excess_kurtosis"] > 10
        assert any("Cornish-Fisher expansion is reliable" in w for w in result.warnings)
        # And the figure really is unusable here, which is why the warning matters.
        assert result.values["cornish_fisher_var"]["0.99"] < result.values[
            "historical_var"
        ]["0.99"] * 3

    def test_short_series_is_refused(self):
        with pytest.raises(InsufficientDataError, match="at least 100"):
            run_method("extreme_value", series_input(random_walk(50, seed=65)))


class TestCorrelation:
    def test_a_series_correlates_perfectly_with_itself(self):
        closes = random_walk(200, seed=71)
        v = run_method("correlation_matrix", series_input(closes, benchmark=closes)).values
        assert v["pearson"]["benchmark"] == pytest.approx(1.0)
        assert v["spearman"]["benchmark"] == pytest.approx(1.0)

    def test_independent_series_are_uncorrelated(self):
        a = random_walk(600, seed=72)
        b = random_walk(600, seed=73)
        v = run_method("correlation_matrix", series_input(a, benchmark=b)).values
        assert abs(v["pearson"]["benchmark"]) < 0.15

    def test_misaligned_series_is_named_not_truncated(self):
        closes = random_walk(100, seed=74)
        result = run_method(
            "correlation_matrix", series_input(closes, series={"short": [1.0, 2.0, 3.0]})
        )
        assert result.values["pearson"]["short"] is None
        assert any("skipped rather than truncated" in w for w in result.warnings)

    def test_zero_variance_series_is_handled(self):
        closes = random_walk(100, seed=75)
        result = run_method(
            "correlation_matrix", series_input(closes, series={"flat": [50.0] * 100})
        )
        assert result.values["pearson"]["flat"] is None
        assert any("zero variance" in w for w in result.warnings)


class TestMultipleTesting:
    def test_bonferroni_scales_by_the_family_size(self):
        v = run_method(
            "multiple_testing",
            series_input([100.0, 101.0], scalars={"n_tests": 20}),
            {"p_values": [0.01, 0.04]},
        ).values
        assert v["bonferroni"] == [pytest.approx(0.20), pytest.approx(0.80)]
        assert v["survives_bonferroni"] == [False, False]

    def test_a_strong_result_survives_correction(self):
        v = run_method(
            "multiple_testing",
            series_input([100.0, 101.0], scalars={"n_tests": 10}),
            {"p_values": [0.001]},
        ).values
        assert v["survives_bonferroni"] == [True]

    def test_benjamini_hochberg_is_monotonic_in_rank(self):
        p_values = [0.001, 0.008, 0.02, 0.04, 0.3]
        v = run_method(
            "multiple_testing",
            series_input([100.0, 101.0], scalars={"n_tests": 5}),
            {"p_values": p_values},
        ).values
        bh = v["benjamini_hochberg"]
        assert bh == sorted(bh), "BH-adjusted values must not decrease with raw rank"
        assert all(b >= p for b, p in zip(bh, p_values, strict=True))

    def test_bh_is_less_conservative_than_bonferroni(self):
        v = run_method(
            "multiple_testing",
            series_input([100.0, 101.0], scalars={"n_tests": 10}),
            {"p_values": [0.001, 0.002, 0.003]},
        ).values
        assert all(
            bh <= bonf
            for bh, bonf in zip(v["benjamini_hochberg"], v["bonferroni"], strict=True)
        )


class TestSeasonality:
    def test_random_data_does_not_survive_correction(self):
        """The data-mining guard.

        With 17 calendar buckets something will look significant uncorrected.
        The corrected verdict must say no.
        """
        closes = random_walk(1500, seed=81)
        result = run_method("seasonality", series_input(closes))
        v = result.values
        assert v["any_survives_correction"] is False
        assert v["tests_performed"] >= 5
        assert any("Bonferroni-corrected" in w for w in result.warnings)

    def test_bucket_counts_are_reported(self):
        v = run_method("seasonality", series_input(random_walk(800, seed=82))).values
        for bucket in v["by_weekday"].values():
            assert "n" in bucket
        assert sum(b["n"] for b in v["by_weekday"].values()) == v["observations"]

    def test_corrected_p_values_are_attached_per_bucket(self):
        v = run_method("seasonality", series_input(random_walk(900, seed=83))).values
        for bucket in v["by_month"].values():
            if bucket["p_value"] is not None:
                assert bucket["p_value_bonferroni"] >= bucket["p_value"]
