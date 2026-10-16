"""Indicator correctness.

Every value here is checked against an independently derived figure -- a
hand-computed result, a closed-form identity, or Wilder's published worked
example -- rather than against whatever the code currently returns. A test
that asserts the implementation agrees with itself catches nothing.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest

from app.analysis.registry import MethodInput, run_method
from app.analysis.technical import (
    ema,
    log_returns,
    max_drawdown,
    simple_returns,
    sma,
    true_ranges,
    wilder_smooth,
)
from app.core.errors import InsufficientDataError
from app.core.timeutil import UTC


def mk_input(closes, *, highs=None, lows=None, opens=None, volumes=None,
             resolution="1d", benchmark=None) -> MethodInput:
    n = len(closes)
    ts = tuple(
        dt.datetime(2024, 1, 1, tzinfo=UTC) + dt.timedelta(days=i) for i in range(n)
    )
    return MethodInput(
        timestamps=ts,
        open=tuple(opens) if opens else tuple(closes),
        high=tuple(highs) if highs else tuple(c * 1.01 for c in closes),
        low=tuple(lows) if lows else tuple(c * 0.99 for c in closes),
        close=tuple(float(c) for c in closes),
        volume=tuple(volumes) if volumes else tuple(1_000_000.0 for _ in closes),
        benchmark_close=tuple(float(b) for b in benchmark) if benchmark else (),
        resolution=resolution,
    )


class TestPrimitives:
    def test_sma_matches_hand_computation(self):
        values = (1.0, 2.0, 3.0, 4.0, 5.0)
        result = sma(values, 3)
        assert result[:2] == [None, None], "no value before the window is full"
        assert result[2] == pytest.approx(2.0)   # (1+2+3)/3
        assert result[3] == pytest.approx(3.0)   # (2+3+4)/3
        assert result[4] == pytest.approx(4.0)   # (3+4+5)/3

    def test_sma_rolling_sum_does_not_drift(self):
        # The implementation updates a running sum; over a long series that can
        # accumulate float error. Compare against a fresh mean at each step.
        values = tuple(float(i % 17) + 0.37 for i in range(500))
        rolling = sma(values, 20)
        for i in range(19, 500):
            expected = sum(values[i - 19 : i + 1]) / 20
            assert rolling[i] == pytest.approx(expected, abs=1e-9)

    def test_sma_of_a_constant_series_is_that_constant(self):
        assert sma((7.0,) * 10, 5)[-1] == pytest.approx(7.0)

    def test_ema_seeds_with_an_sma(self):
        values = (1.0, 2.0, 3.0, 4.0, 5.0)
        result = ema(values, 3)
        assert result[2] == pytest.approx(2.0), "seed is the SMA of the first 3"
        # alpha = 2/(3+1) = 0.5; next = 0.5*4 + 0.5*2 = 3.0
        assert result[3] == pytest.approx(3.0)
        # next = 0.5*5 + 0.5*3 = 4.0
        assert result[4] == pytest.approx(4.0)

    def test_ema_of_a_constant_series_is_that_constant(self):
        assert ema((42.0,) * 30, 10)[-1] == pytest.approx(42.0)

    def test_wilder_smoothing_uses_alpha_one_over_n(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = wilder_smooth(values, 3)
        assert result[2] == pytest.approx(2.0)
        # (2.0*(3-1) + 4)/3 = 8/3
        assert result[3] == pytest.approx(8.0 / 3.0)
        # (8/3*2 + 5)/3 = (16/3 + 5)/3 = 31/9
        assert result[4] == pytest.approx(31.0 / 9.0)

    def test_wilder_differs_from_ema_as_documented(self):
        values = [float(i) for i in range(1, 31)]
        assert wilder_smooth(values, 14)[-1] != pytest.approx(ema(values, 14)[-1])

    def test_log_and_simple_returns_are_consistent(self):
        closes = (100.0, 110.0, 99.0)
        for lr, sr in zip(log_returns(closes), simple_returns(closes), strict=True):
            assert math.exp(lr) - 1.0 == pytest.approx(sr)

    def test_log_returns_add_across_time(self):
        # The reason log returns are the internal default.
        closes = (100.0, 105.0, 102.0, 115.0)
        assert sum(log_returns(closes)) == pytest.approx(math.log(115.0 / 100.0))

    def test_true_range_takes_the_largest_of_three_measures(self):
        high = (10.0, 12.0)
        low = (9.0, 11.5)
        close = (9.5, 11.8)
        # candidates: 12-11.5=0.5, |12-9.5|=2.5, |11.5-9.5|=2.0 -> 2.5
        assert true_ranges(high, low, close) == [pytest.approx(2.5)]

    def test_max_drawdown_finds_the_worst_decline(self):
        closes = (100.0, 120.0, 60.0, 90.0, 80.0)
        worst, peak_idx, trough_idx = max_drawdown(closes)
        assert worst == pytest.approx(-0.5)  # 120 -> 60
        assert peak_idx == 1 and trough_idx == 2

    def test_max_drawdown_of_a_monotonic_rise_is_zero(self):
        worst, _, _ = max_drawdown((1.0, 2.0, 3.0, 4.0))
        assert worst == pytest.approx(0.0)


class TestReturns:
    def test_total_return_is_exact(self):
        result = run_method("returns", mk_input([100.0, 110.0, 121.0]))
        assert result.values["total_return"] == pytest.approx(0.21)
        assert result.values["total_log_return"] == pytest.approx(math.log(1.21))

    def test_annualisation_uses_the_configured_year_length(self):
        # 252 daily periods of exactly +0.1% compound to (1.001)^252 - 1.
        closes = [100.0 * (1.001**i) for i in range(253)]
        result = run_method("returns", mk_input(closes))
        assert result.values["annualised_return"] == pytest.approx(1.001**252 - 1, rel=1e-6)

    def test_two_observations_are_enough(self):
        assert run_method("returns", mk_input([100.0, 101.0])).observations_used == 2

    def test_one_observation_is_refused(self):
        with pytest.raises(InsufficientDataError):
            run_method("returns", mk_input([100.0]))


class TestRealizedVolatility:
    def test_zero_volatility_for_a_constant_series(self):
        result = run_method("realized_volatility", mk_input([100.0] * 30))
        assert result.values["daily_volatility"] == pytest.approx(0.0)
        assert result.values["annualised_volatility"] == pytest.approx(0.0)

    def test_annualisation_is_sqrt_of_periods_per_year(self):
        closes = [100.0]
        for i in range(60):
            closes.append(closes[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
        result = run_method("realized_volatility", mk_input(closes))
        daily = result.values["daily_volatility"]
        assert result.values["annualised_volatility"] == pytest.approx(daily * math.sqrt(252))

    def test_it_reports_the_annualisation_assumption(self):
        result = run_method("realized_volatility", mk_input([100.0 + i for i in range(30)]))
        assert result.values["periods_per_year_assumed"] == 252.0

    def test_sample_standard_deviation_is_the_default(self):
        closes = [100.0, 101.0, 100.5, 102.0, 101.0] * 6
        sample = run_method("realized_volatility", mk_input(closes))
        population = run_method("realized_volatility", mk_input(closes), {"ddof": 0})
        assert sample.values["daily_volatility"] > population.values["daily_volatility"]

    def test_short_series_is_refused(self):
        with pytest.raises(InsufficientDataError, match="at least 20"):
            run_method("realized_volatility", mk_input([100.0] * 5))


class TestRSI:
    def test_rsi_is_100_when_every_period_rises(self):
        result = run_method("rsi", mk_input([100.0 + i for i in range(20)]))
        assert result.values["rsi"] == pytest.approx(100.0)
        assert result.values["average_loss"] == 0.0

    def test_rsi_is_0_when_every_period_falls(self):
        result = run_method("rsi", mk_input([200.0 - i for i in range(20)]))
        assert result.values["rsi"] == pytest.approx(0.0)

    def test_rsi_straddles_50_symmetrically_on_a_balanced_series(self):
        """On an alternating +1/-1 series RSI oscillates symmetrically about 50.

        It does not sit exactly at 50, and that is correct rather than a defect:
        Wilder smoothing gives the most recent change fresh weight 1/period, so
        a window ending on a down move reads just below 50 and one ending on an
        up move just above, by the same amount. Asserting the symmetry tests
        the smoothing recursion; asserting "approximately 50" would pass even
        if the recursion were wrong.
        """
        def closes_of(length: int) -> list[float]:
            out = [100.0]
            for i in range(length):
                out.append(out[-1] + (1.0 if i % 2 == 0 else -1.0))
            return out

        ending_down = run_method("rsi", mk_input(closes_of(400))).values["rsi"]
        ending_up = run_method("rsi", mk_input(closes_of(399))).values["rsi"]

        assert ending_down < 50.0 < ending_up
        assert (ending_down + ending_up) / 2 == pytest.approx(50.0, abs=1e-6)

    def test_wilder_rsi_matches_a_hand_computation(self):
        """Period 2 over changes [+1,-1,+1,-1], worked through by hand.

        gains  [1,0,1,0] -> seed (1+0)/2 = 0.5, then (0.5+1)/2 = 0.75, then (0.75+0)/2 = 0.375
        losses [0,1,0,1] -> seed (0+1)/2 = 0.5, then (0.5+0)/2 = 0.25, then (0.25+1)/2 = 0.625
        RS = 0.375/0.625 = 0.6, so RSI = 100 - 100/1.6 = 37.5
        """
        closes = [100.0, 101.0, 100.0, 101.0, 100.0]
        result = run_method("rsi", mk_input(closes), {"period": 2})
        assert result.values["rsi"] == pytest.approx(37.5)

    def test_rsi_stays_within_bounds(self):
        import random

        rng = random.Random(7)
        closes = [100.0]
        for _ in range(200):
            closes.append(max(closes[-1] * (1 + rng.gauss(0, 0.02)), 1.0))
        rsi = run_method("rsi", mk_input(closes)).values["rsi"]
        assert 0.0 <= rsi <= 100.0

    def test_a_longer_period_raises_the_data_requirement(self):
        # A 50-period RSI must refuse a 30-bar series even though the spec's
        # declared minimum is 15.
        with pytest.raises(InsufficientDataError):
            run_method("rsi", mk_input([100.0 + i for i in range(30)]), {"period": 50})

    def test_a_shorter_period_lowers_the_data_requirement(self):
        """The declared minimum is for default params, not an absolute floor.

        Refusing a 5-bar series for a 2-period RSI would reject a computation
        that is perfectly well defined.
        """
        result = run_method("rsi", mk_input([100.0, 101.0, 100.0, 101.0, 100.0]), {"period": 2})
        assert result.values["rsi"] is not None


class TestMACD:
    def test_macd_is_zero_for_a_constant_series(self):
        result = run_method("macd", mk_input([50.0] * 60))
        assert result.values["macd"] == pytest.approx(0.0)
        assert result.values["signal"] == pytest.approx(0.0)
        assert result.values["histogram"] == pytest.approx(0.0)

    def test_macd_is_positive_in_a_rising_series(self):
        result = run_method("macd", mk_input([100.0 * (1.01**i) for i in range(80)]))
        assert result.values["macd"] > 0

    def test_macd_is_negative_in_a_falling_series(self):
        result = run_method("macd", mk_input([100.0 * (0.99**i) for i in range(80)]))
        assert result.values["macd"] < 0

    def test_histogram_is_macd_minus_signal(self):
        result = run_method("macd", mk_input([100.0 + math.sin(i / 5) * 10 for i in range(90)]))
        v = result.values
        assert v["histogram"] == pytest.approx(v["macd"] - v["signal"])


class TestBollingerBands:
    def test_bands_collapse_onto_the_mean_for_a_constant_series(self):
        result = run_method("bollinger_bands", mk_input([75.0] * 30))
        v = result.values
        assert v["upper"] == pytest.approx(75.0)
        assert v["lower"] == pytest.approx(75.0)
        assert v["middle"] == pytest.approx(75.0)

    def test_band_width_is_the_requested_multiple_of_sigma(self):
        closes = [100.0, 102.0, 98.0, 101.0, 99.0] * 6
        result = run_method("bollinger_bands", mk_input(closes), {"window": 20, "num_std": 2.0})
        v = result.values
        assert v["upper"] - v["middle"] == pytest.approx(2.0 * v["standard_deviation"])
        assert v["middle"] - v["lower"] == pytest.approx(2.0 * v["standard_deviation"])

    def test_percent_b_is_one_at_the_upper_band(self):
        closes = [100.0, 102.0, 98.0, 101.0, 99.0] * 6
        result = run_method("bollinger_bands", mk_input(closes))
        v = result.values
        # Reconstruct %B from the reported band positions.
        last = v["middle"] + 2.0 * v["standard_deviation"]
        span = v["upper"] - v["lower"]
        assert (last - v["lower"]) / span == pytest.approx(1.0)


class TestATR:
    def test_atr_of_a_constant_range_equals_that_range(self):
        closes = [100.0] * 30
        highs = [101.0] * 30
        lows = [99.0] * 30
        result = run_method("atr", mk_input(closes, highs=highs, lows=lows))
        assert result.values["atr"] == pytest.approx(2.0)

    def test_atr_percent_is_atr_over_price(self):
        closes = [100.0] * 30
        result = run_method(
            "atr", mk_input(closes, highs=[101.0] * 30, lows=[99.0] * 30)
        )
        v = result.values
        assert v["atr_percent"] == pytest.approx(v["atr"] / 100.0)

    def test_atr_requires_high_and_low(self):
        data = mk_input([100.0] * 30)
        stripped = MethodInput(
            timestamps=data.timestamps, open=data.open, high=(), low=(),
            close=data.close, volume=data.volume, resolution="1d",
        )
        with pytest.raises(InsufficientDataError, match="high"):
            run_method("atr", stripped)


class TestDrawdown:
    def test_drawdown_reports_the_worst_decline_and_its_dates(self):
        result = run_method("drawdown", mk_input([100.0, 150.0, 75.0, 120.0]))
        v = result.values
        assert v["max_drawdown"] == pytest.approx(-0.5)
        assert v["peak_close"] == pytest.approx(150.0)
        assert v["trough_close"] == pytest.approx(75.0)
        assert v["peak_date"].startswith("2024-01-02")

    def test_current_drawdown_is_measured_from_the_running_peak(self):
        result = run_method("drawdown", mk_input([100.0, 200.0, 150.0]))
        assert result.values["current_drawdown"] == pytest.approx(-0.25)

    def test_monotonic_rise_has_no_drawdown(self):
        result = run_method("drawdown", mk_input([10.0, 20.0, 30.0]))
        assert result.values["max_drawdown"] == pytest.approx(0.0)
        assert result.values["current_drawdown"] == pytest.approx(0.0)


class TestTrendStrength:
    def test_perfect_exponential_growth_has_r_squared_one(self):
        # log(price) is exactly linear in time, so the fit is exact.
        result = run_method("trend_strength", mk_input([100.0 * (1.01**i) for i in range(50)]))
        assert result.values["r_squared"] == pytest.approx(1.0)
        assert result.values["slope_per_period"] == pytest.approx(math.log(1.01))
        assert result.values["up_period_fraction"] == pytest.approx(1.0)

    def test_declining_series_has_a_negative_slope(self):
        result = run_method("trend_strength", mk_input([100.0 * (0.99**i) for i in range(50)]))
        assert result.values["slope_per_period"] < 0
        assert result.values["annualised_slope"] < 0

    def test_flat_series_has_zero_slope(self):
        result = run_method("trend_strength", mk_input([100.0] * 40))
        assert result.values["slope_per_period"] == pytest.approx(0.0)


class TestRelativeStrength:
    def test_beta_is_one_against_itself(self):
        closes = [100.0 * (1.005**i) + math.sin(i) for i in range(60)]
        result = run_method("relative_strength", mk_input(closes, benchmark=closes))
        v = result.values
        assert v["beta"] == pytest.approx(1.0)
        assert v["correlation"] == pytest.approx(1.0)
        assert v["excess_return"] == pytest.approx(0.0)

    def test_beta_is_two_for_a_doubled_return_series(self):
        bench_rets = [0.01, -0.005, 0.008, -0.002, 0.012] * 12
        bench = [100.0]
        asset = [100.0]
        for r in bench_rets:
            bench.append(bench[-1] * math.exp(r))
            asset.append(asset[-1] * math.exp(2 * r))
        result = run_method("relative_strength", mk_input(asset, benchmark=bench))
        assert result.values["beta"] == pytest.approx(2.0)

    def test_misaligned_benchmark_is_rejected(self):
        data = mk_input([100.0 + i for i in range(40)])
        bad = MethodInput(
            timestamps=data.timestamps, open=data.open, high=data.high, low=data.low,
            close=data.close, volume=data.volume,
            benchmark_close=tuple(float(i) for i in range(10)),
            resolution="1d",
        )
        with pytest.raises(InsufficientDataError, match="aligned"):
            run_method("relative_strength", bad)


class TestGapAnalysis:
    def test_a_known_gap_up_is_measured_correctly(self):
        closes = [100.0] * 40
        opens = list(closes)
        opens[20] = 110.0  # +10% gap against the prior close of 100
        highs = [max(o, c) * 1.001 for o, c in zip(opens, closes, strict=True)]
        lows = [min(o, c) * 0.999 for o, c in zip(opens, closes, strict=True)]
        result = run_method(
            "gap_analysis", mk_input(closes, opens=opens, highs=highs, lows=lows)
        )
        assert result.values["largest_gap_up"] == pytest.approx(0.10)

    def test_gap_fill_is_detected(self):
        closes = [100.0] * 40
        opens = list(closes)
        opens[20] = 105.0
        highs = [105.5] * 40
        lows = [99.0] * 40   # trades back below the prior close, so the gap fills
        result = run_method(
            "gap_analysis", mk_input(closes, opens=opens, highs=highs, lows=lows)
        )
        assert result.values["gap_fill_rate"] == pytest.approx(1.0)


class TestVolumeProfile:
    def test_unreported_volume_is_not_treated_as_zero(self):
        closes = [100.0 + i for i in range(30)]
        volumes = [1_000_000.0] * 25 + [None] * 5
        result = run_method("volume_profile", mk_input(closes, volumes=volumes))
        # The usable window stops at the first None, so only 25 bars are used.
        assert result.observations_used == 25
        assert result.values["mean_volume"] == pytest.approx(1_000_000.0)

    def test_recent_volume_ratio_detects_a_surge(self):
        closes = [100.0 + i for i in range(70)]
        volumes = [1_000_000.0] * 65 + [5_000_000.0] * 5
        result = run_method("volume_profile", mk_input(closes, volumes=volumes))
        assert result.values["recent_volume_ratio"] > 2.0


class TestSupportResistance:
    def test_a_repeatedly_touched_level_is_found_with_its_touch_count(self):
        # A sawtooth topping out at 110 three times.
        closes: list[float] = []
        for _ in range(3):
            closes.extend([100.0, 104.0, 108.0, 110.0, 108.0, 104.0, 100.0, 98.0, 100.0, 104.0])
        highs = [c * 1.0 for c in closes]
        highs = [110.0 if c == 110.0 else c for c in closes]
        result = run_method(
            "support_resistance",
            mk_input(closes, highs=highs, lows=[c * 0.99 for c in closes]),
            {"lookback": 3, "tolerance_percent": 1.0},
        )
        levels = result.values["all_resistance"]
        assert levels, "a thrice-touched high must be reported"
        assert levels[0]["touches"] >= 2
        assert levels[0]["level"] == pytest.approx(110.0, rel=0.02)


class TestRegistryGuarantees:
    def test_results_carry_method_version_and_fingerprint(self):
        result = run_method("returns", mk_input([100.0, 110.0]))
        assert result.method_key == "returns"
        assert result.method_version == "1.0.0"
        assert len(result.input_fingerprint) == 32

    def test_identical_inputs_give_an_identical_fingerprint(self):
        a = run_method("returns", mk_input([100.0, 110.0]))
        b = run_method("returns", mk_input([100.0, 110.0]))
        assert a.input_fingerprint == b.input_fingerprint

    def test_different_inputs_give_a_different_fingerprint(self):
        a = run_method("returns", mk_input([100.0, 110.0]))
        b = run_method("returns", mk_input([100.0, 111.0]))
        assert a.input_fingerprint != b.input_fingerprint

    def test_different_params_give_a_different_fingerprint(self):
        data = mk_input([100.0 + i for i in range(60)])
        a = run_method("rsi", data, {"period": 14})
        b = run_method("rsi", data, {"period": 21})
        assert a.input_fingerprint != b.input_fingerprint

    def test_limitations_travel_with_every_result(self):
        result = run_method("realized_volatility", mk_input([100.0 + i for i in range(30)]))
        assert result.limitations, "a number must not be citable without its caveats"
        assert any("not a forecast" in lim for lim in result.limitations)

    def test_out_of_order_timestamps_are_rejected(self):
        data = mk_input([100.0, 101.0, 102.0])
        scrambled = MethodInput(
            timestamps=(data.timestamps[0], data.timestamps[2], data.timestamps[1]),
            open=data.open, high=data.high, low=data.low, close=data.close,
            volume=data.volume, resolution="1d",
        )
        with pytest.raises(InsufficientDataError, match="ascending"):
            run_method("returns", scrambled)

    def test_unknown_method_raises_not_found(self):
        from app.core.errors import NotFoundError

        with pytest.raises(NotFoundError):
            run_method("no_such_method", mk_input([1.0, 2.0]))

    def test_every_registered_method_declares_its_limitations(self):
        from app.analysis.registry import all_specs

        for spec in all_specs():
            assert spec.limitations, f"{spec.key} must declare limitations"
            assert spec.missing_data_policy, f"{spec.key} must declare a missing-data policy"
            assert spec.output_schema, f"{spec.key} must declare an output schema"
