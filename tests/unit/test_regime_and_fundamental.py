"""Regime estimation, similarity features and fundamental ratio tests."""

from __future__ import annotations

import datetime as dt
import math
import random

import pytest

from app.analysis.regime import FEATURE_SET_VERSION, feature_distance
from app.analysis.registry import MethodInput, run_method
from app.core.timeutil import UTC
from app.db.models.enums import RegimeLabel


def mk(closes, *, volumes=None, highs=None, lows=None, scalars=None, series=None):
    n = len(closes)
    ts = tuple(dt.datetime(2022, 1, 3, tzinfo=UTC) + dt.timedelta(days=i) for i in range(n))
    return MethodInput(
        timestamps=ts,
        open=tuple(float(c) for c in closes),
        high=tuple(float(h) for h in highs) if highs else tuple(float(c) * 1.01 for c in closes),
        low=tuple(float(x) for x in lows) if lows else tuple(float(c) * 0.99 for c in closes),
        close=tuple(float(c) for c in closes),
        volume=tuple(volumes) if volumes else tuple(1_000_000.0 for _ in closes),
        scalars=dict(scalars or {}),
        series={k: tuple(float(x) for x in v) for k, v in (series or {}).items()},
        resolution="1d",
    )


def walk(n, *, seed=1, sigma=0.012, mu=0.0, start=100.0):
    rng = random.Random(seed)
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * math.exp(rng.gauss(mu, sigma)))
    return out


class TestRegimeClassification:
    def test_a_clean_uptrend_is_labelled_trending_up(self):
        closes = [100.0 * (1.004**i) for i in range(200)]
        v = run_method("regime_classification", mk(closes)).values
        assert v["label"] == RegimeLabel.TRENDING_UP
        assert v["diagnostics"]["trend_r_squared"] > 0.99
        assert v["diagnostics"]["trend_slope_per_period"] > 0

    def test_a_clean_downtrend_is_labelled_trending_down(self):
        closes = [100.0 * (0.996**i) for i in range(200)]
        v = run_method("regime_classification", mk(closes)).values
        assert v["label"] == RegimeLabel.TRENDING_DOWN

    def test_a_flat_oscillation_is_not_called_trending(self):
        closes = [100.0 + 5 * math.sin(i / 8) for i in range(200)]
        v = run_method("regime_classification", mk(closes)).values
        assert v["label"] != RegimeLabel.TRENDING_UP
        assert v["label"] != RegimeLabel.TRENDING_DOWN
        assert v["diagnostics"]["trend_r_squared"] < 0.5

    def test_a_volatility_expansion_is_detected(self):
        calm = walk(250, seed=5, sigma=0.004)
        rng = random.Random(6)
        loud = [calm[-1]]
        for _ in range(30):
            loud.append(loud[-1] * math.exp(rng.gauss(0, 0.05)))
        v = run_method("regime_classification", mk(calm + loud[1:])).values
        assert v["diagnostics"]["volatility_ratio"] > 1.5

    def test_thresholds_are_reported_so_the_label_can_be_disputed(self):
        v = run_method("regime_classification", mk(walk(150, seed=7))).values
        assert "thresholds_used" in v
        assert "trend_r2_threshold" in v["thresholds_used"]

    def test_probabilities_sum_to_one(self):
        v = run_method("regime_classification", mk(walk(200, seed=8))).values
        assert sum(v["label_probabilities"].values()) == pytest.approx(1.0)

    def test_a_near_tie_is_reported_as_a_transition(self):
        # Construct a window where trend and volatility scores are close.
        closes = [100.0 * (1.0008**i) + 3 * math.sin(i / 6) for i in range(200)]
        v = run_method("regime_classification", mk(closes)).values
        ranked = sorted(v["label_probabilities"].values(), reverse=True)
        if len(ranked) > 1 and ranked[0] - ranked[1] < 0.15:
            assert v["label"] == RegimeLabel.TRANSITION

    def test_limitations_state_that_regimes_are_not_observable(self):
        result = run_method("regime_classification", mk(walk(150, seed=9)))
        assert any("not observable" in lim for lim in result.limitations)


class TestSimilarityFeatures:
    def test_features_are_scale_invariant(self):
        """The property the whole similarity engine rests on.

        A $4 stock and a $400 stock with the same shape must produce
        essentially the same vector, otherwise matching only ever finds
        instruments at a similar price level.
        """
        base = walk(200, seed=11, start=100.0)
        scaled = [c * 100.0 for c in base]
        a = run_method("similarity_features", mk(base)).values["features"]
        b = run_method("similarity_features", mk(scaled)).values["features"]
        for key in set(a) & set(b):
            assert a[key] == pytest.approx(b[key], rel=1e-6, abs=1e-9), (
                f"feature {key} changed when the price level was scaled"
            )

    def test_distance_between_a_series_and_itself_is_zero(self):
        v = run_method("similarity_features", mk(walk(200, seed=12))).values["features"]
        distance, compared = feature_distance(v, v)
        assert distance == pytest.approx(0.0)
        assert compared == len(v)

    def test_an_uptrend_is_farther_from_a_downtrend_than_from_another_uptrend(self):
        up_a = run_method(
            "similarity_features", mk([100.0 * (1.003**i) for i in range(200)])
        ).values["features"]
        up_b = run_method(
            "similarity_features", mk([50.0 * (1.0028**i) for i in range(200)])
        ).values["features"]
        down = run_method(
            "similarity_features", mk([100.0 * (0.997**i) for i in range(200)])
        ).values["features"]
        d_same, _ = feature_distance(up_a, up_b)
        d_diff, _ = feature_distance(up_a, down)
        assert d_same < d_diff

    def test_the_feature_count_compared_is_reported(self):
        a = run_method("similarity_features", mk(walk(100, seed=13))).values["features"]
        partial = {k: a[k] for k in list(a)[:3]}
        _, compared = feature_distance(a, partial)
        assert compared == 3, (
            "a caller must be able to see a match was made on only three features"
        )

    def test_missing_volume_omits_the_feature_rather_than_defaulting_it(self):
        closes = walk(120, seed=14)
        result = run_method("similarity_features", mk(closes, volumes=[None] * 120))
        assert "volume_ratio_recent_to_window" not in result.values["features"]
        assert "volume_ratio_recent_to_window" in result.values["omitted_features"]
        assert any("omitted" in w for w in result.warnings)

    def test_the_feature_set_version_travels_with_the_vector(self):
        v = run_method("similarity_features", mk(walk(100, seed=15))).values
        assert v["feature_set_version"] == FEATURE_SET_VERSION

    def test_disjoint_vectors_give_infinite_distance(self):
        distance, compared = feature_distance({"a": 1.0}, {"b": 2.0})
        assert distance == float("inf")
        assert compared == 0


class TestProfitability:
    def test_margins_match_hand_computation(self):
        v = run_method(
            "profitability",
            mk([100.0], scalars={
                "revenue": 1000.0, "gross_profit": 400.0,
                "operating_income": 250.0, "net_income": 200.0,
                "stockholders_equity": 800.0,
            }),
        ).values
        assert v["gross_margin"] == pytest.approx(0.40)
        assert v["operating_margin"] == pytest.approx(0.25)
        assert v["net_margin"] == pytest.approx(0.20)
        assert v["return_on_equity"] == pytest.approx(0.25)

    def test_gross_profit_is_derived_when_absent(self):
        v = run_method(
            "profitability",
            mk([100.0], scalars={"revenue": 1000.0, "cost_of_revenue": 600.0}),
        ).values
        assert v["gross_margin"] == pytest.approx(0.40)

    def test_a_missing_input_gives_null_not_zero(self):
        v = run_method("profitability", mk([100.0], scalars={"revenue": 1000.0})).values
        assert v["net_margin"] is None, (
            "'could not compute' must not be reported as 0%"
        )

    def test_negative_equity_suppresses_roe_with_an_explanation(self):
        result = run_method(
            "profitability",
            mk([100.0], scalars={
                "revenue": 1000.0, "net_income": 100.0,
                "stockholders_equity": -500.0,
            }),
        )
        assert result.values["return_on_equity"] is None
        assert any("negative" in w for w in result.warnings)

    def test_the_invested_capital_definition_is_disclosed(self):
        v = run_method(
            "profitability",
            mk([100.0], scalars={
                "revenue": 1000.0, "operating_income": 200.0,
                "stockholders_equity": 500.0, "long_term_debt": 300.0,
                "cash_and_equivalents": 100.0,
            }),
        ).values
        assert v["invested_capital"] == pytest.approx(700.0)  # 500 + 300 - 100
        assert "stockholders_equity" in v["invested_capital_definition"]
        assert v["return_on_invested_capital"] == pytest.approx(200.0 / 700.0)


class TestBalanceSheet:
    def test_leverage_ratios_match_hand_computation(self):
        v = run_method(
            "balance_sheet_strength",
            mk([100.0], scalars={
                "total_assets": 2000.0, "stockholders_equity": 800.0,
                "long_term_debt": 500.0, "short_term_debt": 100.0,
                "current_assets": 600.0, "current_liabilities": 300.0,
                "cash_and_equivalents": 200.0, "short_term_investments": 50.0,
            }),
        ).values
        assert v["current_ratio"] == pytest.approx(2.0)
        assert v["total_debt"] == pytest.approx(600.0)
        assert v["debt_to_equity"] == pytest.approx(0.75)
        assert v["liquid_assets"] == pytest.approx(250.0)
        assert v["net_debt"] == pytest.approx(350.0)

    def test_absent_debt_tags_give_null_with_an_explanation(self):
        result = run_method(
            "balance_sheet_strength",
            mk([100.0], scalars={"total_assets": 1000.0, "stockholders_equity": 500.0}),
        )
        assert result.values["debt_to_equity"] is None
        assert any("absence of a tag" in w for w in result.warnings)

    def test_interest_coverage_uses_the_magnitude_of_interest(self):
        v = run_method(
            "balance_sheet_strength",
            mk([100.0], scalars={
                "total_assets": 1000.0, "operating_income": 300.0,
                "interest_expense": -50.0,
            }),
        ).values
        assert v["interest_coverage"] == pytest.approx(6.0)


class TestCashFlow:
    def test_free_cash_flow_subtracts_capex_magnitude(self):
        v = run_method(
            "cash_flow_quality",
            mk([100.0], scalars={
                "operating_cash_flow": 500.0, "capital_expenditure": 120.0,
                "revenue": 2000.0, "net_income": 450.0,
            }),
        ).values
        assert v["free_cash_flow"] == pytest.approx(380.0)
        assert v["fcf_margin"] == pytest.approx(0.19)
        assert v["ocf_to_net_income"] == pytest.approx(500.0 / 450.0)

    def test_capex_sign_does_not_change_the_result(self):
        positive = run_method(
            "cash_flow_quality",
            mk([100.0], scalars={"operating_cash_flow": 500.0, "capital_expenditure": 120.0}),
        ).values
        negative = run_method(
            "cash_flow_quality",
            mk([100.0], scalars={"operating_cash_flow": 500.0, "capital_expenditure": -120.0}),
        ).values
        assert positive["free_cash_flow"] == negative["free_cash_flow"]

    def test_a_poor_cash_conversion_is_flagged(self):
        result = run_method(
            "cash_flow_quality",
            mk([100.0], scalars={"operating_cash_flow": 100.0, "net_income": 500.0}),
        )
        assert any("earnings-quality" in w for w in result.warnings)


class TestValuation:
    def test_multiples_match_hand_computation(self):
        v = run_method(
            "valuation_ratios",
            mk([50.0], scalars={
                "shares_diluted": 100.0, "net_income": 250.0, "revenue": 2000.0,
                "stockholders_equity": 1000.0,
            }),
        ).values
        assert v["market_cap"] == pytest.approx(5000.0)
        assert v["price_to_earnings"] == pytest.approx(20.0)
        assert v["price_to_sales"] == pytest.approx(2.5)
        assert v["price_to_book"] == pytest.approx(5.0)

    def test_a_loss_making_company_has_no_pe_rather_than_a_negative_one(self):
        result = run_method(
            "valuation_ratios",
            mk([50.0], scalars={"shares_diluted": 100.0, "net_income": -250.0,
                                "revenue": 2000.0}),
        )
        assert result.values["price_to_earnings"] is None
        assert any("negative multiple is not interpretable" in w for w in result.warnings)
        # Price-to-sales is still meaningful and must still be reported.
        assert result.values["price_to_sales"] == pytest.approx(2.5)

    def test_no_share_count_is_refused_rather_than_guessed(self):
        from app.core.errors import InsufficientDataError

        with pytest.raises(InsufficientDataError, match="shares_diluted"):
            run_method("valuation_ratios", mk([50.0], scalars={"net_income": 100.0}))

    def test_a_basic_share_count_is_accepted_and_the_substitution_disclosed(self):
        result = run_method(
            "valuation_ratios",
            mk([50.0], scalars={"shares_basic": 100.0, "net_income": 250.0}),
        )
        assert result.values["price_to_earnings"] == pytest.approx(20.0)
        assert any("not dilution-adjusted" in w for w in result.warnings)

    def test_enterprise_value_adds_debt_and_subtracts_cash(self):
        v = run_method(
            "valuation_ratios",
            mk([50.0], scalars={
                "shares_diluted": 100.0, "operating_income": 500.0,
                "long_term_debt": 1000.0, "cash_and_equivalents": 400.0,
            }),
        ).values
        assert v["enterprise_value"] == pytest.approx(5600.0)  # 5000 + 1000 - 400
        assert v["ev_to_ebit"] == pytest.approx(5600.0 / 500.0)

    def test_the_share_count_basis_is_disclosed_in_limitations(self):
        result = run_method(
            "valuation_ratios", mk([50.0], scalars={"shares_diluted": 100.0})
        )
        assert any("weighted average" in lim for lim in result.limitations)


class TestGrowth:
    def test_cagr_matches_hand_computation(self):
        # 100 -> 200 over 4 annual periods: 2^(1/4) - 1
        v = run_method(
            "growth_trajectory",
            mk([100.0], series={"revenue_by_period": [100.0, 120.0, 150.0, 175.0, 200.0]}),
        ).values
        assert v["revenue_cagr"] == pytest.approx(2.0 ** (1 / 4) - 1)
        assert v["revenue_growth_latest"] == pytest.approx(200.0 / 175.0 - 1)

    def test_growth_from_a_loss_is_not_reported(self):
        v = run_method(
            "growth_trajectory",
            mk([100.0], series={"net_income_by_period": [-50.0, -10.0]}),
        ).values
        assert v["net_income_growth_latest"] is None, (
            "a percentage change from a loss is not meaningful"
        )

    def test_a_non_positive_first_period_suppresses_cagr_with_a_reason(self):
        result = run_method(
            "growth_trajectory",
            mk([100.0], series={"revenue_by_period": [0.0, 100.0, 200.0]}),
        )
        assert result.values["revenue_cagr"] is None
        assert any("zero or negative" in w for w in result.warnings)

    def test_consistency_counts_positive_periods(self):
        v = run_method(
            "growth_trajectory",
            mk([100.0], series={"revenue_by_period": [100.0, 110.0, 105.0, 120.0, 130.0]}),
        ).values
        # Three of four period-over-period changes are positive.
        assert v["revenue_growth_consistency"] == pytest.approx(0.75)
