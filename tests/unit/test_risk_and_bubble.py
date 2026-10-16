"""Risk and bubble-assessment tests.

The bubble tests focus on the constraints that keep the assessment honest:
missing data must not read as safety, a single signal must not produce a
verdict, opposing evidence must be collected, and no crash prediction may
appear anywhere in the output.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pytest

from app.analysis.registry import MethodInput, run_method
from app.core.errors import InsufficientDataError
from app.core.timeutil import UTC


def mk(closes, *, opens=None, highs=None, lows=None, volumes=None, scalars=None):
    n = len(closes)
    ts = tuple(dt.datetime(2021, 1, 4, tzinfo=UTC) + dt.timedelta(days=i) for i in range(n))
    return MethodInput(
        timestamps=ts,
        open=tuple(float(o) for o in opens) if opens else tuple(float(c) for c in closes),
        high=tuple(float(h) for h in highs) if highs else tuple(float(c) * 1.015 for c in closes),
        low=tuple(float(x) for x in lows) if lows else tuple(float(c) * 0.985 for c in closes),
        close=tuple(float(c) for c in closes),
        volume=tuple(volumes) if volumes is not None else tuple(2_000_000.0 for _ in closes),
        scalars=dict(scalars or {}),
        resolution="1d",
    )


def walk(n, *, seed=1, sigma=0.012, mu=0.0, start=100.0):
    rng = random.Random(seed)
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * math.exp(rng.gauss(mu, sigma)))
    return out


def parabolic(n, *, start=100.0, base=0.0008, accel_from=0.85, accel=0.012):
    """A series that advances steadily then accelerates sharply."""
    out = [start]
    for i in range(n - 1):
        rate = base if i < n * accel_from else accel
        out.append(out[-1] * math.exp(rate))
    return out


class TestRiskProfile:
    def test_core_measures_are_produced(self):
        v = run_method("risk_profile", mk(walk(400, seed=2))).values
        assert v["annualised_volatility"] > 0
        assert v["var_95"] < 0
        assert v["expected_shortfall_95"] <= v["var_95"]
        assert v["max_drawdown"] <= 0

    def test_expected_shortfall_is_never_better_than_var(self):
        for seed in (3, 4, 5):
            v = run_method("risk_profile", mk(walk(400, seed=seed))).values
            assert v["expected_shortfall_95"] <= v["var_95"]

    def test_annualisation_uses_the_square_root_of_252(self):
        v = run_method("risk_profile", mk(walk(500, seed=6))).values
        assert v["annualised_volatility"] == pytest.approx(
            v["period_volatility"] * math.sqrt(252)
        )

    def test_missing_volume_is_disclosed_not_assumed_adequate(self):
        result = run_method("risk_profile", mk(walk(300, seed=7), volumes=[None] * 300))
        assert result.values["liquidity"] == {}
        assert any("not assessed rather than being assumed adequate" in w
                   for w in result.warnings)

    def test_gap_risk_is_measured_separately_from_volatility(self):
        closes = walk(300, seed=8)
        opens = list(closes)
        opens[150] = closes[149] * 1.12  # a 12% overnight gap
        v = run_method("risk_profile", mk(closes, opens=opens)).values
        assert v["gap_risk"]["largest_gap_up"] > 0.1

    def test_the_disclaimer_is_attached(self):
        v = run_method("risk_profile", mk(walk(300, seed=9))).values
        assert v["disclaimer_code"] == "educational_research"
        assert "not investment advice" in v["disclaimer"]

    def test_high_volatility_is_flagged(self):
        result = run_method("risk_profile", mk(walk(400, seed=10, sigma=0.06)))
        assert any("very high" in w for w in result.warnings)


class TestTradeStructure:
    def test_stop_and_targets_are_volatility_scaled(self):
        closes = walk(300, seed=11)
        v = run_method(
            "trade_structure", mk(closes),
            {"direction": "long", "stop_atr_multiple": 2.0,
             "target_atr_multiples": [2.0, 4.0]},
        ).values
        s = v["structure"]
        atr = v["atr"]
        assert s["stop"] == pytest.approx(s["entry"] - 2.0 * atr)
        assert s["targets"][0] == pytest.approx(s["entry"] + 2.0 * atr)
        assert s["risk_reward_ratios"] == [pytest.approx(1.0), pytest.approx(2.0)]

    def test_a_short_structure_inverts_the_levels(self):
        v = run_method(
            "trade_structure", mk(walk(300, seed=12)), {"direction": "short"}
        ).values
        s = v["structure"]
        assert s["stop"] > s["entry"]
        assert all(t < s["entry"] for t in s["targets"])

    def test_gap_jump_frequency_is_reported_and_warned_about(self):
        """A stop is not a guaranteed exit, and the output must say so."""
        closes = walk(400, seed=13, sigma=0.01)
        opens = list(closes)
        # Inject many gaps larger than a 2-ATR stop.
        for i in range(50, 400, 10):
            opens[i] = closes[i - 1] * 1.15
        result = run_method(
            "trade_structure", mk(closes, opens=opens), {"stop_atr_multiple": 2.0}
        )
        assert result.values["stop_jump_probability"] > 0.01
        assert any("would have filled worse than its stated price" in w
                   for w in result.warnings)

    def test_position_size_follows_the_risk_budget(self):
        v = run_method(
            "trade_structure", mk(walk(300, seed=14)),
            {"account_value": 100_000.0, "account_risk_fraction": 0.01,
             "stop_atr_multiple": 2.0},
        ).values
        s = v["structure"]
        expected = 1_000.0 / (2.0 * v["atr"])
        assert s["position_size"] == pytest.approx(expected)

    def test_limitations_state_that_ratios_carry_no_probability(self):
        result = run_method("trade_structure", mk(walk(300, seed=15)))
        assert any("no probability of reaching either level" in lim or
                   "say nothing about the probability" in lim
                   for lim in result.limitations)

    def test_no_entry_is_presented_as_a_recommendation(self):
        result = run_method("trade_structure", mk(walk(300, seed=16)))
        assert any(
            "is a recommendation" in lim and "no entry" in lim
            for lim in result.limitations
        ), "the output must state that its levels are not recommendations"
        assert result.values["disclaimer_code"] == "educational_research"


class TestPositionSizing:
    def test_volatility_targeting_scales_inversely_with_volatility(self):
        quiet = run_method(
            "position_sizing", mk(walk(400, seed=17, sigma=0.005)),
            {"target_annual_volatility": 0.15},
        ).values
        loud = run_method(
            "position_sizing", mk(walk(400, seed=17, sigma=0.03)),
            {"target_annual_volatility": 0.15},
        ).values
        assert quiet["volatility_target_weight"] > loud["volatility_target_weight"]

    def test_weight_is_capped(self):
        v = run_method(
            "position_sizing", mk(walk(400, seed=18, sigma=0.002)),
            {"target_annual_volatility": 0.15, "max_weight": 1.0},
        ).values
        assert v["volatility_target_weight"] <= 1.0
        assert v["weight_capped"] is True

    def test_kelly_is_reported_with_a_half_kelly_alongside(self):
        v = run_method(
            "position_sizing", mk(walk(400, seed=19)), {"expected_edge": 0.08}
        ).values
        assert v["kelly_fraction"] is not None
        assert v["half_kelly_fraction"] == pytest.approx(v["kelly_fraction"] / 2)

    def test_leveraged_kelly_is_warned_about(self):
        result = run_method(
            "position_sizing", mk(walk(400, seed=20, sigma=0.004)),
            {"expected_edge": 0.50},
        )
        if result.values["kelly_fraction"] and result.values["kelly_fraction"] > 1.0:
            assert any("ruinous" in w for w in result.warnings)

    def test_limitations_warn_against_full_kelly(self):
        result = run_method("position_sizing", mk(walk(300, seed=21)))
        assert any("Half Kelly" in lim for lim in result.limitations)


class TestScenarioAnalysis:
    def test_scenarios_translate_shocks_to_price_levels(self):
        v = run_method(
            "scenario_analysis", mk(walk(400, seed=22)), {"shocks": [-0.2, 0.2]}
        ).values
        last = v["current_price"]
        down, up = v["scenarios"]
        assert down["resulting_price"] == pytest.approx(last * 0.8)
        assert up["resulting_price"] == pytest.approx(last * 1.2)

    def test_historical_frequency_is_reported_with_its_basis(self):
        v = run_method(
            "scenario_analysis", mk(walk(600, seed=23)), {"shocks": [-0.1]}
        ).values
        s = v["scenarios"][0]
        assert s["historical_frequency"] is not None
        assert "overlapping" in s["frequency_basis"]

    def test_a_larger_shock_is_rarer(self):
        v = run_method(
            "scenario_analysis", mk(walk(800, seed=24)), {"shocks": [-0.3, -0.1]}
        ).values
        big, small = v["scenarios"]
        assert big["historical_frequency"] <= small["historical_frequency"]

    def test_overlapping_windows_are_disclosed(self):
        result = run_method("scenario_analysis", mk(walk(600, seed=25)))
        assert any("not independent observations" in w for w in result.warnings)

    def test_each_scenario_exposes_its_assumptions(self):
        v = run_method("scenario_analysis", mk(walk(400, seed=26))).values
        for s in v["scenarios"]:
            assert s["assumptions"]


class TestBubbleAssessment:
    def test_a_quiet_random_walk_scores_low(self):
        v = run_method("bubble_assessment", mk(walk(600, seed=31, sigma=0.01))).values
        assert v["vulnerability_score"] < 40
        assert v["signals_triggered"] <= 2

    def test_a_parabolic_advance_triggers_multiple_signals(self):
        closes = parabolic(600)
        v = run_method("bubble_assessment", mk(closes)).values
        assert v["signals_triggered"] >= 3
        assert v["vulnerability_score"] > 30
        triggered = {s["key"] for s in v["signals"] if s["status"] == "triggered"}
        assert "parabolic_acceleration" in triggered

    def test_every_signal_reports_a_status(self):
        v = run_method("bubble_assessment", mk(walk(600, seed=32))).values
        assert len(v["signals"]) == v["signals_total"] == 10
        for s in v["signals"]:
            assert s["status"] in {"triggered", "not_triggered", "unavailable"}

    def test_unavailable_signals_are_named_with_a_reason_not_counted_as_safe(self):
        """Missing data must never read as low risk."""
        v = run_method("bubble_assessment", mk(walk(600, seed=33))).values
        unavailable = v["signals_unavailable"]
        assert unavailable, "options and concentration data are not ingested in this build"
        for entry in unavailable:
            assert entry["reason"], f"{entry['signal']} must explain why it is unavailable"
        assert v["signals_evaluated"] + len(unavailable) == v["signals_total"]

    def test_the_score_is_computed_over_evaluated_signals_only(self):
        """Dividing by all ten would let missing data look like safety."""
        closes = parabolic(600)
        with_options = run_method(
            "bubble_assessment",
            mk(closes, scalars={"implied_volatility": 0.9, "realized_volatility": 0.3,
                                "call_put_volume_ratio": 3.0}),
        ).values
        without = run_method("bubble_assessment", mk(closes)).values
        assert without["signals_evaluated"] < with_options["signals_evaluated"]
        assert without["vulnerability_score"] is not None

    def test_missing_half_the_signals_produces_an_explicit_warning(self):
        result = run_method("bubble_assessment", mk(walk(300, seed=34)))
        if len(result.values["signals_unavailable"]) >= 5:
            assert any("may indicate missing data rather than low risk" in w
                       for w in result.warnings)

    def test_a_single_triggered_signal_is_called_out_as_insufficient(self):
        for seed in range(40, 70):
            result = run_method("bubble_assessment", mk(walk(600, seed=seed)))
            if result.values["signals_triggered"] == 1:
                assert any("single indicator is not evidence" in w
                           for w in result.warnings)
                return
        pytest.skip("no seed in the scanned range produced exactly one triggered signal")

    def test_opposing_evidence_is_collected_alongside_supporting(self):
        """An assessment that lists only alarming facts is an argument, not analysis."""
        v = run_method("bubble_assessment", mk(parabolic(600))).values
        assert v["supporting_evidence"], "triggered signals must cite their evidence"
        assert v["opposing_evidence"], "untriggered signals must say why they did not fire"

    def test_no_crash_prediction_appears_anywhere(self):
        """No affirmative crash claim may appear in any signal's own output.

        The scan deliberately excludes the disclaimer and limitation fields,
        whose job is to say the words "crash date" in order to rule one out.
        Scanning them too would make the test fail on the very text that makes
        the output honest.
        """
        v = run_method("bubble_assessment", mk(parabolic(600))).values
        assert "does not predict a crash" in v["no_crash_prediction"]

        scanned = []
        for signal in v["signals"]:
            scanned.extend(
                [signal["description"], signal["evidence"], signal["counter_evidence"]]
            )
        scanned.extend(e["evidence"] for e in v["supporting_evidence"])
        scanned.extend(e["evidence"] for e in v["opposing_evidence"])
        scanned.append(v["vulnerability_band"] or "")
        scanned.append(v["scoring_formula"])

        blob = " ".join(scanned).lower()
        for forbidden in (
            "crash probability", "will crash", "crash date", "will collapse",
            "is a bubble", "certain to", "guaranteed",
        ):
            assert forbidden not in blob, (
                f"a signal's own output must not contain {forbidden!r}"
            )

    def test_the_score_is_named_vulnerability_not_crash_risk(self):
        v = run_method("bubble_assessment", mk(parabolic(600))).values
        assert "vulnerability_score" in v
        assert "crash_probability" not in v
        assert "crash_risk" not in v

    def test_the_scoring_formula_is_disclosed(self):
        v = run_method("bubble_assessment", mk(walk(600, seed=35))).values
        assert "divided by the total weight" in v["scoring_formula"]

    def test_limitations_state_it_cannot_distinguish_a_justified_rerating(self):
        result = run_method("bubble_assessment", mk(parabolic(600)))
        assert any("justified re-rating" in lim for lim in result.limitations)
        assert any("No crash date" in lim for lim in result.limitations)
        assert any("Survivorship bias" in lim for lim in result.limitations)

    def test_valuation_signal_uses_supplied_fundamentals(self):
        v = run_method(
            "bubble_assessment",
            mk(walk(600, seed=36), scalars={"price_to_earnings": 120.0,
                                            "price_to_sales": 30.0}),
        ).values
        valuation = next(s for s in v["signals"] if s["key"] == "valuation_extreme")
        assert valuation["status"] == "triggered"
        assert "120" in valuation["evidence"]

    def test_a_loss_making_company_makes_the_valuation_signal_unavailable(self):
        v = run_method("bubble_assessment", mk(walk(600, seed=37))).values
        valuation = next(s for s in v["signals"] if s["key"] == "valuation_extreme")
        assert valuation["status"] == "unavailable"
        assert "loss-making" in valuation["unavailable_reason"]

    def test_fundamental_divergence_is_detected_when_both_rates_are_supplied(self):
        v = run_method(
            "bubble_assessment",
            mk(walk(600, seed=38), scalars={"price_growth_1y": 1.5,
                                            "revenue_growth_1y": 0.1}),
        ).values
        divergence = next(s for s in v["signals"] if s["key"] == "fundamental_divergence")
        assert divergence["status"] == "triggered"
        assert "gap of" in divergence["evidence"]

    def test_strengthening_and_weakening_conditions_are_both_given(self):
        v = run_method("bubble_assessment", mk(parabolic(600))).values
        assert len(v["strengthening_conditions"]) >= 3
        assert len(v["weakening_conditions"]) >= 3

    def test_short_history_is_refused(self):
        with pytest.raises(InsufficientDataError, match="at least 250"):
            run_method("bubble_assessment", mk(walk(100, seed=39)))

    def test_bands_are_labels_not_false_precision(self):
        v = run_method("bubble_assessment", mk(parabolic(600))).values
        assert v["vulnerability_band"] in {"limited", "moderate", "elevated", "high"}
