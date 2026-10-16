"""Report generation and LLM-guard tests.

The central assertions: every finding carries an epistemic label, a model
estimate is never phrased as an observed fact, missing data appears as an
explicit "unanswered" finding rather than a gap, and a narrative containing an
invented number is discarded rather than published.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.core.timeutil import UTC
from app.db.models.enums import ClaimKind, EvidenceStance
from app.reporting.evidence import EvidenceItem, Finding, Report
from app.reporting.generator import generate_report
from app.reporting.llm.base import (
    NarrativeRequest,
    NullNarrativeProvider,
    extract_numbers,
    permitted_numbers,
    validate_narrative,
)
from app.services.analysis_service import AnalysisBundle, MethodOutcome


def outcome(key, values, *, version="1.0.0", n=250, limitations=None, warnings=None):
    return MethodOutcome(
        key=key,
        version=version,
        display_name=key.replace("_", " ").title(),
        ok=True,
        values=values,
        observations_used=n,
        input_fingerprint=f"fp_{key}",
        limitations=limitations or [f"{key} has limits"],
        warnings=warnings or [],
    )


def bundle_with(outcomes, **overrides) -> AnalysisBundle:
    defaults = {
        "instrument_id": "11111111-1111-1111-1111-111111111111",
        "symbol": "TEST",
        "name": "Test Instrument",
        "as_of": dt.datetime(2024, 6, 28, tzinfo=UTC),
        "purpose": "historical_reconstruction",
        "resolution": "1d",
        "adjustment_basis": "total_return",
        "period_start": dt.datetime(2023, 6, 30, tzinfo=UTC),
        "period_end": dt.datetime(2024, 6, 28, tzinfo=UTC),
        "observations": 250,
        "currency": "USD",
        "outcomes": outcomes,
        "data_sources": ["eodhd"],
    }
    defaults.update(overrides)
    return AnalysisBundle(**defaults)


FULL_OUTCOMES = {
    "returns": outcome("returns", {
        "total_return": 0.1834, "total_log_return": 0.1683,
        "annualised_return": 0.1834, "first_close": 100.0, "last_close": 118.34,
        "periods": 249,
    }),
    "realized_volatility": outcome("realized_volatility", {
        "annualised_volatility": 0.2412, "daily_volatility": 0.0152,
        "observations": 249, "periods_per_year_assumed": 252.0,
    }, limitations=["Realised volatility is backward-looking; it is not a forecast."]),
    "drawdown": outcome("drawdown", {
        "max_drawdown": -0.1125, "current_drawdown": -0.0210,
        "peak_date": "2024-03-15T13:30:00+00:00",
        "trough_date": "2024-04-19T13:30:00+00:00",
        "peak_close": 121.0, "trough_close": 107.4,
    }),
    "trend_strength": outcome("trend_strength", {
        "slope_per_period": 0.00067, "annualised_slope": 0.1826,
        "r_squared": 0.7421, "up_period_fraction": 0.546,
    }),
    "regime_classification": outcome("regime_classification", {
        "label": "trending_up", "confidence": 0.68,
        "label_probabilities": {"trending_up": 0.68, "low_volatility": 0.32},
        "diagnostics": {
            "trend_r_squared": 0.7421, "volatility_ratio": 0.88,
            "max_drawdown": -0.1125,
        },
    }, limitations=["Regimes are not observable; this is one rule's opinion."]),
    "relative_strength": outcome("relative_strength", {
        "excess_return": 0.0412, "asset_return": 0.1834,
        "benchmark_return": 0.1422, "beta": 1.14, "alpha_annualised": 0.0298,
        "correlation": 0.78, "r_squared": 0.6084,
    }),
}


class TestFindingValidation:
    def test_an_unlabelled_finding_is_rejected(self):
        with pytest.raises(ValueError, match="claim_kind must be one of"):
            Finding(section="x", conclusion="y", claim_kind="made_up_label")

    def test_every_valid_claim_kind_is_accepted(self):
        for kind in ClaimKind:
            Finding(section="x", conclusion="y", claim_kind=kind)

    def test_supporting_and_contradicting_evidence_are_separable(self):
        finding = Finding(
            section="x", conclusion="y", claim_kind=ClaimKind.OBSERVED_FACT,
            evidence=[
                EvidenceItem(stance=EvidenceStance.SUPPORTING, description="a",
                             source_table="price_bars"),
                EvidenceItem(stance=EvidenceStance.CONTRADICTING, description="b",
                             source_table="price_bars"),
            ],
        )
        assert len(finding.supporting) == 1
        assert len(finding.contradicting) == 1


class TestReportGeneration:
    def test_a_report_is_produced_with_labelled_findings(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        assert report.findings
        for finding in report.findings:
            assert finding.claim_kind in set(ClaimKind), (
                "every finding must carry an epistemic label"
            )

    def test_a_computed_return_is_labelled_a_calculation_not_a_fact(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        perf = [f for f in report.findings if f.section == "performance"]
        assert perf
        assert perf[0].claim_kind == ClaimKind.DETERMINISTIC_CALCULATION

    def test_a_regime_label_is_a_model_estimate_not_a_fact(self):
        """The label that matters most: a regime is never presented as observed."""
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        regime = [
            f for f in report.findings
            if f.section == "regime" and "classified as" in f.conclusion
        ]
        assert regime
        assert regime[0].claim_kind == ClaimKind.MODEL_ESTIMATE
        assert "not" in regime[0].conclusion.lower()
        assert "estimate" in regime[0].conclusion.lower()

    def test_beta_is_a_statistical_relationship_not_a_causal_claim(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        bench = [f for f in report.findings if f.section == "benchmark"]
        assert bench
        assert bench[0].claim_kind == ClaimKind.STATISTICAL_RELATIONSHIP

    def test_every_finding_with_a_number_names_the_method_that_produced_it(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        for finding in report.findings:
            for calc in finding.calculations:
                assert calc.method_key
                assert calc.method_version
                assert calc.input_fingerprint, (
                    "a figure must be traceable to the exact inputs it came from"
                )

    def test_findings_carry_their_methods_limitations(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        vol = [f for f in report.findings if f.section == "volatility"]
        assert vol
        assert any("not a forecast" in lim for lim in vol[0].limitations)

    def test_every_finding_has_evidence_or_is_an_unknown(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        for finding in report.findings:
            if finding.claim_kind == ClaimKind.UNKNOWN:
                continue
            assert finding.evidence, (
                f"finding in {finding.section} asserts something with no evidence"
            )

    def test_volatility_is_rendered_unsigned(self):
        """A volatility of "+35.6%" implies it could be negative. It cannot.

        Signed rendering is correct for a return and wrong for a standard
        deviation, so the two use different formatters.
        """
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        vol = next(f for f in report.findings if f.section == "volatility")
        assert "+" not in vol.conclusion, vol.conclusion
        assert "24.1%" in vol.conclusion
        assert "annualised volatility was 24.1%" in report.summary

    def test_returns_are_still_rendered_signed(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        perf = next(f for f in report.findings if f.section == "performance")
        assert "+18.3%" in perf.conclusion, perf.conclusion

    def test_the_summary_counts_the_claim_kinds_it_contains(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        assert "This report contains" in report.summary
        counts = report.claim_kind_counts()
        assert sum(counts.values()) == len(report.findings)

    def test_non_normal_returns_appear_as_contradicting_evidence(self):
        """Quoting a volatility figure must come with its own counterweight."""
        outcomes = dict(FULL_OUTCOMES)
        outcomes["return_distribution"] = outcome("return_distribution", {
            "excess_kurtosis": 4.21, "skewness": -0.62,
            "jarque_bera_p": 0.0001, "normality_rejected_at_5pct": True,
            "observations": 249,
        })
        report = generate_report(bundle_with(outcomes))
        vol = next(f for f in report.findings if f.section == "volatility")
        assert vol.contradicting, (
            "a non-normal distribution contradicts a single volatility number"
        )
        assert "kurtosis" in vol.contradicting[0].description


class TestMissingDataIsReported:
    def test_an_unavailable_method_becomes_an_explicit_unanswered_finding(self):
        outcomes = dict(FULL_OUTCOMES)
        outcomes["seasonality"] = MethodOutcome.unavailable(
            "seasonality", "Calendar seasonality", "1.0.0",
            "needs at least 250 observations but only 90 are available",
            "insufficient_data",
        )
        report = generate_report(bundle_with(outcomes))
        unanswered = [f for f in report.findings if f.section == "limitations"]
        assert unanswered
        assert unanswered[0].claim_kind == ClaimKind.UNKNOWN
        assert any("250 observations" in lim for lim in unanswered[0].limitations)

    def test_a_missing_benchmark_is_stated_not_omitted(self):
        outcomes = dict(FULL_OUTCOMES)
        outcomes["relative_strength"] = MethodOutcome.unavailable(
            "relative_strength", "Relative strength versus benchmark", "1.0.0",
            "the benchmark SPY is not ingested", "insufficient_data",
        )
        report = generate_report(bundle_with(outcomes))
        bench = next(f for f in report.findings if f.section == "benchmark")
        assert bench.claim_kind == ClaimKind.UNKNOWN
        assert "not ingested" in bench.conclusion

    def test_data_gaps_become_a_finding_of_their_own(self):
        report = generate_report(
            bundle_with(
                dict(FULL_OUTCOMES),
                gaps=[{
                    "gap_start": "2024-02-05T14:30:00+00:00",
                    "gap_end": "2024-02-07T21:00:00+00:00",
                    "expected_sessions": 3,
                    "reason": "3 sessions expected by the XNYS calendar have no bar",
                    "filled": False,
                }],
            )
        )
        quality = next(f for f in report.findings if f.section == "data_quality")
        assert "3 expected session" in quality.conclusion
        assert "Nothing was interpolated" in quality.conclusion
        assert quality.evidence

    def test_a_clean_window_says_so_explicitly(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        quality = next(f for f in report.findings if f.section == "data_quality")
        assert "No data gaps" in quality.conclusion
        assert quality.claim_kind == ClaimKind.OBSERVED_FACT


class TestReproducibility:
    def test_the_same_bundle_produces_the_same_reproduction_key(self):
        a = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        b = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        assert a.reproduction_key == b.reproduction_key

    def test_a_different_cutoff_changes_the_reproduction_key(self):
        a = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        b = generate_report(
            bundle_with(dict(FULL_OUTCOMES), as_of=dt.datetime(2024, 6, 29, tzinfo=UTC))
        )
        assert a.reproduction_key != b.reproduction_key

    def test_different_input_data_changes_the_reproduction_key(self):
        outcomes = dict(FULL_OUTCOMES)
        changed = outcome("returns", dict(FULL_OUTCOMES["returns"].values))
        changed.input_fingerprint = "a_different_fingerprint"
        outcomes["returns"] = changed
        a = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        b = generate_report(bundle_with(outcomes))
        assert a.reproduction_key != b.reproduction_key

    def test_method_versions_are_recorded(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        assert report.method_versions["returns"] == "1.0.0"
        assert len(report.method_versions) == len(FULL_OUTCOMES)

    def test_the_cutoff_and_window_are_recorded_in_inputs(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        assert report.inputs["as_of"] == "2024-06-28T00:00:00+00:00"
        assert report.inputs["adjustment_basis"] == "total_return"


class TestNarrativeGuard:
    @staticmethod
    def request_from_report(report: Report) -> NarrativeRequest:
        return NarrativeRequest(
            title=report.title,
            summary=report.summary,
            findings=[f.as_dict() for f in report.findings],
            limitations=report.limitations,
        )

    def test_the_null_provider_generates_nothing_and_says_why(self):
        result = NullNarrativeProvider().generate(
            NarrativeRequest(title="t", summary="s", findings=[], limitations=[])
        )
        assert result.text is None
        assert "no LLM provider is configured" in result.rejected_reason

    def test_a_narrative_reusing_only_supplied_figures_is_accepted(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        request = self.request_from_report(report)
        text = (
            "The instrument returned 18.3% over the period, with annualised "
            "volatility of 24.1%. Its deepest decline was 11.3%."
        )
        result = validate_narrative(text, request)
        assert result.ok, result.rejected_reason

    def test_a_narrative_with_an_invented_figure_is_discarded(self):
        """A narrative is optional; a wrong number is not acceptable."""
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        request = self.request_from_report(report)
        text = (
            "The instrument returned 18.3% over the period and analysts expect "
            "a further 42.7% gain next year, with revenue of 9876 million."
        )
        result = validate_narrative(text, request)
        assert not result.ok
        assert "discarded rather than published" in result.rejected_reason
        assert "42.7" in result.fabricated_numbers

    def test_an_invented_price_target_is_caught(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        request = self.request_from_report(report)
        result = validate_narrative(
            "The shares should reach 215.50 within twelve months.", request
        )
        assert not result.ok
        assert "215.5" in result.fabricated_numbers or "215.50" in result.fabricated_numbers

    def test_prose_with_no_numbers_is_accepted(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        request = self.request_from_report(report)
        result = validate_narrative(
            "The instrument rose over the period while remaining less volatile "
            "than its benchmark, and the trend estimate is moderately strong.",
            request,
        )
        assert result.ok

    def test_trivial_numbers_do_not_trigger_rejection(self):
        report = generate_report(bundle_with(dict(FULL_OUTCOMES)))
        request = self.request_from_report(report)
        result = validate_narrative(
            "There are 2 main observations and 1 caveat worth noting.", request
        )
        assert result.ok

    def test_number_extraction_normalises_signs_and_separators(self):
        found = extract_numbers("up +18.34% from 1,234.50 to -99")
        assert "18.34" in found
        assert "1234.50" in found or "1234.5" in found
        assert "99" in found

    def test_permitted_numbers_include_percentage_renderings_of_fractions(self):
        request = NarrativeRequest(
            title="t", summary="s",
            findings=[{
                "conclusion": "c", "calculations": [
                    {"label": "Total return", "value": 0.1834, "unit": "fraction"}
                ],
                "evidence": [], "limitations": [],
            }],
            limitations=[],
        )
        allowed = permitted_numbers(request)
        assert "18.3" in allowed, "a fraction must permit its percentage rendering"
        assert "18" in allowed
