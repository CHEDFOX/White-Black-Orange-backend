"""Report generation from verified calculations.

Every sentence a report contains is written here, in Python, from numbers the
analysis engines produced. The LLM layer is optional and additive: it may
paraphrase these findings into prose, but it is never the source of a figure,
a citation or a confidence level. With ``LLM_PROVIDER=null`` the platform is
fully functional and the reports contain no generated text at all.

Phrasing rules applied throughout, because wording is where honesty is usually
lost:

* A price movement is described, never explained, unless the explanation has
  evidence. "The stock fell 18% over the period" is an observed fact; "the
  stock fell because of the rate decision" is a causal hypothesis and is
  labelled as one.
* Temporal adjacency is stated as adjacency. An event before a move earns
  "this preceded the move", not "this caused it".
* Where several explanations fit, all are listed with their evidence, and the
  finding says the question is unresolved.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.core.timeutil import now_utc
from app.db.models.enums import ClaimKind, EvidenceStance
from app.reporting.evidence import (
    GENERATOR_VERSION,
    Calculation,
    EvidenceItem,
    Finding,
    Report,
    calculation_from_outcome,
    method_evidence,
)
from app.services.analysis_service import AnalysisBundle

log = get_logger(__name__)


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:+.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _confidence_from_observations(n: int) -> str:
    """Qualitative confidence from sample size alone.

    Deliberately coarse. A numeric confidence would imply a calibrated
    probability the platform has not established for these descriptive
    statistics.
    """
    if n >= 500:
        return "based on a long sample"
    if n >= 250:
        return "based on about a year of data"
    if n >= 60:
        return "based on a short sample"
    return "based on very little data"


# --- section builders --------------------------------------------------------


def _performance_findings(bundle: AnalysisBundle) -> list[Finding]:
    out: list[Finding] = []
    returns = bundle.outcomes.get("returns")
    if returns is None or not returns.ok:
        return out

    total = returns.values.get("total_return")
    annualised = returns.values.get("annualised_return")
    first_close = returns.values.get("first_close")
    last_close = returns.values.get("last_close")

    calcs = [
        c
        for c in (
            calculation_from_outcome(returns, "Total return", "total_return", "fraction"),
            calculation_from_outcome(
                returns, "Annualised return", "annualised_return", "fraction_per_year"
            ),
        )
        if c is not None
    ]

    period = ""
    if bundle.period_start and bundle.period_end:
        period = (
            f" between {bundle.period_start.date()} and {bundle.period_end.date()}"
        )

    out.append(
        Finding(
            section="performance",
            # An observed fact: this is arithmetic on stored prices.
            conclusion=(
                f"{bundle.symbol} returned {_pct(total)}{period}, moving from "
                f"{_num(first_close)} to {_num(last_close)} {bundle.currency}"
                + (f" (equivalent to {_pct(annualised)} annualised)" if annualised else "")
                + "."
            ),
            claim_kind=ClaimKind.DETERMINISTIC_CALCULATION,
            confidence_label="computed directly from stored prices",
            methodology=(
                f"Computed from the first and last close of the window on the "
                f"{bundle.adjustment_basis} basis, using {returns.observations_used} bars."
            ),
            uncertainty=(
                "None in the arithmetic. The figure depends entirely on the window "
                "boundaries chosen; a window shifted by a week can give a very "
                "different number."
            ),
            limitations=returns.limitations,
            calculations=calcs,
            evidence=[
                method_evidence(
                    returns,
                    f"Return computed from {returns.observations_used} stored bars "
                    f"from {', '.join(bundle.data_sources) or 'the configured provider'}",
                    value_field="total_return",
                )
            ],
        )
    )

    drawdown = bundle.outcomes.get("drawdown")
    if drawdown and drawdown.ok:
        worst = drawdown.values.get("max_drawdown")
        peak = drawdown.values.get("peak_date")
        trough = drawdown.values.get("trough_date")
        out.append(
            Finding(
                section="performance",
                conclusion=(
                    f"The deepest peak-to-trough decline in this window was "
                    f"{_pct(worst)}"
                    + (
                        f", from {str(peak)[:10]} to {str(trough)[:10]}"
                        if peak and trough
                        else ""
                    )
                    + "."
                ),
                claim_kind=ClaimKind.DETERMINISTIC_CALCULATION,
                confidence_label="computed directly from stored prices",
                methodology=(
                    "Maximum decline from a running peak, measured on closing prices."
                ),
                uncertainty=(
                    "Measured close-to-close, so the intraday experience was worse "
                    "than this figure. It is a lower bound on the true drawdown."
                ),
                limitations=drawdown.limitations,
                calculations=[
                    c
                    for c in (
                        calculation_from_outcome(
                            drawdown, "Maximum drawdown", "max_drawdown", "fraction"
                        ),
                        calculation_from_outcome(
                            drawdown, "Current drawdown", "current_drawdown", "fraction"
                        ),
                    )
                    if c is not None
                ],
                evidence=[method_evidence(drawdown, "Drawdown over the stored series")],
            )
        )
    return out


def _volatility_findings(bundle: AnalysisBundle) -> list[Finding]:
    out: list[Finding] = []
    vol = bundle.outcomes.get("realized_volatility")
    if vol is None or not vol.ok:
        return out

    annualised = vol.values.get("annualised_volatility")
    distribution = bundle.outcomes.get("return_distribution")

    evidence: list[EvidenceItem] = [
        method_evidence(
            vol,
            f"Realised volatility from {vol.observations_used} returns",
            value_field="annualised_volatility",
        )
    ]
    contradicting: list[EvidenceItem] = []
    uncertainty = (
        "Realised volatility is backward-looking and is not a forecast. "
        "Annualisation assumes returns are serially uncorrelated."
    )

    if distribution and distribution.ok:
        kurtosis = distribution.values.get("excess_kurtosis")
        rejected = distribution.values.get("normality_rejected_at_5pct")
        if rejected:
            # This is the honest counterweight to quoting a volatility number:
            # the distribution it summarises is not the one it implies.
            contradicting.append(
                method_evidence(
                    distribution,
                    (
                        f"A Jarque-Bera test rejects normality (excess kurtosis "
                        f"{_num(kurtosis)}), so a single volatility figure "
                        "understates the chance of a large move"
                    ),
                    stance=EvidenceStance.CONTRADICTING,
                    value_field="excess_kurtosis",
                )
            )
            uncertainty += (
                " The return distribution in this window is not normal, so "
                "intervals derived from this volatility alone understate tail risk."
            )

    out.append(
        Finding(
            section="volatility",
            conclusion=(
                f"Annualised realised volatility over the window was "
                f"{_pct(annualised, 1)}, {_confidence_from_observations(vol.observations_used)}."
            ),
            claim_kind=ClaimKind.DETERMINISTIC_CALCULATION,
            confidence_label=_confidence_from_observations(vol.observations_used),
            methodology=(
                "Sample standard deviation of log returns, annualised by the square "
                f"root of {vol.values.get('periods_per_year_assumed')} periods per year."
            ),
            uncertainty=uncertainty,
            limitations=vol.limitations,
            calculations=[
                c
                for c in (
                    calculation_from_outcome(
                        vol, "Annualised volatility", "annualised_volatility",
                        "fraction_per_year",
                    ),
                    calculation_from_outcome(
                        vol, "Per-period volatility", "daily_volatility", "fraction",
                    ),
                )
                if c is not None
            ],
            evidence=evidence + contradicting,
        )
    )
    return out


def _regime_findings(bundle: AnalysisBundle) -> list[Finding]:
    out: list[Finding] = []
    regime = bundle.outcomes.get("regime_classification")
    trend = bundle.outcomes.get("trend_strength")
    if regime is None or not regime.ok:
        return out

    label = regime.values.get("label")
    confidence = regime.values.get("confidence")
    diagnostics = regime.values.get("diagnostics", {})
    probabilities = regime.values.get("label_probabilities", {})

    runner_up = None
    if len(probabilities) > 1:
        ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
        runner_up = ranked[1]

    evidence = [
        method_evidence(
            regime,
            (
                f"Trend R-squared {_num(diagnostics.get('trend_r_squared'))}, "
                f"volatility ratio {_num(diagnostics.get('volatility_ratio'))}, "
                f"maximum drawdown {_pct(diagnostics.get('max_drawdown'))}"
            ),
            value_field="label",
        )
    ]
    if runner_up:
        evidence.append(
            method_evidence(
                regime,
                (
                    f"The alternative classification {runner_up[0]!r} scored "
                    f"{runner_up[1]:.2f} against {confidence:.2f} for the reported "
                    "label, so the boundary is not sharp"
                ),
                stance=EvidenceStance.CONTRADICTING,
            )
        )

    out.append(
        Finding(
            section="regime",
            conclusion=(
                f"The window is classified as {str(label).replace('_', ' ')} with "
                f"confidence {_num(confidence)}. This is one rule's estimate, not "
                "an observed property of the market."
            ),
            # Explicitly a model estimate, never a fact.
            claim_kind=ClaimKind.MODEL_ESTIMATE,
            confidence_label=f"rule confidence {_num(confidence)}",
            methodology=(
                "Scored from the R-squared and sign of an OLS trend on log price, "
                "recent volatility against its longer-run baseline, and drawdown "
                "depth. Thresholds are stated in the method's parameters."
            ),
            uncertainty=(
                "Regimes are not directly observable. A different rule or a "
                "different threshold set would classify some windows differently, "
                "especially near the boundaries."
            ),
            limitations=regime.limitations,
            calculations=[
                Calculation(
                    label="Regime label",
                    value=str(label),
                    unit="label",
                    method_key=regime.key,
                    method_version=regime.version,
                    observations_used=regime.observations_used,
                    input_fingerprint=regime.input_fingerprint,
                    params=regime.params,
                )
            ],
            evidence=evidence,
        )
    )

    if trend and trend.ok:
        r2 = trend.values.get("r_squared")
        slope = trend.values.get("annualised_slope")
        out.append(
            Finding(
                section="regime",
                conclusion=(
                    f"A least-squares trend fitted to log price implies "
                    f"{_pct(slope)} per year with an R-squared of {_num(r2)}."
                ),
                claim_kind=ClaimKind.STATISTICAL_RELATIONSHIP,
                confidence_label=f"R-squared {_num(r2)}",
                methodology="OLS regression of log close on a time index.",
                uncertainty=(
                    "A fitted trend describes this window only and does not extend "
                    "beyond it. A high R-squared over a short window is common even "
                    "in a random walk."
                ),
                limitations=trend.limitations,
                calculations=[
                    c
                    for c in (
                        calculation_from_outcome(
                            trend, "Annualised trend slope", "annualised_slope",
                            "fraction_per_year",
                        ),
                        calculation_from_outcome(
                            trend, "Trend R-squared", "r_squared", "fraction"
                        ),
                    )
                    if c is not None
                ],
                evidence=[method_evidence(trend, "OLS fit over the stored series")],
            )
        )
    return out


def _benchmark_findings(bundle: AnalysisBundle) -> list[Finding]:
    out: list[Finding] = []
    rel = bundle.outcomes.get("relative_strength")
    if rel is None:
        return out
    if not rel.ok:
        out.append(
            Finding(
                section="benchmark",
                conclusion=(
                    "No benchmark comparison is available for this window: "
                    f"{rel.unavailable_reason}"
                ),
                claim_kind=ClaimKind.UNKNOWN,
                uncertainty=(
                    "Without a benchmark it is not possible to separate "
                    "instrument-specific movement from market movement."
                ),
            )
        )
        return out

    excess = rel.values.get("excess_return")
    beta = rel.values.get("beta")
    correlation = rel.values.get("correlation")
    r_squared = rel.values.get("r_squared")

    evidence = [
        method_evidence(
            rel,
            (
                f"Asset return {_pct(rel.values.get('asset_return'))} against "
                f"benchmark {bundle.benchmark_symbol} at "
                f"{_pct(rel.values.get('benchmark_return'))}, over "
                f"{rel.observations_used} aligned bars"
            ),
            value_field="excess_return",
        )
    ]
    if r_squared is not None and r_squared < 0.3:
        evidence.append(
            method_evidence(
                rel,
                (
                    f"The benchmark explains only {_pct(r_squared, 0)} of this "
                    "instrument's return variance, so the beta estimate is weakly "
                    "determined and the excess return is mostly idiosyncratic"
                ),
                stance=EvidenceStance.CONTRADICTING,
            )
        )

    out.append(
        Finding(
            section="benchmark",
            conclusion=(
                f"Against {bundle.benchmark_symbol}, {bundle.symbol} returned "
                f"{_pct(excess)} in excess, with a beta of {_num(beta)} and a "
                f"return correlation of {_num(correlation)}."
            ),
            claim_kind=ClaimKind.STATISTICAL_RELATIONSHIP,
            confidence_label=f"R-squared {_num(r_squared)}",
            methodology=(
                "Beta from the covariance of log returns with the benchmark over "
                "the bars where both series have data, aligned by exact timestamp."
            ),
            uncertainty=(
                "Beta is an in-sample coefficient for this window, not a stable "
                "property. It moves substantially across periods and regimes."
            ),
            limitations=rel.limitations,
            calculations=[
                c
                for c in (
                    calculation_from_outcome(rel, "Excess return", "excess_return", "fraction"),
                    calculation_from_outcome(rel, "Beta", "beta", "ratio"),
                    calculation_from_outcome(rel, "Correlation", "correlation", "correlation"),
                )
                if c is not None
            ],
            evidence=evidence,
        )
    )
    return out


def _data_quality_findings(bundle: AnalysisBundle) -> list[Finding]:
    """State the holes in the data as a finding, not a footnote."""
    out: list[Finding] = []
    if not bundle.gaps and not bundle.quality_issues:
        out.append(
            Finding(
                section="data_quality",
                conclusion=(
                    f"No data gaps or validation failures are recorded for this "
                    f"window: {bundle.observations} bars were available with no "
                    "missing expected sessions."
                ),
                claim_kind=ClaimKind.OBSERVED_FACT,
                methodology=(
                    "Checked against the recorded data-gap and data-quality tables "
                    "for the window."
                ),
                evidence=[
                    EvidenceItem(
                        stance=EvidenceStance.SUPPORTING,
                        description=(
                            f"{bundle.observations} bars loaded from "
                            f"{', '.join(bundle.data_sources) or 'the provider'} with "
                            "no recorded gaps"
                        ),
                        source_table="price_bars",
                        detail={"observations": bundle.observations},
                    )
                ],
            )
        )
        return out

    gap_sessions = sum(g["expected_sessions"] for g in bundle.gaps)
    errors = [i for i in bundle.quality_issues if i["severity"] == "error"]
    warnings = [i for i in bundle.quality_issues if i["severity"] == "warning"]

    parts: list[str] = []
    if gap_sessions:
        parts.append(f"{gap_sessions} expected session(s) have no data")
    if errors:
        parts.append(f"{len(errors)} bar(s) failed structural validation")
    if warnings:
        parts.append(f"{len(warnings)} data-quality warning(s) are recorded")


    out.append(
        Finding(
            section="data_quality",
            conclusion=(
                "This window has known data-quality issues: "
                + "; ".join(parts)
                + ". Nothing was interpolated to fill them, so the statistics above "
                "describe the bars actually present."
            ),
            claim_kind=ClaimKind.OBSERVED_FACT,
            methodology="Recorded by the ingestion validator at load time.",
            uncertainty=(
                "Statistics computed over a window with gaps measure the sessions "
                "present. Returns spanning a gap cover more elapsed time than their "
                "bar count implies."
            ),
            limitations=[
                "A provider that silently omits a session cannot be distinguished "
                "from a genuine trading halt without a second source.",
            ],
            evidence=[
                EvidenceItem(
                    stance=EvidenceStance.SUPPORTING,
                    description=(
                        f"Gap of {g['expected_sessions']} session(s) from "
                        f"{g['gap_start'][:10]} to {g['gap_end'][:10]}: {g['reason']}"
                    ),
                    source_table="data_gaps",
                    detail=g,
                )
                for g in bundle.gaps[:10]
            ]
            + [
                EvidenceItem(
                    stance=EvidenceStance.SUPPORTING,
                    description=f"{i['kind']} at {str(i['at'])[:10]}: {i['detail']}",
                    source_table="data_quality_issues",
                    detail=i,
                )
                for i in (errors + warnings)[:10]
            ],
        )
    )
    return out


def _unavailable_findings(bundle: AnalysisBundle) -> list[Finding]:
    """Record what could not be computed, and why."""
    if not bundle.unavailable:
        return []
    return [
        Finding(
            section="limitations",
            conclusion=(
                f"{len(bundle.unavailable)} analysis method(s) could not be run on "
                "this window, so the corresponding questions are unanswered rather "
                "than answered with a default."
            ),
            claim_kind=ClaimKind.UNKNOWN,
            methodology="Each method declares a minimum data requirement and refuses below it.",
            limitations=[
                f"{o.display_name}: {o.unavailable_reason}"
                for o in bundle.unavailable.values()
            ],
            evidence=[
                EvidenceItem(
                    stance=EvidenceStance.CONTEXTUAL,
                    description=f"{o.display_name} ({o.key}@{o.version}): {o.unavailable_reason}",
                    source_table="method_definitions",
                    detail={"code": o.unavailable_code},
                )
                for o in bundle.unavailable.values()
            ],
        )
    ]


def _fundamental_findings(bundle: AnalysisBundle) -> list[Finding]:
    out: list[Finding] = []
    valuation = bundle.outcomes.get("valuation_ratios")
    profitability = bundle.outcomes.get("profitability")

    if profitability and profitability.ok:
        net_margin = profitability.values.get("net_margin")
        gross_margin = profitability.values.get("gross_margin")
        roic = profitability.values.get("return_on_invested_capital")
        if any(v is not None for v in (net_margin, gross_margin, roic)):
            out.append(
                Finding(
                    section="fundamentals",
                    conclusion=(
                        "On the most recent figures published before the analysis "
                        f"timestamp, gross margin was {_pct(gross_margin)}, net "
                        f"margin {_pct(net_margin)}"
                        + (f" and return on invested capital {_pct(roic)}" if roic else "")
                        + "."
                    ),
                    claim_kind=ClaimKind.OBSERVED_FACT,
                    confidence_label="from filed financial statements",
                    methodology=(
                        "Ratios computed from individual XBRL facts, each selected "
                        "as the newest value filed on or before the analysis "
                        "timestamp, so a later restatement is not used."
                    ),
                    uncertainty=(
                        "Accounting definitions differ between issuers, so margins "
                        "are not strictly comparable across companies."
                    ),
                    limitations=profitability.limitations,
                    calculations=[
                        c
                        for c in (
                            calculation_from_outcome(
                                profitability, "Gross margin", "gross_margin", "fraction"
                            ),
                            calculation_from_outcome(
                                profitability, "Net margin", "net_margin", "fraction"
                            ),
                            calculation_from_outcome(
                                profitability, "Return on invested capital",
                                "return_on_invested_capital", "fraction",
                            ),
                        )
                        if c is not None
                    ],
                    evidence=[
                        EvidenceItem(
                            stance=EvidenceStance.SUPPORTING,
                            description=(
                                "Computed from financial facts filed on or before "
                                f"{bundle.as_of.date()}"
                            ),
                            source_table="financial_facts",
                            source_row_id=profitability.input_fingerprint,
                            detail={"revenue": profitability.values.get("revenue")},
                        )
                    ],
                )
            )

    if valuation and valuation.ok:
        pe = valuation.values.get("price_to_earnings")
        ps = valuation.values.get("price_to_sales")
        if pe is not None or ps is not None:
            out.append(
                Finding(
                    section="fundamentals",
                    conclusion=(
                        "At the analysis timestamp the shares traded at "
                        + (f"{_num(pe)}x trailing earnings" if pe else "no meaningful P/E")
                        + (f" and {_num(ps)}x sales" if ps else "")
                        + "."
                    ),
                    claim_kind=ClaimKind.DETERMINISTIC_CALCULATION,
                    methodology=(
                        "Market capitalisation from the close at the analysis "
                        "timestamp multiplied by the diluted share count from the "
                        "most recent filing available then, divided by the "
                        "corresponding trailing figure."
                    ),
                    uncertainty=(
                        "A multiple is not a valuation. Trailing multiples lag the "
                        "business, and the share count is a weighted average for the "
                        "reporting period rather than the count outstanding today."
                    ),
                    limitations=valuation.limitations,
                    calculations=[
                        c
                        for c in (
                            calculation_from_outcome(
                                valuation, "Price/earnings", "price_to_earnings", "ratio"
                            ),
                            calculation_from_outcome(
                                valuation, "Price/sales", "price_to_sales", "ratio"
                            ),
                            calculation_from_outcome(
                                valuation, "Market capitalisation", "market_cap", "currency"
                            ),
                        )
                        if c is not None
                    ],
                    evidence=[
                        method_evidence(
                            valuation,
                            "Multiples from the close and the then-latest filed figures",
                            value_field="market_cap",
                        )
                    ],
                )
            )
    return out


# --- top-level generators ----------------------------------------------------


SECTION_BUILDERS = (
    _performance_findings,
    _volatility_findings,
    _regime_findings,
    _benchmark_findings,
    _fundamental_findings,
    _data_quality_findings,
    _unavailable_findings,
)


def generate_report(
    bundle: AnalysisBundle,
    *,
    report_kind: str = "historical",
    attribution: list[dict] | None = None,
) -> Report:
    """Build a report from an analysis bundle.

    Deterministic: the same bundle produces the same report, which is what
    makes ``reproduction_key`` meaningful.
    """
    findings: list[Finding] = []
    for builder in SECTION_BUILDERS:
        try:
            findings.extend(builder(bundle))
        except Exception:  # pragma: no cover - a builder must not sink the report
            log.exception("report_section_failed", builder=builder.__name__)

    method_versions = {
        o.key: o.version for o in bundle.outcomes.values()
    }

    period = ""
    if bundle.period_start and bundle.period_end:
        period = f" — {bundle.period_start.date()} to {bundle.period_end.date()}"

    title = f"{bundle.symbol} {report_kind} analysis{period}"
    summary = _build_summary(bundle, findings)

    return Report(
        instrument_id=bundle.instrument_id,
        symbol=bundle.symbol,
        name=bundle.name,
        report_kind=report_kind,
        title=title,
        summary=summary,
        created_at=now_utc(),
        information_cutoff=bundle.as_of,
        period_start=bundle.period_start,
        period_end=bundle.period_end,
        findings=findings,
        limitations=bundle.disclosures,
        data_coverage={
            "observations": bundle.observations,
            "resolution": bundle.resolution,
            "adjustment_basis": bundle.adjustment_basis,
            "sources": bundle.data_sources,
            "gap_count": len(bundle.gaps),
            "gap_sessions": sum(g["expected_sessions"] for g in bundle.gaps),
            "quality_issue_count": len(bundle.quality_issues),
            "methods_available": sorted(bundle.available),
            "methods_unavailable": sorted(bundle.unavailable),
            "benchmark": bundle.benchmark_symbol,
        },
        method_versions=method_versions,
        inputs={
            "as_of": bundle.as_of.isoformat(),
            "resolution": bundle.resolution,
            "adjustment_basis": bundle.adjustment_basis,
            "period_start": bundle.period_start.isoformat() if bundle.period_start else None,
            "period_end": bundle.period_end.isoformat() if bundle.period_end else None,
            "purpose": bundle.purpose,
        },
        attribution=attribution or [],
        generator_version=GENERATOR_VERSION,
    )


def _build_summary(bundle: AnalysisBundle, findings: list[Finding]) -> str:
    """A factual summary, built from calculations rather than characterisation."""
    returns = bundle.outcomes.get("returns")
    vol = bundle.outcomes.get("realized_volatility")
    regime = bundle.outcomes.get("regime_classification")

    parts: list[str] = []
    if returns and returns.ok:
        parts.append(
            f"{bundle.symbol} returned {_pct(returns.values.get('total_return'))} "
            f"over {bundle.observations} {bundle.resolution} bars"
        )
    if vol and vol.ok:
        parts.append(
            f"annualised volatility was {_pct(vol.values.get('annualised_volatility'))}"
        )
    if regime and regime.ok:
        parts.append(
            f"the window is estimated as {str(regime.values.get('label')).replace('_', ' ')}"
        )

    body = "; ".join(parts) if parts else (
        f"No summary statistics could be computed for {bundle.symbol} in this window"
    )

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.claim_kind] = counts.get(finding.claim_kind, 0) + 1
    label_summary = ", ".join(
        f"{n} {kind.replace('_', ' ')}" for kind, n in sorted(counts.items())
    )

    caveat = ""
    if bundle.unavailable:
        caveat = (
            f" {len(bundle.unavailable)} method(s) could not be computed and are "
            "listed as unanswered."
        )
    return f"{body}. This report contains {label_summary}.{caveat}"
