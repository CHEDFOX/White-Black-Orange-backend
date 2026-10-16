"""Analysis orchestration.

Runs a set of methods over a loaded series and collects the results *together
with what failed and why*. A method that cannot run because there is not enough
data produces a recorded, explained absence -- not a silent omission and not a
fabricated value. That distinction is the whole reason this layer exists rather
than a list of direct calls.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy.orm import Session

from app.analysis.registry import MethodInput, MethodResult, get_spec, run_method
from app.core.errors import InsufficientDataError, NotFoundError, PlatformError
from app.core.logging import get_logger
from app.db.models.enums import AdjustmentBasis
from app.pit.availability import AsOf
from app.services.data_access import (
    FundamentalSnapshot,
    SeriesBundle,
    load_benchmark,
    load_fundamentals,
    load_series,
)

log = get_logger(__name__)

# Method sets for each view. Ordered so a reader sees the description of what
# happened before any interpretation of it.
PAST_METHODS = (
    "returns",
    "realized_volatility",
    "drawdown",
    "trend_strength",
    "relative_strength",
    "return_distribution",
    "regime_classification",
    "volume_profile",
    "gap_analysis",
    "change_point",
)

PRESENT_METHODS = (
    "returns",
    "moving_averages",
    "rsi",
    "macd",
    "bollinger_bands",
    "atr",
    "realized_volatility",
    "drawdown",
    "support_resistance",
    "regime_classification",
    "relative_strength",
    "volume_profile",
    "trend_strength",
)

STATISTICAL_METHODS = (
    "return_distribution",
    "autocorrelation",
    "stationarity",
    "volatility_clustering",
    "change_point",
    "extreme_value",
    "correlation_matrix",
    "seasonality",
)

FUNDAMENTAL_METHODS = (
    "profitability",
    "balance_sheet_strength",
    "cash_flow_quality",
    "valuation_ratios",
    "growth_trajectory",
)

RISK_METHODS = ("risk_profile", "position_sizing", "scenario_analysis")


@dataclass
class MethodOutcome:
    """A method's result, or the reason it could not run."""

    key: str
    version: str
    display_name: str
    ok: bool
    values: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    observations_used: int = 0
    input_fingerprint: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    unavailable_reason: str | None = None
    unavailable_code: str | None = None

    @classmethod
    def from_result(cls, result: MethodResult, display_name: str) -> MethodOutcome:
        return cls(
            key=result.method_key,
            version=result.method_version,
            display_name=display_name,
            ok=True,
            values=result.values,
            warnings=list(result.warnings),
            limitations=list(result.limitations),
            observations_used=result.observations_used,
            input_fingerprint=result.input_fingerprint,
            params=result.params,
        )

    @classmethod
    def unavailable(
        cls, key: str, display_name: str, version: str, reason: str, code: str
    ) -> MethodOutcome:
        return cls(
            key=key,
            version=version,
            display_name=display_name,
            ok=False,
            unavailable_reason=reason,
            unavailable_code=code,
        )

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "version": self.version,
            "display_name": self.display_name,
            "ok": self.ok,
            "values": self.values,
            "warnings": self.warnings,
            "limitations": self.limitations,
            "observations_used": self.observations_used,
            "input_fingerprint": self.input_fingerprint,
            "params": self.params,
            "unavailable_reason": self.unavailable_reason,
            "unavailable_code": self.unavailable_code,
        }


@dataclass
class AnalysisBundle:
    """Everything one analysis run produced, including its own limitations."""

    instrument_id: str
    symbol: str
    name: str
    as_of: dt.datetime
    purpose: str
    resolution: str
    adjustment_basis: str
    period_start: dt.datetime | None
    period_end: dt.datetime | None
    observations: int
    currency: str
    outcomes: dict[str, MethodOutcome] = field(default_factory=dict)
    data_sources: list[str] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    quality_issues: list[dict] = field(default_factory=list)
    coverage_notes: list[str] = field(default_factory=list)
    benchmark_symbol: str | None = None
    benchmark_note: str | None = None
    fundamentals_note: str | None = None
    disclosures: list[str] = field(default_factory=list)

    @property
    def available(self) -> dict[str, MethodOutcome]:
        return {k: v for k, v in self.outcomes.items() if v.ok}

    @property
    def unavailable(self) -> dict[str, MethodOutcome]:
        return {k: v for k, v in self.outcomes.items() if not v.ok}

    def value(self, method: str, field_name: str) -> Any:
        outcome = self.outcomes.get(method)
        if outcome is None or not outcome.ok:
            return None
        return outcome.values.get(field_name)

    def as_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "symbol": self.symbol,
            "name": self.name,
            "as_of": self.as_of.isoformat(),
            "purpose": self.purpose,
            "resolution": self.resolution,
            "adjustment_basis": self.adjustment_basis,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "observations": self.observations,
            "currency": self.currency,
            "methods": {k: v.as_dict() for k, v in self.outcomes.items()},
            "methods_available": sorted(self.available),
            "methods_unavailable": {
                k: v.unavailable_reason for k, v in self.unavailable.items()
            },
            "data_sources": self.data_sources,
            "gaps": self.gaps,
            "quality_issues": self.quality_issues,
            "coverage_notes": self.coverage_notes,
            "benchmark_symbol": self.benchmark_symbol,
            "benchmark_note": self.benchmark_note,
            "fundamentals_note": self.fundamentals_note,
            "disclosures": self.disclosures,
        }


def run_methods(
    method_keys: tuple[str, ...],
    data: MethodInput,
    *,
    params_by_method: dict[str, dict] | None = None,
) -> dict[str, MethodOutcome]:
    """Run each method, recording failures as explained absences."""
    out: dict[str, MethodOutcome] = {}
    for key in method_keys:
        try:
            spec = get_spec(key)
        except NotFoundError:
            log.warning("method_not_registered", method=key)
            continue

        params = (params_by_method or {}).get(key, {})
        try:
            result = run_method(key, data, params)
        except InsufficientDataError as exc:
            # The designed outcome for thin data. Recorded with its reason so a
            # report can say "volatility was not computed because ..." instead
            # of leaving a hole a reader fills with an assumption.
            out[key] = MethodOutcome.unavailable(
                key, spec.display_name, spec.version, exc.message, exc.code
            )
        except PlatformError as exc:
            out[key] = MethodOutcome.unavailable(
                key, spec.display_name, spec.version, exc.message, exc.code
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("method_failed", method=key)
            out[key] = MethodOutcome.unavailable(
                key,
                spec.display_name,
                spec.version,
                f"{type(exc).__name__}: {exc}",
                "method_error",
            )
        else:
            out[key] = MethodOutcome.from_result(result, spec.display_name)
    return out


def analyse(
    session: Session,
    instrument,
    *,
    as_of: AsOf,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    resolution: str = "1d",
    adjustment_basis: str = AdjustmentBasis.TOTAL_RETURN,
    method_keys: tuple[str, ...] = PRESENT_METHODS,
    include_fundamentals: bool = True,
    include_benchmark: bool = True,
    benchmark_ticker: str | None = None,
    params_by_method: dict[str, dict] | None = None,
    purpose: str = "analysis",
) -> AnalysisBundle:
    """Load a point-in-time series and run the requested methods over it."""
    series = load_series(
        session,
        instrument,
        as_of=as_of,
        start=start,
        end=end,
        resolution=resolution,
        adjustment_basis=adjustment_basis,
    )

    benchmark: SeriesBundle | None = None
    benchmark_note: str | None = None
    if include_benchmark:
        benchmark = load_benchmark(
            session,
            as_of=as_of,
            start=series.first_timestamp,
            end=series.last_timestamp,
            resolution=resolution,
            ticker=benchmark_ticker,
        )
        if benchmark is None:
            benchmark_note = (
                f"The benchmark {benchmark_ticker or 'configured default'} is not "
                "ingested or has too little history in this window, so "
                "benchmark-relative measures (beta, alpha, relative strength) "
                "are not reported."
            )
        elif benchmark.instrument.id == instrument.id:
            benchmark = None
            benchmark_note = (
                "The instrument is its own benchmark, so relative measures are "
                "not reported."
            )

    fundamentals = FundamentalSnapshot()
    fundamentals_note: str | None = None
    if include_fundamentals:
        fundamentals = load_fundamentals(session, instrument, as_of=as_of)
        if fundamentals.is_empty:
            fundamentals_note = (
                "; ".join(fundamentals.notes)
                or f"No fundamentals were available for {instrument.symbol} as of "
                f"{as_of.cutoff.date()}."
            )

    scalars = dict(fundamentals.scalars)
    # Expose derived scalars the bubble and risk engines read by name.
    if len(series) > 1:
        scalars["realized_volatility"] = _annualised_vol(series)
    if len(series) > 252:
        scalars["price_growth_1y"] = (series.close[-1] / series.close[-253]) - 1.0

    data = series.to_method_input(
        benchmark=benchmark,
        scalars=scalars,
        series=fundamentals.series,
    )

    if "bubble_assessment" in method_keys:
        data = _with_derived_fundamental_scalars(data, scalars, fundamentals)

    outcomes = run_methods(method_keys, data, params_by_method=params_by_method)

    bundle = AnalysisBundle(
        instrument_id=str(instrument.id),
        symbol=instrument.symbol,
        name=instrument.name,
        as_of=as_of.cutoff,
        purpose=purpose,
        resolution=resolution,
        adjustment_basis=adjustment_basis,
        period_start=series.first_timestamp,
        period_end=series.last_timestamp,
        observations=len(series),
        currency=series.currency,
        outcomes=outcomes,
        data_sources=series.source_keys,
        gaps=series.gaps,
        quality_issues=series.quality_issues,
        coverage_notes=series.coverage_notes,
        benchmark_symbol=benchmark.instrument.symbol if benchmark else None,
        benchmark_note=benchmark_note,
        fundamentals_note=fundamentals_note,
    )
    bundle.disclosures = build_disclosures(bundle, as_of, series, fundamentals)
    return bundle


# The bubble engine reads *derived* fundamentals by name, but the fundamentals
# snapshot carries only raw reported facts, so nothing ever supplied these keys.
# The two signals that need them are the two highest-weighted of the ten, and
# both reported "unavailable" for every instrument while the same API response
# showed the very ratios they were asking for.
_VALUATION_SCALARS = ("price_to_earnings", "price_to_sales")
# growth_trajectory reports period-over-period growth. Mapping it to a name
# ending in `_1y` is only honest when the periods really are annual.
_GROWTH_SCALARS = {
    "revenue_growth_latest": "revenue_growth_1y",
    "net_income_growth_latest": "earnings_growth_1y",
}


def _with_derived_fundamental_scalars(
    data: MethodInput,
    scalars: dict[str, float],
    fundamentals: FundamentalSnapshot,
) -> MethodInput:
    """Add the derived ratios and growth rates the bubble signals read.

    Computed by running the same registered methods that produce the numbers
    shown to the reader, so a signal can never disagree with the valuation
    table next to it.
    """
    derived = dict(scalars)

    try:
        ratios = run_method("valuation_ratios", data).values
    except (InsufficientDataError, PlatformError):
        ratios = {}
    for key in _VALUATION_SCALARS:
        value = ratios.get(key)
        if value is not None:
            derived[key] = float(value)

    annual = bool(fundamentals.period_labels) and all(
        label.endswith(" FY") for label in fundamentals.period_labels
    )
    if annual:
        try:
            growth = run_method("growth_trajectory", data).values
        except (InsufficientDataError, PlatformError):
            growth = {}
        for source, target in _GROWTH_SCALARS.items():
            value = growth.get(source)
            if value is not None:
                derived[target] = float(value)

    if derived == scalars:
        return data

    return replace(data, scalars=derived)


def _annualised_vol(series: SeriesBundle) -> float:
    import math
    import statistics

    rets = [
        math.log(series.close[i] / series.close[i - 1])
        for i in range(1, len(series.close))
    ]
    if len(rets) < 2:
        return 0.0
    return statistics.stdev(rets) * math.sqrt(252)


def build_disclosures(
    bundle: AnalysisBundle,
    as_of: AsOf,
    series: SeriesBundle,
    fundamentals: FundamentalSnapshot,
) -> list[str]:
    """Assemble the statements a report must repeat.

    Collected here rather than written per report so a caveat cannot be
    dropped by whoever renders the output.
    """
    out: list[str] = []

    if as_of.is_live:
        out.append(
            "This is a current analysis using the most recent stored data. The "
            "latest session may be incomplete."
        )
    else:
        out.append(
            f"This is a point-in-time reconstruction as of "
            f"{as_of.cutoff.isoformat()}. Only information available at that "
            "instant was used; nothing published later was visible to it."
        )
    out.extend(as_of.disclosures)

    if series.adjustment_basis == AdjustmentBasis.RAW:
        out.append(
            "Prices are on the raw (unadjusted) basis, so returns across a split "
            "or dividend in this window are not comparable. Use the total-return "
            "basis for performance measurement."
        )

    if series.gaps:
        total = sum(g["expected_sessions"] for g in series.gaps)
        out.append(
            f"{total} expected trading session(s) in this window have no data, in "
            f"{len(series.gaps)} gap(s). Nothing was interpolated across them, so "
            "statistics measure the sessions present."
        )

    errors = [i for i in series.quality_issues if i["severity"] == "error"]
    warnings = [i for i in series.quality_issues if i["severity"] == "warning"]
    if errors:
        out.append(
            f"{len(errors)} bar(s) in this window failed structural validation and "
            "were excluded from the stored series."
        )
    if warnings:
        kinds = sorted({i["kind"] for i in warnings})
        out.append(
            f"{len(warnings)} data-quality warning(s) are recorded in this window "
            f"({', '.join(kinds)})."
        )

    if bundle.benchmark_note:
        out.append(bundle.benchmark_note)
    if bundle.fundamentals_note:
        out.append(bundle.fundamentals_note)
    out.extend(fundamentals.notes)

    if bundle.unavailable:
        out.append(
            f"{len(bundle.unavailable)} requested method(s) could not be computed: "
            + "; ".join(
                f"{o.display_name} ({o.unavailable_reason})"
                for o in bundle.unavailable.values()
            )
        )

    if series.source_keys:
        out.append(
            "Price data for this analysis came from: " + ", ".join(series.source_keys) + "."
        )

    # De-duplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped
