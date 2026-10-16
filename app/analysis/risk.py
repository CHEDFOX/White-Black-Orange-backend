"""Risk framing and trade-structure analysis.

Regulatory posture, stated once and enforced throughout: this module produces
**educational research output**, not personalised investment advice. It never
recommends a trade. It answers conditional questions -- "if you entered here
with this stop, what would the risk/reward arithmetic be, and what would
invalidate the premise?" -- and every output carries
``disclaimer_code='educational_research'``.

Two engineering choices that matter more than they look:

* **Stops are volatility-scaled, not percentage-based.** A 5% stop is tight on
  one instrument and loose on another. Distances are quoted in ATR multiples
  with the percentage shown alongside.
* **Gap risk is reported separately from volatility.** An intraday stop does
  not hold through an overnight gap, and the single most common error in
  retail risk sizing is treating a stop as a guaranteed exit. The gap
  statistics are computed so the output can say how often a gap exceeded the
  proposed stop distance.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.analysis.registry import MethodInput, MethodSpec, register_method
from app.analysis.technical import log_returns, max_drawdown, true_ranges, wilder_smooth

DISCLAIMER_CODE = "educational_research"

RISK_DISCLAIMER = (
    "This is quantitative research output for education and analysis. It is not "
    "investment advice, not a recommendation to buy or sell, and takes no account "
    "of any individual's circumstances, objectives or risk tolerance."
)


@dataclass
class TradeStructure:
    """The arithmetic of one hypothetical entry, stop and target."""

    direction: str
    entry: float
    stop: float
    targets: list[float]
    stop_distance_percent: float
    stop_distance_atr: float | None
    risk_reward_ratios: list[float]
    position_size_at_1pct_risk: float | None
    notes: list[str] = field(default_factory=list)


@register_method(
    MethodSpec(
        key="risk_profile",
        version="1.0.0",
        family="risk",
        display_name="Risk profile",
        description=(
            "Volatility, value-at-risk, expected shortfall, drawdown history, "
            "gap risk and liquidity measures for the instrument over the window."
        ),
        purpose=(
            "Quantify how much the instrument moves and how badly it has "
            "behaved, so position size can be set from evidence."
        ),
        required_inputs=("close", "high", "low"),
        min_observations=100,
        output_schema={
            "annualised_volatility": "fraction_per_year",
            "atr_percent": "fraction",
            "var_95": "negative_fraction",
            "expected_shortfall_95": "negative_fraction",
            "max_drawdown": "negative_fraction",
            "gap_risk": "map",
        },
        missing_data_policy=(
            "Requires 100 bars. Liquidity measures are omitted, not defaulted, "
            "when volume is unavailable."
        ),
        limitations=(
            "Every figure here is realised history, not a forecast. The next "
            "period can and sometimes will be worse than anything in the sample.",
            "Value-at-risk says nothing about the size of a loss once that "
            "threshold is breached; expected shortfall is reported for that "
            "reason and is itself bounded by the observed sample.",
            "Drawdown is measured close-to-close, so the intraday experience was "
            "worse than the number shown.",
            "Volatility is not stable: a quiet window understates the risk of "
            "the next turbulent one.",
            "This is educational research output, not investment advice.",
        ),
        supported_resolutions=("1d", "1wk"),
        params_schema={"defaults": {"atr_period": 14, "var_confidence": 0.95}},
    )
)
def _risk_profile(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    rets = np.asarray(log_returns(data.close), dtype=float)
    confidence = float(params.get("var_confidence", 0.95))
    atr_period = int(params.get("atr_period", 14))
    last = float(data.close[-1])
    periods_per_year = float(data.trading_days_per_year if data.resolution == "1d" else 52)
    warnings: list[str] = []

    sigma = float(np.std(rets, ddof=1))
    annualised = sigma * math.sqrt(periods_per_year)

    alpha = 1.0 - confidence
    var = float(np.quantile(rets, alpha))
    tail = rets[rets <= var]
    expected_shortfall = float(np.mean(tail)) if tail.size else var
    var_99 = float(np.quantile(rets, 0.01))

    tr = true_ranges(data.high, data.low, data.close)
    atr = wilder_smooth(tr, min(atr_period, len(tr)))[-1]

    worst_dd, peak_idx, trough_idx = max_drawdown(data.close)
    current_dd = (last / float(max(data.close))) - 1.0

    # Gap risk: the overnight move an intraday stop cannot protect against.
    gaps = [
        (float(data.open[i]) / float(data.close[i - 1])) - 1.0
        for i in range(1, len(data.close))
    ] if data.open else []
    gap_stats: dict[str, Any] = {}
    if gaps:
        abs_gaps = [abs(g) for g in gaps]
        gap_stats = {
            "mean_absolute_gap": statistics.fmean(abs_gaps),
            "p95_absolute_gap": float(np.quantile(abs_gaps, 0.95)),
            "largest_gap_down": min(gaps),
            "largest_gap_up": max(gaps),
            "gaps_beyond_1_percent": sum(1 for g in abs_gaps if g > 0.01) / len(gaps),
        }
    else:
        warnings.append("no open prices available, so gap risk could not be measured")

    liquidity: dict[str, Any] = {}
    volumes = [v for v in data.volume if v is not None]
    if volumes:
        median_volume = statistics.median(volumes)
        liquidity = {
            "median_volume": median_volume,
            "median_dollar_volume": median_volume * last,
            "observations": len(volumes),
        }
    else:
        warnings.append(
            "no volume data available, so liquidity and execution risk are not "
            "assessed rather than being assumed adequate"
        )

    if annualised > 0.6:
        warnings.append(
            f"annualised volatility of {annualised:.0%} is very high; position "
            "sizes that look small in share terms carry large monetary risk"
        )

    return (
        {
            "annualised_volatility": annualised,
            "period_volatility": sigma,
            "atr": atr,
            "atr_percent": (atr / last) if atr else None,
            "var_95": var,
            "var_99": var_99,
            "expected_shortfall_95": expected_shortfall,
            "max_drawdown": worst_dd,
            "current_drawdown": current_dd,
            "drawdown_peak_date": (
                data.timestamps[peak_idx].isoformat() if data.timestamps else None
            ),
            "drawdown_trough_date": (
                data.timestamps[trough_idx].isoformat() if data.timestamps else None
            ),
            "gap_risk": gap_stats,
            "liquidity": liquidity,
            "worst_observed_return": float(np.min(rets)),
            "best_observed_return": float(np.max(rets)),
            "observations": len(rets),
            "disclaimer_code": DISCLAIMER_CODE,
            "disclaimer": RISK_DISCLAIMER,
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="trade_structure",
        version="1.0.0",
        family="risk",
        display_name="Trade structure arithmetic",
        description=(
            "For a hypothetical entry and a volatility-scaled stop, computes the "
            "stop distance in ATR and percentage terms, risk/reward ratios to "
            "supplied or derived targets, position size at a stated risk budget, "
            "and how often a historical overnight gap would have jumped the stop."
        ),
        purpose=(
            "Make the arithmetic of a hypothetical trade explicit, including the "
            "ways the stop would not have held."
        ),
        required_inputs=("close", "high", "low"),
        min_observations=100,
        output_schema={
            "structure": "trade_structure",
            "stop_jump_probability": "fraction",
            "invalidation_levels": "list_of_prices",
        },
        missing_data_policy="Requires 100 bars for an ATR and gap distribution.",
        limitations=(
            "A stop is not a guaranteed exit. The reported gap-jump frequency is "
            "the historical rate at which an overnight move exceeded the stop "
            "distance; in those cases the realised loss exceeds the planned one.",
            "Risk/reward ratios say nothing about the probability of reaching "
            "either level. A 3:1 ratio with a 20% hit rate loses money.",
            "Position sizing assumes the stop fills at its stated price and "
            "ignores commission, slippage and borrow cost.",
            "Support and resistance used for targets are descriptive clusters of "
            "past pivots, not levels with demonstrated predictive value.",
            "This is educational research output, not investment advice, and no "
            "entry, stop or target here is a recommendation.",
        ),
        supported_resolutions=("1d",),
        params_schema={
            "defaults": {
                "direction": "long",
                "entry": None,
                "atr_period": 14,
                "stop_atr_multiple": 2.0,
                "target_atr_multiples": [2.0, 4.0, 6.0],
                "account_risk_fraction": 0.01,
                "account_value": None,
            }
        },
    )
)
def _trade_structure(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    direction = str(params.get("direction") or "long").lower()
    if direction not in {"long", "short"}:
        direction = "long"

    last = float(data.close[-1])
    entry = float(params["entry"]) if params.get("entry") else last
    atr_period = int(params.get("atr_period", 14))
    stop_multiple = float(params.get("stop_atr_multiple", 2.0))
    target_multiples = [float(m) for m in params.get("target_atr_multiples", [2.0, 4.0, 6.0])]
    risk_fraction = float(params.get("account_risk_fraction", 0.01))
    account_value = params.get("account_value")

    tr = true_ranges(data.high, data.low, data.close)
    atr = wilder_smooth(tr, min(atr_period, len(tr)))[-1]
    warnings: list[str] = []
    if not atr:
        return (
            {"structure": None, "stop_jump_probability": None, "invalidation_levels": []},
            ["an ATR could not be computed, so no stop distance is proposed"],
        )

    stop_distance = stop_multiple * atr
    if direction == "long":
        stop = entry - stop_distance
        targets = [entry + m * atr for m in target_multiples]
    else:
        stop = entry + stop_distance
        targets = [entry - m * atr for m in target_multiples]

    if stop <= 0:
        warnings.append(
            "the proposed stop is at or below zero, which means the stop multiple "
            "exceeds the instrument's price in ATR terms"
        )

    reward_distances = [abs(t - entry) for t in targets]
    ratios = [d / stop_distance for d in reward_distances]

    position_size = None
    if account_value:
        risk_budget = float(account_value) * risk_fraction
        position_size = risk_budget / stop_distance

    # How often an overnight gap alone would have exceeded the stop distance.
    gap_jump_probability = None
    if data.open:
        gaps = [
            abs((float(data.open[i]) / float(data.close[i - 1])) - 1.0)
            for i in range(1, len(data.close))
        ]
        stop_fraction = stop_distance / entry
        if gaps:
            gap_jump_probability = sum(1 for g in gaps if g > stop_fraction) / len(gaps)
            if gap_jump_probability > 0.01:
                warnings.append(
                    f"in {gap_jump_probability:.1%} of sessions in this window the "
                    f"overnight gap alone exceeded the proposed stop distance of "
                    f"{stop_fraction:.1%}; in those cases the stop would have "
                    "filled worse than its stated price"
                )
    else:
        warnings.append("no open prices available, so gap-jump frequency is not reported")

    structure = TradeStructure(
        direction=direction,
        entry=entry,
        stop=stop,
        targets=targets,
        stop_distance_percent=stop_distance / entry,
        stop_distance_atr=stop_multiple,
        risk_reward_ratios=ratios,
        position_size_at_1pct_risk=position_size,
        notes=[
            f"stop placed {stop_multiple}x ATR ({stop_distance:.4f}) from entry",
            "risk/reward ratios carry no probability of reaching either level",
        ],
    )

    return (
        {
            "structure": {
                "direction": structure.direction,
                "entry": structure.entry,
                "stop": structure.stop,
                "targets": structure.targets,
                "stop_distance_percent": structure.stop_distance_percent,
                "stop_distance_atr": structure.stop_distance_atr,
                "risk_reward_ratios": structure.risk_reward_ratios,
                "position_size": structure.position_size_at_1pct_risk,
                "account_risk_fraction": risk_fraction,
                "notes": structure.notes,
            },
            "atr": atr,
            "stop_jump_probability": gap_jump_probability,
            "invalidation_levels": [stop],
            "disclaimer_code": DISCLAIMER_CODE,
            "disclaimer": RISK_DISCLAIMER,
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="position_sizing",
        version="1.0.0",
        family="risk",
        display_name="Volatility-targeted position sizing",
        description=(
            "Position weight that targets a stated portfolio volatility, with "
            "the Kelly fraction shown for reference and explicitly not "
            "recommended at full size."
        ),
        purpose="Convert a volatility estimate into a position weight.",
        required_inputs=("close",),
        min_observations=100,
        output_schema={
            "volatility_target_weight": "fraction",
            "kelly_fraction": "fraction",
            "notes": "list",
        },
        missing_data_policy="Requires 100 bars for a volatility estimate.",
        limitations=(
            "Volatility targeting uses a backward-looking estimate, so the "
            "realised portfolio volatility will differ, usually upward in "
            "stress.",
            "The Kelly fraction is reported for reference only. It assumes the "
            "estimated edge and distribution are correct, and it is catastrophic "
            "when the edge is overestimated -- which it almost always is. Half "
            "Kelly or less is the conventional practical cap.",
            "Sizing a single position ignores correlation with everything else "
            "held; concentrated correlated positions are not diversified by "
            "being individually small.",
            "This is educational research output, not investment advice.",
        ),
        supported_resolutions=("1d",),
        params_schema={
            "defaults": {
                "target_annual_volatility": 0.15,
                "max_weight": 1.0,
                "expected_edge": None,
            }
        },
    )
)
def _position_sizing(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    rets = np.asarray(log_returns(data.close), dtype=float)
    target = float(params.get("target_annual_volatility", 0.15))
    max_weight = float(params.get("max_weight", 1.0))
    periods_per_year = float(data.trading_days_per_year if data.resolution == "1d" else 52)

    sigma_annual = float(np.std(rets, ddof=1)) * math.sqrt(periods_per_year)
    warnings: list[str] = []
    if sigma_annual <= 0:
        return (
            {"volatility_target_weight": None, "kelly_fraction": None, "notes": []},
            ["realised volatility is zero, so a volatility-targeted weight is undefined"],
        )

    weight = min(target / sigma_annual, max_weight)

    kelly = None
    edge = params.get("expected_edge")
    if edge is not None:
        variance_annual = sigma_annual**2
        kelly = float(edge) / variance_annual if variance_annual > 0 else None
        if kelly is not None and kelly > 1.0:
            warnings.append(
                f"the Kelly fraction of {kelly:.2f} exceeds 1.0, implying leverage "
                "on an estimated edge; this is reported arithmetic, not a "
                "suggestion, and a modest error in the edge estimate makes full "
                "Kelly ruinous"
            )

    return (
        {
            "volatility_target_weight": weight,
            "annualised_volatility": sigma_annual,
            "target_annual_volatility": target,
            "weight_capped": weight >= max_weight,
            "kelly_fraction": kelly,
            "half_kelly_fraction": (kelly / 2) if kelly is not None else None,
            "notes": [
                f"a weight of {weight:.1%} targets {target:.0%} annual portfolio "
                f"volatility given a realised {sigma_annual:.0%}",
                "correlation with other holdings is not considered",
            ],
            "disclaimer_code": DISCLAIMER_CODE,
            "disclaimer": RISK_DISCLAIMER,
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="scenario_analysis",
        version="1.0.0",
        family="risk",
        display_name="Scenario analysis",
        description=(
            "Applies named shocks to the current price and reports the resulting "
            "levels, each shock's historical frequency in this window, and the "
            "assumptions behind it."
        ),
        purpose="Show what specified adverse and favourable moves would mean in price terms.",
        required_inputs=("close",),
        min_observations=100,
        output_schema={"scenarios": "list_of_scenarios"},
        missing_data_policy="Requires 100 bars to estimate historical frequencies.",
        limitations=(
            "A scenario is a conditional statement, not a forecast, and carries "
            "no probability beyond the historical frequency shown.",
            "Historical frequency is measured in this window only and does not "
            "transfer to a different regime.",
            "Single-factor shocks ignore that factors move together: a rate "
            "shock and an equity drawdown usually arrive on the same day.",
            "This is educational research output, not investment advice.",
        ),
        supported_resolutions=("1d",),
        params_schema={
            "defaults": {
                "shocks": [-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30],
                "horizon_bars": 21,
            }
        },
    )
)
def _scenario_analysis(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    shocks = [float(s) for s in params.get("shocks", [])]
    horizon = int(params.get("horizon_bars", 21))
    close = np.asarray(data.close, dtype=float)
    last = float(close[-1])

    # Historical frequency of a move of at least this size over the horizon.
    log_close = np.log(close)
    if len(log_close) > horizon:
        horizon_returns = log_close[horizon:] - log_close[:-horizon]
    else:
        horizon_returns = np.array([])

    scenarios = []
    for shock in sorted(shocks):
        target_log = math.log(1.0 + shock)
        if horizon_returns.size:
            if shock < 0:
                frequency = float(np.mean(horizon_returns <= target_log))
            else:
                frequency = float(np.mean(horizon_returns >= target_log))
        else:
            frequency = None
        scenarios.append(
            {
                "label": f"{shock:+.0%}",
                "shock": shock,
                "resulting_price": last * (1.0 + shock),
                "horizon_bars": horizon,
                "historical_frequency": frequency,
                "frequency_basis": (
                    f"share of overlapping {horizon}-bar windows in this sample "
                    f"({horizon_returns.size} windows) that moved at least this far"
                    if horizon_returns.size
                    else None
                ),
                "assumptions": [
                    "the shock is applied instantaneously to the current price",
                    "no dividend, split or corporate action within the horizon",
                ],
            }
        )

    return (
        {
            "scenarios": scenarios,
            "current_price": last,
            "horizon_bars": horizon,
            "overlapping_windows_used": int(horizon_returns.size),
            "disclaimer_code": DISCLAIMER_CODE,
            "disclaimer": RISK_DISCLAIMER,
        },
        [
            "historical frequencies are computed from overlapping windows, so "
            "they are not independent observations and the effective sample is "
            f"roughly {max(int(horizon_returns.size / horizon), 1)} windows"
        ],
    )
