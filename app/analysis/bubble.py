"""Speculative-risk ("bubble") assessment.

Three constraints are built into the structure rather than left to the caller:

1. **No single indicator produces a verdict.** Each warning signal is evaluated
   independently and returns its own status, evidence and the reason it fired
   or did not. The composite score is a documented aggregation over the signals
   that could actually be evaluated, and the count of unavailable signals is
   reported so a score computed from three of nine is not mistaken for one
   computed from nine.
2. **Opposing evidence is collected, not just supporting evidence.** A signal
   that argues *against* elevated risk is recorded in the same structure. An
   assessment that only lists alarming facts is an argument, not an analysis.
3. **No crash date, ever.** The output is a *vulnerability* estimate --
   "conditions resemble historical episodes that ended badly" -- plus the
   conditions that would strengthen or weaken that reading. Overvaluation can
   persist for years, and the engine says so.

The honest summary of what this can do: it detects *conditions associated with*
past speculative episodes. It cannot tell you whether this one ends in a crash,
a long stagnation, or earnings catching up to the price.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from app.analysis.registry import MethodInput, MethodSpec, register_method
from app.analysis.technical import log_returns, max_drawdown

METHOD_VERSION = "1.0.0"

# Vulnerability bands. Deliberately qualitative: a decimal score implies a
# precision the underlying signals do not have.
BANDS: tuple[tuple[float, str], ...] = (
    (0.0, "limited"),
    (25.0, "moderate"),
    (50.0, "elevated"),
    (75.0, "high"),
)


@dataclass
class SignalResult:
    """One warning signal's independent verdict."""

    key: str
    display_name: str
    # triggered | not_triggered | unavailable
    status: str
    description: str
    observed_value: float | None = None
    threshold: float | None = None
    # Contribution weight, only meaningful when status != "unavailable".
    weight: float = 1.0
    # 0..1 intensity when triggered.
    intensity: float = 0.0
    evidence: str = ""
    counter_evidence: str = ""
    unavailable_reason: str = ""
    limitations: list[str] = field(default_factory=list)


def _band(score: float) -> str:
    label = BANDS[0][1]
    for threshold, name in BANDS:
        if score >= threshold:
            label = name
    return label


def _signal_price_appreciation(data: MethodInput, params: dict) -> SignalResult:
    """Rapid price appreciation relative to the instrument's own history."""
    close = np.asarray(data.close, dtype=float)
    window = min(int(params["appreciation_window"]), len(close) - 1)
    if window < 20:
        return SignalResult(
            key="price_appreciation",
            display_name="Rapid price appreciation",
            status="unavailable",
            description="Price gain over the recent window versus its longer history.",
            unavailable_reason=f"needs at least 21 bars, has {len(close)}",
        )

    recent_return = float(close[-1] / close[-window - 1]) - 1.0
    threshold = float(params["appreciation_threshold"])

    # Compare against the instrument's own distribution of same-length windows,
    # so a habitually volatile instrument is not flagged for ordinary behaviour.
    log_close = np.log(close)
    historical = (
        np.exp(log_close[window:] - log_close[:-window]) - 1.0
        if len(close) > window
        else np.array([])
    )
    percentile = (
        float(np.mean(historical <= recent_return) * 100) if historical.size else None
    )

    triggered = recent_return >= threshold
    intensity = min(max((recent_return - threshold) / threshold, 0.0), 1.0) if triggered else 0.0
    return SignalResult(
        key="price_appreciation",
        display_name="Rapid price appreciation",
        status="triggered" if triggered else "not_triggered",
        description=(
            f"Price change over the last {window} bars, compared with the "
            "distribution of all same-length windows in the sample."
        ),
        observed_value=recent_return,
        threshold=threshold,
        weight=1.0,
        intensity=intensity,
        evidence=(
            f"price rose {recent_return:+.1%} over {window} bars"
            + (
                f", which is at the {percentile:.0f}th percentile of this "
                "instrument's own history"
                if percentile is not None
                else ""
            )
        ),
        counter_evidence=(
            ""
            if triggered
            else f"price change of {recent_return:+.1%} is below the "
            f"{threshold:.0%} threshold"
        ),
        limitations=[
            "A large gain is not evidence of mispricing: it may reflect genuine "
            "improvement in the business.",
            "The threshold is a convention, not an estimated parameter.",
        ],
    )


def _signal_valuation(data: MethodInput, params: dict) -> SignalResult:
    """Valuation multiples against supplied sector or historical reference."""
    pe = data.scalars.get("price_to_earnings")
    ps = data.scalars.get("price_to_sales")
    reference_pe = data.scalars.get("reference_price_to_earnings")

    if pe is None and ps is None:
        return SignalResult(
            key="valuation_extreme",
            display_name="Valuation extremes",
            status="unavailable",
            description="Earnings and sales multiples versus a reference level.",
            unavailable_reason=(
                "no valuation multiple was available: fundamentals may be "
                "missing, or the company may be loss-making so no P/E exists"
            ),
        )

    threshold = float(params["pe_threshold"])
    observed = pe if pe is not None else None
    triggered = False
    evidence_parts: list[str] = []
    counter_parts: list[str] = []

    if pe is not None:
        evidence_parts.append(f"price/earnings of {pe:.1f}")
        if pe >= threshold:
            triggered = True
        else:
            counter_parts.append(f"P/E of {pe:.1f} is below the {threshold:.0f} threshold")
    if ps is not None:
        evidence_parts.append(f"price/sales of {ps:.1f}")
        if ps >= float(params["ps_threshold"]):
            triggered = True
    if reference_pe:
        ratio = (pe / reference_pe) if pe else None
        if ratio:
            evidence_parts.append(
                f"{ratio:.1f}x the reference P/E of {reference_pe:.1f}"
            )

    if pe and triggered:
        intensity = min(max((pe - threshold) / threshold, 0.0), 1.0)
    else:
        intensity = 0.5 if triggered else 0.0
    return SignalResult(
        key="valuation_extreme",
        display_name="Valuation extremes",
        status="triggered" if triggered else "not_triggered",
        description="Earnings and sales multiples versus a reference level.",
        observed_value=observed,
        threshold=threshold,
        weight=1.5,
        intensity=intensity,
        evidence="; ".join(evidence_parts),
        counter_evidence="; ".join(counter_parts),
        limitations=[
            "A high multiple can be justified by growth; it is not by itself "
            "evidence of a bubble.",
            "Multiples are only comparable within an industry.",
            "Trailing multiples lag the business.",
        ],
    )


def _signal_fundamental_divergence(data: MethodInput, params: dict) -> SignalResult:
    """Price growth running ahead of revenue or earnings growth."""
    price_growth = data.scalars.get("price_growth_1y")
    revenue_growth = data.scalars.get("revenue_growth_1y")
    earnings_growth = data.scalars.get("earnings_growth_1y")

    fundamental_growth = revenue_growth if revenue_growth is not None else earnings_growth
    if price_growth is None or fundamental_growth is None:
        return SignalResult(
            key="fundamental_divergence",
            display_name="Divergence from fundamentals",
            status="unavailable",
            description="Price growth compared with revenue or earnings growth.",
            unavailable_reason=(
                "needs both a price growth rate and a revenue or earnings growth "
                "rate over a comparable period"
            ),
        )

    gap = price_growth - fundamental_growth
    threshold = float(params["divergence_threshold"])
    triggered = gap >= threshold
    basis = "revenue" if revenue_growth is not None else "earnings"
    return SignalResult(
        key="fundamental_divergence",
        display_name="Divergence from fundamentals",
        status="triggered" if triggered else "not_triggered",
        description=f"Price growth compared with {basis} growth over one year.",
        observed_value=gap,
        threshold=threshold,
        weight=1.5,
        intensity=min(max(gap / (2 * threshold), 0.0), 1.0) if triggered else 0.0,
        evidence=(
            f"price grew {price_growth:+.1%} against {basis} growth of "
            f"{fundamental_growth:+.1%}, a gap of {gap:+.1%}"
        ),
        counter_evidence=(
            ""
            if triggered
            else f"the {gap:+.1%} gap between price and {basis} growth is below "
            f"the {threshold:.0%} threshold"
        ),
        limitations=[
            "Price should lead fundamentals when expectations change for good "
            "reason; divergence alone is not mispricing.",
            "One year is a short comparison window and is sensitive to its "
            "endpoints.",
        ],
    )


def _signal_volume_surge(data: MethodInput, params: dict) -> SignalResult:
    """Speculative participation proxied by a volume surge."""
    volumes = [v for v in data.volume if v is not None]
    recent_window = int(params["volume_recent_window"])
    if len(volumes) < recent_window * 4:
        return SignalResult(
            key="volume_surge",
            display_name="Speculative volume",
            status="unavailable",
            description="Recent volume versus the window's baseline.",
            unavailable_reason=(
                f"needs at least {recent_window * 4} bars with reported volume, "
                f"has {len(volumes)}"
            ),
        )

    recent = statistics.fmean(volumes[-recent_window:])
    baseline = statistics.fmean(volumes)
    ratio = recent / baseline if baseline > 0 else None
    if ratio is None:
        return SignalResult(
            key="volume_surge",
            display_name="Speculative volume",
            status="unavailable",
            description="Recent volume versus the window's baseline.",
            unavailable_reason="baseline volume is zero",
        )

    threshold = float(params["volume_ratio_threshold"])
    triggered = ratio >= threshold
    return SignalResult(
        key="volume_surge",
        display_name="Speculative volume",
        status="triggered" if triggered else "not_triggered",
        description="Recent volume versus the window's baseline.",
        observed_value=ratio,
        threshold=threshold,
        weight=0.75,
        intensity=min(max((ratio - threshold) / threshold, 0.0), 1.0) if triggered else 0.0,
        evidence=f"recent volume is {ratio:.2f}x the window baseline",
        counter_evidence=(
            "" if triggered else f"volume ratio of {ratio:.2f}x is below the "
            f"{threshold:.1f}x threshold"
        ),
        limitations=[
            "Elevated volume accompanies index inclusion, earnings and news as "
            "readily as speculation.",
            "Reported volume depends on venue consolidation and differs between "
            "providers.",
        ],
    )


def _signal_parabolic_acceleration(data: MethodInput, params: dict) -> SignalResult:
    """Acceleration: the hallmark of a late-stage speculative advance."""
    close = np.asarray(data.close, dtype=float)
    short_w = int(params["acceleration_short_window"])
    long_w = int(params["acceleration_long_window"])
    if len(close) < long_w + 1:
        return SignalResult(
            key="parabolic_acceleration",
            display_name="Parabolic acceleration",
            status="unavailable",
            description="Short-window rate of advance versus the longer-window rate.",
            unavailable_reason=f"needs {long_w + 1} bars, has {len(close)}",
        )

    short_rate = (math.log(float(close[-1]) / float(close[-short_w - 1]))) / short_w
    long_rate = (math.log(float(close[-1]) / float(close[-long_w - 1]))) / long_w
    if long_rate <= 0:
        return SignalResult(
            key="parabolic_acceleration",
            display_name="Parabolic acceleration",
            status="not_triggered",
            description="Short-window rate of advance versus the longer-window rate.",
            observed_value=None,
            weight=1.25,
            counter_evidence=(
                "the longer-window trend is flat or negative, so there is no "
                "advance to accelerate"
            ),
        )

    ratio = short_rate / long_rate
    threshold = float(params["acceleration_threshold"])
    triggered = ratio >= threshold and short_rate > 0
    return SignalResult(
        key="parabolic_acceleration",
        display_name="Parabolic acceleration",
        status="triggered" if triggered else "not_triggered",
        description="Short-window rate of advance versus the longer-window rate.",
        observed_value=ratio,
        threshold=threshold,
        weight=1.25,
        intensity=min(max((ratio - threshold) / threshold, 0.0), 1.0) if triggered else 0.0,
        evidence=(
            f"the last {short_w} bars advanced at {ratio:.1f}x the pace of the "
            f"last {long_w} bars"
        ),
        counter_evidence=(
            "" if triggered else f"the short-window pace is {ratio:.1f}x the "
            f"long-window pace, below the {threshold:.1f}x threshold"
        ),
        limitations=[
            "Acceleration appears in genuine re-ratings as well as in bubbles.",
            "The window lengths are conventions.",
        ],
    )


def _signal_volatility_expansion(data: MethodInput, params: dict) -> SignalResult:
    """Rising volatility during an advance: instability, not confidence."""
    rets = np.asarray(log_returns(data.close), dtype=float)
    recent_w = int(params["volatility_recent_window"])
    if len(rets) < recent_w * 3:
        return SignalResult(
            key="volatility_expansion",
            display_name="Volatility expansion during advance",
            status="unavailable",
            description="Recent volatility versus the window baseline, while price rises.",
            unavailable_reason=f"needs {recent_w * 3} returns, has {len(rets)}",
        )

    recent_vol = float(np.std(rets[-recent_w:], ddof=1))
    baseline_vol = float(np.std(rets, ddof=1))
    ratio = (recent_vol / baseline_vol) if baseline_vol > 0 else None
    rising = float(data.close[-1]) > float(data.close[-recent_w])

    if ratio is None:
        return SignalResult(
            key="volatility_expansion",
            display_name="Volatility expansion during advance",
            status="unavailable",
            description="Recent volatility versus the window baseline, while price rises.",
            unavailable_reason="baseline volatility is zero",
        )

    threshold = float(params["volatility_ratio_threshold"])
    triggered = ratio >= threshold and rising
    return SignalResult(
        key="volatility_expansion",
        display_name="Volatility expansion during advance",
        status="triggered" if triggered else "not_triggered",
        description="Recent volatility versus the window baseline, while price rises.",
        observed_value=ratio,
        threshold=threshold,
        weight=1.0,
        intensity=min(max((ratio - threshold) / threshold, 0.0), 1.0) if triggered else 0.0,
        evidence=(
            f"recent volatility is {ratio:.2f}x the window baseline"
            + (" while price is rising" if rising else "")
        ),
        counter_evidence=(
            ""
            if triggered
            else (
                f"volatility ratio of {ratio:.2f}x is below the {threshold:.1f}x "
                "threshold"
                if ratio < threshold
                else "price is not rising, so this is not an advance"
            )
        ),
        limitations=[
            "Volatility expansion also accompanies ordinary earnings and macro "
            "events.",
        ],
    )


def _signal_drawdown_absence(data: MethodInput, params: dict) -> SignalResult:
    """An unusually long advance without a correction."""
    close = data.close
    worst_dd, _, _ = max_drawdown(close)
    threshold = float(params["shallow_drawdown_threshold"])
    total_return = (float(close[-1]) / float(close[0])) - 1.0

    triggered = worst_dd > threshold and total_return > 0.3
    return SignalResult(
        key="drawdown_absence",
        display_name="Advance without correction",
        status="triggered" if triggered else "not_triggered",
        description=(
            "A substantial advance whose deepest pullback was unusually shallow."
        ),
        observed_value=worst_dd,
        threshold=threshold,
        weight=0.75,
        intensity=min(max((worst_dd - threshold) / abs(threshold), 0.0), 1.0) if triggered else 0.0,
        evidence=(
            f"the window returned {total_return:+.1%} with a maximum drawdown of "
            f"only {worst_dd:.1%}"
        ),
        counter_evidence=(
            ""
            if triggered
            else f"the window's maximum drawdown of {worst_dd:.1%} is deeper than "
            f"the {threshold:.1%} threshold, or the advance was modest"
        ),
        limitations=[
            "A shallow drawdown is a description of the past, and a quiet advance "
            "can continue for years.",
            "Measured close-to-close, so intraday pullbacks are not captured.",
        ],
    )


def _signal_distance_from_trend(data: MethodInput, params: dict) -> SignalResult:
    """Price stretched far above its own long moving average."""
    close = np.asarray(data.close, dtype=float)
    window = int(params["trend_window"])
    if len(close) < window:
        return SignalResult(
            key="distance_from_trend",
            display_name="Distance above long-run trend",
            status="unavailable",
            description=f"Price relative to its {window}-bar moving average.",
            unavailable_reason=f"needs {window} bars, has {len(close)}",
        )

    ma = float(np.mean(close[-window:]))
    distance = (float(close[-1]) / ma) - 1.0 if ma > 0 else None
    if distance is None:
        return SignalResult(
            key="distance_from_trend",
            display_name="Distance above long-run trend",
            status="unavailable",
            description=f"Price relative to its {window}-bar moving average.",
            unavailable_reason="moving average is zero",
        )

    # Compare against this instrument's own history of such stretches.
    rolling = [
        (float(close[i]) / float(np.mean(close[i - window + 1 : i + 1]))) - 1.0
        for i in range(window - 1, len(close))
    ]
    percentile = float(np.mean(np.asarray(rolling) <= distance) * 100) if rolling else None

    threshold = float(params["trend_distance_threshold"])
    triggered = distance >= threshold
    return SignalResult(
        key="distance_from_trend",
        display_name="Distance above long-run trend",
        status="triggered" if triggered else "not_triggered",
        description=f"Price relative to its {window}-bar moving average.",
        observed_value=distance,
        threshold=threshold,
        weight=1.0,
        intensity=min(max((distance - threshold) / threshold, 0.0), 1.0) if triggered else 0.0,
        evidence=(
            f"price is {distance:+.1%} above its {window}-bar average"
            + (f", the {percentile:.0f}th percentile of its own history" if percentile else "")
        ),
        counter_evidence=(
            "" if triggered else f"price is {distance:+.1%} from its {window}-bar "
            f"average, inside the {threshold:.0%} threshold"
        ),
        limitations=[
            "Distance from a moving average mean-reverts eventually, but 'eventually' "
            "can be years and the reversion can come from the average rising.",
        ],
    )


def _signal_concentration(data: MethodInput, params: dict) -> SignalResult:
    """Market or index concentration, when supplied."""
    concentration = data.scalars.get("index_weight")
    top_n_share = data.scalars.get("index_top_n_share")
    if concentration is None and top_n_share is None:
        return SignalResult(
            key="concentration",
            display_name="Index concentration",
            status="unavailable",
            description="The instrument's weight in its benchmark, or the index's top-N share.",
            unavailable_reason=(
                "index constituent weights are not ingested; this requires a "
                "licensed index-data provider"
            ),
            limitations=[
                "Historical index weights must be point-in-time; using today's "
                "weights for a past date is survivorship bias.",
            ],
        )

    threshold = float(params["concentration_threshold"])
    value = concentration if concentration is not None else top_n_share
    triggered = value >= threshold
    return SignalResult(
        key="concentration",
        display_name="Index concentration",
        status="triggered" if triggered else "not_triggered",
        description="The instrument's weight in its benchmark, or the index's top-N share.",
        observed_value=value,
        threshold=threshold,
        weight=0.75,
        intensity=min(max((value - threshold) / threshold, 0.0), 1.0) if triggered else 0.0,
        evidence=f"concentration measure of {value:.1%}",
        counter_evidence="" if triggered else f"concentration of {value:.1%} is below "
        f"the {threshold:.1%} threshold",
    )


def _signal_options_activity(data: MethodInput, params: dict) -> SignalResult:
    """Options-implied speculation. Unavailable without an options feed."""
    implied_vol = data.scalars.get("implied_volatility")
    call_put_ratio = data.scalars.get("call_put_volume_ratio")
    realized = data.scalars.get("realized_volatility")

    if implied_vol is None and call_put_ratio is None:
        return SignalResult(
            key="options_activity",
            display_name="Options-implied speculation",
            status="unavailable",
            description=(
                "Implied-versus-realised volatility and call/put volume skew."
            ),
            unavailable_reason=(
                "no options data is ingested; this requires a licensed options "
                "feed (OPRA or an equivalent vendor)"
            ),
            limitations=[
                "Options activity is one of the stronger speculation signals, so "
                "its absence materially weakens this assessment.",
            ],
        )

    triggered = False
    evidence_parts = []
    if implied_vol is not None and realized:
        premium = implied_vol / realized
        evidence_parts.append(f"implied volatility is {premium:.2f}x realised")
        if premium >= float(params["iv_rv_threshold"]):
            triggered = True
    if call_put_ratio is not None:
        evidence_parts.append(f"call/put volume ratio of {call_put_ratio:.2f}")
        if call_put_ratio >= float(params["call_put_threshold"]):
            triggered = True

    return SignalResult(
        key="options_activity",
        display_name="Options-implied speculation",
        status="triggered" if triggered else "not_triggered",
        description="Implied-versus-realised volatility and call/put volume skew.",
        observed_value=implied_vol,
        weight=1.25,
        intensity=0.6 if triggered else 0.0,
        evidence="; ".join(evidence_parts),
        counter_evidence="" if triggered else "options measures are inside their thresholds",
    )


SIGNAL_FUNCTIONS = (
    _signal_price_appreciation,
    _signal_valuation,
    _signal_fundamental_divergence,
    _signal_parabolic_acceleration,
    _signal_volume_surge,
    _signal_volatility_expansion,
    _signal_drawdown_absence,
    _signal_distance_from_trend,
    _signal_concentration,
    _signal_options_activity,
)


@register_method(
    MethodSpec(
        key="bubble_assessment",
        version=METHOD_VERSION,
        family="bubble",
        display_name="Speculative-risk (bubble) assessment",
        description=(
            "Evaluates ten warning signals independently -- price appreciation, "
            "valuation, divergence from fundamentals, acceleration, volume, "
            "volatility expansion, absence of correction, distance from trend, "
            "concentration and options activity -- and aggregates only those "
            "that could be evaluated into a vulnerability estimate, reporting "
            "both supporting and opposing evidence."
        ),
        purpose=(
            "Identify whether conditions resemble historical speculative "
            "episodes, without claiming to know how this one ends."
        ),
        required_inputs=("close",),
        min_observations=250,
        output_schema={
            "vulnerability_score": "score_0_100",
            "vulnerability_band": "label",
            "signals": "list_of_signals",
            "signals_triggered": "count",
            "signals_evaluated": "count",
            "signals_unavailable": "list",
            "strengthening_conditions": "list",
            "weakening_conditions": "list",
        },
        missing_data_policy=(
            "Each signal reports 'unavailable' with a reason rather than "
            "defaulting to 'not triggered'. The score is computed over evaluated "
            "signals only, and the unavailable count is reported so a score from "
            "three signals is not read as one from ten."
        ),
        limitations=(
            "This detects conditions ASSOCIATED WITH past speculative episodes. "
            "It cannot distinguish a bubble from a justified re-rating, and most "
            "instruments that trigger several signals do not crash.",
            "No crash date or crash probability is produced. Overvaluation can "
            "persist for years, and an episode can end in stagnation or in "
            "fundamentals catching up rather than in a decline.",
            "Thresholds are conventions chosen for interpretability. They are not "
            "estimated from a sample of historical bubbles, because that sample "
            "is tiny and selected after the fact.",
            "The score's weights are a documented design choice, not a fitted "
            "model. Two reasonable people would weight these signals differently.",
            "The strongest signals -- options positioning, leverage, margin debt, "
            "retail flows -- require licensed feeds this build does not have, so "
            "a low score may reflect missing data rather than low risk.",
            "Survivorship bias affects every historical bubble comparison: the "
            "episodes we remember are the ones that ended dramatically.",
        ),
        supported_resolutions=("1d", "1wk"),
        params_schema={
            "defaults": {
                "appreciation_window": 252,
                "appreciation_threshold": 1.0,
                "pe_threshold": 50.0,
                "ps_threshold": 15.0,
                "divergence_threshold": 0.5,
                "volume_recent_window": 20,
                "volume_ratio_threshold": 1.75,
                "acceleration_short_window": 21,
                "acceleration_long_window": 252,
                "acceleration_threshold": 2.5,
                "volatility_recent_window": 20,
                "volatility_ratio_threshold": 1.5,
                "shallow_drawdown_threshold": -0.1,
                "trend_window": 200,
                "trend_distance_threshold": 0.4,
                "concentration_threshold": 0.05,
                "iv_rv_threshold": 1.5,
                "call_put_threshold": 2.0,
            }
        },
    )
)
def _bubble_assessment(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    signals = [fn(data, params) for fn in SIGNAL_FUNCTIONS]

    evaluated = [s for s in signals if s.status != "unavailable"]
    unavailable = [s for s in signals if s.status == "unavailable"]
    triggered = [s for s in evaluated if s.status == "triggered"]

    # The score is the weighted intensity of triggered signals over the total
    # weight of *evaluated* signals. Dividing by the total weight of all ten
    # would let missing data look like safety.
    score = None
    if evaluated:
        total_weight = sum(s.weight for s in evaluated)
        achieved = sum(s.weight * max(s.intensity, 0.35 if s.status == "triggered" else 0.0)
                       for s in triggered)
        score = 100.0 * achieved / total_weight if total_weight > 0 else 0.0

    warnings: list[str] = []
    if len(unavailable) >= len(signals) / 2:
        warnings.append(
            f"{len(unavailable)} of {len(signals)} signals could not be evaluated "
            f"({', '.join(s.key for s in unavailable)}); the score reflects only "
            f"the {len(evaluated)} that could, and a low value may indicate "
            "missing data rather than low risk"
        )
    if len(triggered) == 1:
        warnings.append(
            f"only one signal ({triggered[0].key}) is triggered; a single "
            "indicator is not evidence of a speculative episode"
        )

    supporting = [
        {"signal": s.key, "evidence": s.evidence} for s in triggered if s.evidence
    ]
    opposing = [
        {"signal": s.key, "evidence": s.counter_evidence}
        for s in evaluated
        if s.status == "not_triggered" and s.counter_evidence
    ]

    strengthening = [
        "A further acceleration in price with no improvement in reported "
        "fundamentals.",
        "Valuation multiples expanding while growth rates decelerate.",
        "A volume surge concentrated in short-dated call options (requires an "
        "options feed to observe).",
        "Rising leverage or margin debt against the position (not observable in "
        "this build).",
        "A widening gap between price growth and revenue growth over several "
        "consecutive quarters.",
    ]
    weakening = [
        "Reported revenue or earnings growth catching up to the price advance.",
        "A normal correction that resets the distance from trend without "
        "breaking the longer-run uptrend.",
        "Valuation multiples compressing while the price holds, which is "
        "fundamentals validating the move.",
        "Volume and volatility returning to their baselines.",
    ]

    return (
        {
            "vulnerability_score": score,
            "vulnerability_band": _band(score) if score is not None else None,
            "signals": [asdict(s) for s in signals],
            "signals_triggered": len(triggered),
            "signals_evaluated": len(evaluated),
            "signals_total": len(signals),
            "signals_unavailable": [
                {"signal": s.key, "reason": s.unavailable_reason} for s in unavailable
            ],
            "supporting_evidence": supporting,
            "opposing_evidence": opposing,
            "strengthening_conditions": strengthening,
            "weakening_conditions": weakening,
            "method_version": METHOD_VERSION,
            "no_crash_prediction": (
                "This assessment does not predict a crash or a crash date. It "
                "reports whether current conditions resemble those seen in past "
                "speculative episodes, many of which resolved without a decline."
            ),
            "scoring_formula": (
                "weighted sum of triggered-signal intensities divided by the "
                "total weight of signals that could be evaluated, scaled to 0-100"
            ),
        },
        warnings,
    )
