"""Market-regime estimation and historical similarity features.

A regime label here is an *estimate from an explicit rule*, never a fact about
the market. The rule is stated in each method's description so a reader can
disagree with the classification rather than having to accept it. Two design
choices follow:

* Labels come with probabilities and a confidence, and ``INDETERMINATE`` is a
  first-class outcome. A classifier that always picks a label is less useful
  than one that admits ambiguity.
* The similarity feature vector is **scale-free**. Comparing a $4 stock to a
  $400 stock on raw prices finds nothing; comparing them on normalised
  volatility, trend and drawdown shape can.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np

from app.analysis.registry import MethodInput, MethodSpec, register_method
from app.analysis.technical import (
    log_returns,
    max_drawdown,
    sma,
    true_ranges,
    wilder_smooth,
)
from app.db.models.enums import RegimeLabel

# Bumping this invalidates stored episodes: a vector built under one version is
# not comparable to one built under another.
FEATURE_SET_VERSION = "1.0.0"


@register_method(
    MethodSpec(
        key="regime_classification",
        version="1.0.0",
        family="regime",
        display_name="Market regime estimate",
        description=(
            "Classifies the window using three explicit rules: the R-squared and "
            "sign of an OLS trend on log price (trending vs range-bound), realised "
            "volatility against its own longer-run history (high vs low "
            "volatility), and drawdown depth with volatility expansion (stress). "
            "Thresholds are parameters, not constants of nature."
        ),
        purpose=(
            "Condition other analysis on the kind of market the window was, so a "
            "statistic from a trending period is not applied to a quiet one."
        ),
        required_inputs=("close",),
        min_observations=60,
        output_schema={
            "label": "regime_label",
            "confidence": "fraction",
            "label_probabilities": "map_label_to_fraction",
            "diagnostics": "map_name_to_value",
        },
        missing_data_policy=(
            "Needs 60 bars so the volatility comparison has a baseline. Gaps "
            "shorten the effective window and the diagnostics report the count used."
        ),
        limitations=(
            "Regimes are not observable; this is one rule's opinion and a "
            "different rule will disagree at the boundaries.",
            "The classification is computed from the window's own data, so it "
            "identifies a regime only after it is visible -- it does not call "
            "turning points as they happen.",
            "Thresholds are conventions chosen for interpretability, not values "
            "estimated from data.",
            "A window spanning a regime change is classified as whatever dominates "
            "it; run change-point detection to find the split.",
        ),
        supported_resolutions=("1d", "1wk"),
        params_schema={
            "defaults": {
                "trend_r2_threshold": 0.5,
                "trend_slope_threshold": 0.0002,
                "high_vol_ratio": 1.5,
                "low_vol_ratio": 0.7,
                "stress_drawdown": -0.15,
                "baseline_window": 250,
                "recent_window": 20,
            }
        },
    )
)
def _regime_classification(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    close = np.asarray(data.close, dtype=float)
    rets = np.asarray(log_returns(data.close), dtype=float)
    n = len(close)

    recent_window = min(int(params["recent_window"]), len(rets))
    baseline_window = min(int(params["baseline_window"]), len(rets))

    # --- trend ---------------------------------------------------------------
    y = np.log(close)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - fitted) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # --- volatility ----------------------------------------------------------
    recent_vol = float(np.std(rets[-recent_window:], ddof=1)) if recent_window > 1 else 0.0
    baseline_vol = float(np.std(rets[-baseline_window:], ddof=1)) if baseline_window > 1 else 0.0
    vol_ratio = (recent_vol / baseline_vol) if baseline_vol > 0 else None

    # --- drawdown ------------------------------------------------------------
    worst_dd, _, _ = max_drawdown(data.close)
    running_peak = float(np.max(close))
    current_dd = (float(close[-1]) / running_peak) - 1.0

    diagnostics = {
        "trend_slope_per_period": float(slope),
        "trend_r_squared": r2,
        "recent_volatility": recent_vol,
        "baseline_volatility": baseline_vol,
        "volatility_ratio": vol_ratio,
        "max_drawdown": worst_dd,
        "current_drawdown": current_dd,
        "observations": n,
        "recent_window": recent_window,
        "baseline_window": baseline_window,
    }

    # Score each label, then normalise. Scores are deliberately simple and
    # readable; a reader can recompute them from the diagnostics above.
    scores: dict[str, float] = dict.fromkeys(RegimeLabel, 0.0)

    trending = r2 >= float(params["trend_r2_threshold"])
    slope_meaningful = abs(float(slope)) >= float(params["trend_slope_threshold"])
    if trending and slope_meaningful:
        target = RegimeLabel.TRENDING_UP if slope > 0 else RegimeLabel.TRENDING_DOWN
        scores[target] += r2
    else:
        scores[RegimeLabel.RANGE_BOUND] += max(0.0, 1.0 - r2)

    if vol_ratio is not None:
        if vol_ratio >= float(params["high_vol_ratio"]):
            scores[RegimeLabel.HIGH_VOLATILITY] += min(vol_ratio / 3.0, 1.0)
        elif vol_ratio <= float(params["low_vol_ratio"]):
            scores[RegimeLabel.LOW_VOLATILITY] += min(
                (float(params["low_vol_ratio"]) - vol_ratio + 0.3), 1.0
            )

    if current_dd <= float(params["stress_drawdown"]) and (vol_ratio or 0) >= 1.3:
        scores[RegimeLabel.STRESS] += min(abs(current_dd) * 3, 1.0)

    total = sum(scores.values())
    if total <= 0:
        probabilities = {RegimeLabel.INDETERMINATE.value: 1.0}
        label = RegimeLabel.INDETERMINATE
        confidence = 0.0
    else:
        probabilities = {k: v / total for k, v in scores.items() if v > 0}
        label = max(probabilities, key=lambda k: probabilities[k])
        confidence = probabilities[label]
        # A near-tie between two labels is ambiguity, and saying so is more
        # useful than asserting the marginal winner.
        ranked = sorted(probabilities.values(), reverse=True)
        if len(ranked) > 1 and ranked[0] - ranked[1] < 0.15:
            label = RegimeLabel.TRANSITION
            confidence = ranked[0] - ranked[1]

    return {
        "label": str(label),
        "confidence": confidence,
        "label_probabilities": {str(k): v for k, v in probabilities.items()},
        "diagnostics": diagnostics,
        "thresholds_used": {k: params[k] for k in sorted(params)},
    }


@register_method(
    MethodSpec(
        key="similarity_features",
        version=FEATURE_SET_VERSION,
        family="regime",
        display_name="Scale-free similarity feature vector",
        description=(
            "Normalised, dimensionless features describing a window's shape: "
            "trend, volatility, drawdown, return distribution, momentum and "
            "volume dynamics. Designed so two instruments at different price "
            "levels are comparable."
        ),
        purpose="Provide the vector the historical-similarity search matches on.",
        required_inputs=("close",),
        min_observations=60,
        output_schema={"features": "map_name_to_dimensionless"},
        missing_data_policy=(
            "Features that need volume are omitted (not zeroed) when volume is "
            "unavailable, and the omission is listed so a match is not made on a "
            "silently truncated vector."
        ),
        limitations=(
            "Two windows with similar features are similar in these measured "
            "respects only. Nothing guarantees the underlying situations, "
            "valuations or macro backdrops resemble each other.",
            "Chart-shape similarity has no established predictive value; the "
            "similarity engine must be benchmarked before its outputs are used.",
            "The feature set is a design choice. A different set finds different "
            "neighbours, which is why the version is part of the stored vector.",
        ),
        supported_resolutions=("1d", "1wk"),
        params_schema={"defaults": {"short_window": 20, "long_window": 60}},
    )
)
def _similarity_features(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    close = np.asarray(data.close, dtype=float)
    rets = np.asarray(log_returns(data.close), dtype=float)
    n = len(close)
    short_w = min(int(params["short_window"]), n)
    long_w = min(int(params["long_window"]), n)
    omitted: list[str] = []

    y = np.log(close)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / ss_tot if ss_tot > 0 else 0.0

    vol = float(np.std(rets, ddof=1))
    recent_vol = float(np.std(rets[-short_w:], ddof=1)) if short_w > 1 else vol
    worst_dd, _, _ = max_drawdown(data.close)
    current_dd = (float(close[-1]) / float(np.max(close))) - 1.0

    # Every feature below is dimensionless or volatility-normalised, which is
    # what makes a $4 and a $400 instrument comparable.
    features: dict[str, float] = {
        # Trend measured in volatility units per period, so it is scale-free.
        "trend_slope_in_vol_units": float(slope) / vol if vol > 0 else 0.0,
        "trend_r_squared": r2,
        "volatility_ratio_recent_to_window": (recent_vol / vol) if vol > 0 else 1.0,
        "max_drawdown": worst_dd,
        "current_drawdown": current_dd,
        "drawdown_recovery_fraction": (
            1.0 - (current_dd / worst_dd) if worst_dd < 0 else 1.0
        ),
        "up_period_fraction": float(np.mean(rets > 0)),
        "return_skewness": float(
            statistics.fmean([((r - rets.mean()) / vol) ** 3 for r in rets])
            if vol > 0
            else 0.0
        ),
        # Total window return in volatility units: a +20% move is a different
        # event in a quiet stock than in a volatile one.
        "window_return_in_vol_units": (
            float(np.sum(rets)) / (vol * math.sqrt(len(rets))) if vol > 0 else 0.0
        ),
        "position_in_window_range": (
            (float(close[-1]) - float(np.min(close)))
            / (float(np.max(close)) - float(np.min(close)))
            if float(np.max(close)) > float(np.min(close))
            else 0.5
        ),
    }

    short_ma = sma(data.close, short_w)[-1]
    long_ma = sma(data.close, long_w)[-1]
    if short_ma and long_ma:
        features["ma_spread_normalised"] = (short_ma - long_ma) / long_ma
    else:
        omitted.append("ma_spread_normalised")

    if data.high and data.low and len(data.high) == n:
        tr = true_ranges(data.high, data.low, data.close)
        atr = wilder_smooth(tr, min(14, len(tr)))[-1]
        if atr:
            features["atr_normalised"] = atr / float(close[-1])
        else:
            omitted.append("atr_normalised")
    else:
        omitted.append("atr_normalised")

    volumes = [v for v in data.volume if v is not None]
    if len(volumes) >= short_w + 1:
        recent_v = statistics.fmean(volumes[-short_w:])
        base_v = statistics.fmean(volumes)
        features["volume_ratio_recent_to_window"] = (
            recent_v / base_v if base_v > 0 else 1.0
        )
    else:
        # Omitted rather than defaulted: a 1.0 here would assert "volume was
        # normal", which we do not know.
        omitted.append("volume_ratio_recent_to_window")

    warnings = (
        [f"features omitted for lack of input data: {', '.join(omitted)}"]
        if omitted
        else []
    )
    return (
        {
            "features": features,
            "feature_set_version": FEATURE_SET_VERSION,
            "omitted_features": omitted,
            "observations": n,
        },
        warnings,
    )


def feature_distance(
    a: dict[str, float], b: dict[str, float], *, weights: dict[str, float] | None = None
) -> tuple[float, int]:
    """Weighted Euclidean distance over features present in both vectors.

    Returns the distance and the number of features compared. The count matters:
    a distance computed over four features is far weaker evidence of similarity
    than one over twelve, and the caller must be able to see that rather than
    receiving a bare number.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return float("inf"), 0
    total = 0.0
    weight_sum = 0.0
    for key in shared:
        w = (weights or {}).get(key, 1.0)
        total += w * (a[key] - b[key]) ** 2
        weight_sum += w
    if weight_sum <= 0:
        return float("inf"), 0
    # Normalise by total weight so vectors with different overlap counts are
    # comparable.
    return math.sqrt(total / weight_sum), len(shared)
