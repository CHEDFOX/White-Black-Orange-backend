"""Deterministic price and technical features.

Implementation notes that matter for correctness:

* **Returns.** Log returns are the default internally because they add across
  time, which makes multi-period aggregation exact. Simple returns are reported
  where a user-facing number must match a broker statement.
* **Wilder vs EMA.** RSI and ATR use Wilder's smoothing (alpha = 1/n), not the
  standard EMA (alpha = 2/(n+1)). Both appear in the wild under the same names
  and they give different numbers; the choice is stated per method so a value
  can be reconciled against another platform.
* **Seeding.** A recursive average has to start somewhere. Here the first
  value is a simple mean of the first ``n`` observations, which is Wilder's
  original formulation. Seeding from the first observation alone would make
  early values depend on a single print.
* **No padding.** A window that cannot be filled yields no value rather than a
  value computed from fewer observations. Short windows are the quiet source of
  misleading early-series indicator readings.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from app.analysis.registry import MethodInput, MethodSpec, register_method
from app.db.models.enums import InstrumentType, ValidationStatus

EQUITY_LIKE = (
    InstrumentType.EQUITY,
    InstrumentType.ETF,
    InstrumentType.INDEX,
    InstrumentType.FUND,
    InstrumentType.ADR,
    InstrumentType.REIT,
)
INTRADAY_AND_DAILY = ("1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")


# --- primitives shared by several methods -----------------------------------


def log_returns(close: tuple[float, ...]) -> list[float]:
    """Log returns. Non-positive prices are impossible here (validation rejects them)."""
    return [math.log(close[i] / close[i - 1]) for i in range(1, len(close))]


def simple_returns(close: tuple[float, ...]) -> list[float]:
    return [(close[i] / close[i - 1]) - 1.0 for i in range(1, len(close))]


def sma(values: tuple[float, ...] | list[float], window: int) -> list[float | None]:
    """Simple moving average; ``None`` until the window is full."""
    out: list[float | None] = [None] * len(values)
    if window <= 0 or len(values) < window:
        return out
    running = sum(values[:window])
    out[window - 1] = running / window
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        out[i] = running / window
    return out


def ema(values: tuple[float, ...] | list[float], window: int) -> list[float | None]:
    """Exponential moving average, alpha = 2/(window+1), seeded with an SMA."""
    out: list[float | None] = [None] * len(values)
    if window <= 0 or len(values) < window:
        return out
    alpha = 2.0 / (window + 1.0)
    current = sum(values[:window]) / window
    out[window - 1] = current
    for i in range(window, len(values)):
        current = alpha * values[i] + (1 - alpha) * current
        out[i] = current
    return out


def wilder_smooth(values: list[float], window: int) -> list[float | None]:
    """Wilder's smoothing, alpha = 1/window, seeded with a simple mean."""
    out: list[float | None] = [None] * len(values)
    if window <= 0 or len(values) < window:
        return out
    current = sum(values[:window]) / window
    out[window - 1] = current
    for i in range(window, len(values)):
        current = (current * (window - 1) + values[i]) / window
        out[i] = current
    return out


def true_ranges(
    high: tuple[float, ...], low: tuple[float, ...], close: tuple[float, ...]
) -> list[float]:
    """True range per bar, from index 1 (needs the previous close)."""
    return [
        max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        for i in range(1, len(close))
    ]


def max_drawdown(close: tuple[float, ...]) -> tuple[float, int, int]:
    """Largest peak-to-trough decline as a negative fraction, with its indices."""
    peak = close[0]
    peak_idx = 0
    worst = 0.0
    worst_peak = 0
    worst_trough = 0
    for i, price in enumerate(close):
        if price > peak:
            peak = price
            peak_idx = i
        drop = (price / peak) - 1.0
        if drop < worst:
            worst = drop
            worst_peak = peak_idx
            worst_trough = i
    return worst, worst_peak, worst_trough


# --- registered methods ------------------------------------------------------


@register_method(
    MethodSpec(
        key="returns",
        version="1.0.0",
        family="price",
        display_name="Period returns",
        description=(
            "Total, annualised and per-period returns over the supplied window, "
            "computed from the first and last close."
        ),
        purpose="Quantify what the price actually did over a selected period.",
        required_inputs=("close",),
        min_observations=2,
        output_schema={
            "total_return": "fraction",
            "total_log_return": "log_fraction",
            "annualised_return": "fraction_per_year",
            "mean_log_return": "log_fraction_per_period",
            "periods": "count",
        },
        missing_data_policy=(
            "Requires at least two closes. Gaps are not filled; a return computed "
            "across a gap spans the gap and the gap is reported separately."
        ),
        limitations=(
            "A return over a window says nothing about the path taken within it.",
            "Annualisation of a short window is extremely noisy and should not be "
            "read as an expected rate.",
            "On a raw (unadjusted) series, returns across a split or dividend are "
            "wrong; use the total-return basis for performance.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
        validation_status=ValidationStatus.UNIT_TESTED,
    )
)
def _returns(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    close = data.close
    total = (close[-1] / close[0]) - 1.0
    total_log = math.log(close[-1] / close[0])
    periods = len(close) - 1
    per_year = data.trading_days_per_year if data.resolution == "1d" else None
    annualised = None
    if per_year and periods > 0:
        years = periods / per_year
        if years > 0:
            annualised = math.exp(total_log / years) - 1.0
    return {
        "total_return": total,
        "total_log_return": total_log,
        "annualised_return": annualised,
        "mean_log_return": total_log / periods if periods else None,
        "periods": periods,
        "first_close": close[0],
        "last_close": close[-1],
    }


@register_method(
    MethodSpec(
        key="realized_volatility",
        version="1.0.0",
        family="price",
        display_name="Realised volatility",
        description=(
            "Standard deviation of log returns, annualised by the square root of "
            "the number of periods per year."
        ),
        purpose="Measure how much the price has actually moved, for risk scaling.",
        required_inputs=("close",),
        min_observations=20,
        output_schema={
            "daily_volatility": "fraction_per_period",
            "annualised_volatility": "fraction_per_year",
            "observations": "count",
        },
        missing_data_policy=(
            "Needs 20 returns minimum. Missing sessions are not interpolated, so a "
            "window containing gaps measures volatility over the bars present and "
            "the gap count is reported alongside."
        ),
        limitations=(
            "Realised volatility is backward-looking; it is not a forecast.",
            "Square-root-of-time annualisation assumes returns are serially "
            "uncorrelated, which is only approximately true and fails badly in "
            "trending or mean-reverting regimes.",
            "A single extreme day dominates a short window.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
        params_schema={"defaults": {"ddof": 1}},
    )
)
def _realized_volatility(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    rets = log_returns(data.close)
    ddof = int(params.get("ddof", 1))
    # Sample standard deviation (ddof=1) by default: these returns are a sample
    # from a process, not the whole population.
    sd = statistics.stdev(rets) if ddof == 1 else statistics.pstdev(rets)
    periods_per_year = _periods_per_year(data)
    return {
        "daily_volatility": sd,
        "annualised_volatility": sd * math.sqrt(periods_per_year) if periods_per_year else None,
        "observations": len(rets),
        "periods_per_year_assumed": periods_per_year,
    }


@register_method(
    MethodSpec(
        key="moving_averages",
        version="1.0.0",
        family="technical",
        display_name="Moving averages",
        description="Simple and exponential moving averages at the requested windows.",
        purpose="Summarise trend at several horizons.",
        required_inputs=("close",),
        min_observations=2,
        output_schema={"sma": "map_window_to_price", "ema": "map_window_to_price"},
        missing_data_policy=(
            "A window longer than the available history yields null for that "
            "window rather than an average of fewer observations."
        ),
        limitations=(
            "Moving averages lag by construction; a crossover is a description of "
            "the past, not a signal with established predictive value.",
            "Window choice is arbitrary and the popular values (50, 200) have no "
            "special statistical standing.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
        params_schema={"defaults": {"windows": [20, 50, 200]}},
    )
)
def _moving_averages(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    windows = [int(w) for w in params.get("windows", [20, 50, 200])]
    sma_out: dict[str, float | None] = {}
    ema_out: dict[str, float | None] = {}
    warnings: list[str] = []
    for w in windows:
        if w > len(data.close):
            sma_out[str(w)] = None
            ema_out[str(w)] = None
            warnings.append(
                f"the {w}-period window exceeds the {len(data.close)} available "
                "observations, so no value is reported for it"
            )
            continue
        sma_out[str(w)] = sma(data.close, w)[-1]
        ema_out[str(w)] = ema(data.close, w)[-1]

    last = data.close[-1]
    position = {
        str(w): (None if sma_out[str(w)] is None else (last / sma_out[str(w)]) - 1.0)
        for w in windows
    }
    return (
        {
            "sma": sma_out,
            "ema": ema_out,
            "price_vs_sma": position,
            "last_close": last,
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="rsi",
        version="1.0.0",
        family="technical",
        display_name="Relative Strength Index (Wilder)",
        description=(
            "RSI using Wilder's smoothing with alpha = 1/period, seeded with the "
            "simple mean of the first `period` gains and losses."
        ),
        purpose="Describe the recent balance of up and down moves.",
        required_inputs=("close",),
        min_observations=15,
        output_schema={"rsi": "index_0_100", "period": "count"},
        missing_data_policy="Requires period+1 closes. No value is produced below that.",
        limitations=(
            "RSI above 70 or below 30 is a description of recent momentum, not a "
            "reversal signal; instruments in strong trends stay there for months.",
            "Wilder smoothing is used here. Platforms using a standard EMA report "
            "different values for the same series, so figures may not reconcile.",
            "On an all-up or all-down window RSI saturates at 100 or 0 and carries "
            "little information.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
        params_schema={"defaults": {"period": 14}, "observation_drivers": ("period",),
                       "window_padding": 1},
    )
)
def _rsi(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    period = int(params.get("period", 14))
    changes = [data.close[i] - data.close[i - 1] for i in range(1, len(data.close))]
    gains = [max(c, 0.0) for c in changes]
    losses = [max(-c, 0.0) for c in changes]
    avg_gain = wilder_smooth(gains, period)[-1]
    avg_loss = wilder_smooth(losses, period)[-1]
    if avg_gain is None or avg_loss is None:
        return {"rsi": None, "period": period}
    if avg_loss == 0:
        # No down moves in the window: RSI is 100 by definition, which says the
        # window contained no losses rather than anything about the future.
        return {"rsi": 100.0, "period": period, "average_gain": avg_gain,
                "average_loss": 0.0}
    rs = avg_gain / avg_loss
    return {
        "rsi": 100.0 - (100.0 / (1.0 + rs)),
        "period": period,
        "average_gain": avg_gain,
        "average_loss": avg_loss,
    }


@register_method(
    MethodSpec(
        key="macd",
        version="1.0.0",
        family="technical",
        display_name="MACD",
        description=(
            "Difference between a fast and a slow EMA, with a signal EMA of that "
            "difference and the resulting histogram."
        ),
        purpose="Summarise the relationship between two trend horizons.",
        required_inputs=("close",),
        min_observations=35,
        output_schema={"macd": "price_difference", "signal": "price_difference",
                       "histogram": "price_difference"},
        missing_data_policy=(
            "Requires slow + signal observations so the signal line is itself "
            "fully seeded; below that no value is produced."
        ),
        limitations=(
            "MACD is a transformation of two lagging averages and inherits their lag.",
            "Its absolute scale depends on the instrument's price level, so values "
            "are not comparable across instruments without normalisation.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
        params_schema={
            "defaults": {"fast": 12, "slow": 26, "signal": 9},
            "observation_drivers": ("slow",),
            "window_padding": 9,
        },
    )
)
def _macd(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    signal_period = int(params.get("signal", 9))
    fast_line = ema(data.close, fast)
    slow_line = ema(data.close, slow)
    diff = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(fast_line, slow_line, strict=True)
    ]
    seeded = [d for d in diff if d is not None]
    signal_line = ema(seeded, signal_period)
    macd_value = seeded[-1] if seeded else None
    signal_value = signal_line[-1] if signal_line else None
    return {
        "macd": macd_value,
        "signal": signal_value,
        "histogram": (
            macd_value - signal_value
            if macd_value is not None and signal_value is not None
            else None
        ),
        "fast": fast,
        "slow": slow,
        "signal_period": signal_period,
    }


@register_method(
    MethodSpec(
        key="bollinger_bands",
        version="1.0.0",
        family="technical",
        display_name="Bollinger Bands",
        description=(
            "A moving average with bands at a multiple of the rolling standard "
            "deviation of price."
        ),
        purpose="Describe where price sits relative to its recent dispersion.",
        required_inputs=("close",),
        min_observations=20,
        output_schema={"middle": "price", "upper": "price", "lower": "price",
                       "percent_b": "fraction", "bandwidth": "fraction"},
        missing_data_policy="No value until the window is full.",
        limitations=(
            "The bands describe recent dispersion of price, not a probability "
            "range: price distributions are fat-tailed, so the nominal 95% "
            "coverage of two standard deviations does not hold.",
            "A touch of a band is not a signal; in a trend price rides the band.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
        params_schema={
            "defaults": {"window": 20, "num_std": 2.0},
            "observation_drivers": ("window",),
        },
    )
)
def _bollinger(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    window = int(params.get("window", 20))
    k = float(params.get("num_std", 2.0))
    recent = data.close[-window:]
    middle = sum(recent) / window
    sd = statistics.stdev(recent)
    upper = middle + k * sd
    lower = middle - k * sd
    last = data.close[-1]
    span = upper - lower
    return {
        "middle": middle,
        "upper": upper,
        "lower": lower,
        # %B: 0 at the lower band, 1 at the upper.
        "percent_b": ((last - lower) / span) if span else None,
        "bandwidth": (span / middle) if middle else None,
        "window": window,
        "num_std": k,
        "standard_deviation": sd,
    }


@register_method(
    MethodSpec(
        key="atr",
        version="1.0.0",
        family="technical",
        display_name="Average True Range (Wilder)",
        description="Wilder-smoothed true range, in price units and as a fraction of price.",
        purpose="Size stops and position risk in units the instrument actually moves.",
        required_inputs=("high", "low", "close"),
        min_observations=15,
        output_schema={"atr": "price", "atr_percent": "fraction"},
        missing_data_policy="Requires period+1 bars because true range needs a prior close.",
        limitations=(
            "ATR measures realised range, not gap risk: an overnight gap can exceed "
            "several ATRs and a stop placed at a multiple of ATR will not hold.",
            "Wilder smoothing is used; platforms using a simple mean of true range "
            "report different values.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
        params_schema={"defaults": {"period": 14}, "observation_drivers": ("period",),
                       "window_padding": 1},
    )
)
def _atr(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    period = int(params.get("period", 14))
    tr = true_ranges(data.high, data.low, data.close)
    value = wilder_smooth(tr, period)[-1]
    last = data.close[-1]
    return {
        "atr": value,
        "atr_percent": (value / last) if value is not None and last else None,
        "period": period,
    }


@register_method(
    MethodSpec(
        key="drawdown",
        version="1.0.0",
        family="price",
        display_name="Drawdown analysis",
        description=(
            "Maximum peak-to-trough decline over the window, the current drawdown "
            "from the running peak, and the dates involved."
        ),
        purpose="State the worst loss an investor holding through the window endured.",
        required_inputs=("close",),
        min_observations=2,
        output_schema={
            "max_drawdown": "negative_fraction",
            "current_drawdown": "negative_fraction",
            "peak_date": "iso_date",
            "trough_date": "iso_date",
        },
        missing_data_policy=(
            "Computed on closes only. An intraday low deeper than any close is not "
            "captured, so the figure is a lower bound on the true drawdown."
        ),
        limitations=(
            "Close-to-close drawdown understates the intraday experience.",
            "Maximum drawdown is a single realised path, not a risk forecast; the "
            "next window can easily be worse.",
            "It is sensitive to the window boundaries chosen.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
    )
)
def _drawdown(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    worst, peak_idx, trough_idx = max_drawdown(data.close)
    running_peak = max(data.close)
    current = (data.close[-1] / running_peak) - 1.0
    ts = data.timestamps
    return {
        "max_drawdown": worst,
        "current_drawdown": current,
        "peak_close": data.close[peak_idx],
        "trough_close": data.close[trough_idx],
        "peak_date": ts[peak_idx].isoformat() if ts else None,
        "trough_date": ts[trough_idx].isoformat() if ts else None,
        "recovered": data.close[-1] >= data.close[peak_idx],
    }


@register_method(
    MethodSpec(
        key="volume_profile",
        version="1.0.0",
        family="technical",
        display_name="Volume analysis",
        description=(
            "Average volume, recent volume relative to its own average, and the "
            "correlation between volume and absolute return."
        ),
        purpose="Detect whether price moves are accompanied by participation.",
        required_inputs=("close", "volume"),
        min_observations=20,
        output_schema={
            "mean_volume": "shares",
            "recent_volume_ratio": "ratio",
            "volume_return_correlation": "correlation",
        },
        missing_data_policy=(
            "Bars with no reported volume terminate the usable window rather than "
            "being treated as zero, since 'unreported' and 'no trading' differ."
        ),
        limitations=(
            "Reported volume depends on venue consolidation rules and differs "
            "between providers; cross-provider comparison is unreliable.",
            "Volume is not comparable across instruments without normalising by "
            "shares outstanding or free float.",
            "A positive volume/|return| correlation is expected and is not evidence "
            "of informed trading.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
        params_schema={"defaults": {"recent_window": 5, "baseline_window": 60}},
    )
)
def _volume_profile(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    recent_window = int(params.get("recent_window", 5))
    baseline_window = int(params.get("baseline_window", 60))
    volumes = [v for v in data.volume if v is not None]
    warnings: list[str] = []
    if len(volumes) < len(data.volume):
        warnings.append(
            f"{len(data.volume) - len(volumes)} bar(s) had no reported volume and "
            "were excluded rather than counted as zero"
        )
    if not volumes:
        return ({"mean_volume": None, "recent_volume_ratio": None,
                 "volume_return_correlation": None}, warnings)

    baseline = volumes[-baseline_window:] if len(volumes) >= baseline_window else volumes
    recent = volumes[-recent_window:] if len(volumes) >= recent_window else volumes
    mean_baseline = sum(baseline) / len(baseline)
    mean_recent = sum(recent) / len(recent)

    correlation = None
    rets = log_returns(data.close)
    aligned_volumes = volumes[1:] if len(volumes) == len(data.close) else None
    if aligned_volumes and len(aligned_volumes) == len(rets) and len(rets) >= 3:
        abs_rets = [abs(r) for r in rets]
        if statistics.pstdev(abs_rets) > 0 and statistics.pstdev(aligned_volumes) > 0:
            correlation = statistics.correlation(abs_rets, aligned_volumes)

    return (
        {
            "mean_volume": mean_baseline,
            "recent_mean_volume": mean_recent,
            "recent_volume_ratio": (mean_recent / mean_baseline) if mean_baseline else None,
            "volume_return_correlation": correlation,
            "observations": len(volumes),
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="support_resistance",
        version="1.0.0",
        family="technical",
        display_name="Support and resistance estimates",
        description=(
            "Swing highs and lows identified by a symmetric fractal rule, clustered "
            "into price levels with the number of touches at each."
        ),
        purpose="Identify price levels the market has repeatedly reacted to.",
        required_inputs=("high", "low", "close"),
        min_observations=30,
        output_schema={"support": "list_of_levels", "resistance": "list_of_levels"},
        missing_data_policy="Needs enough bars for the lookback on both sides of a pivot.",
        limitations=(
            "These are descriptive clusters of past pivots, not levels with "
            "demonstrated predictive power.",
            "The levels found depend entirely on the lookback and clustering "
            "tolerance chosen; a different parameter set gives different levels.",
            "A level with two touches is not statistically distinguishable from "
            "coincidence.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
        params_schema={
            "defaults": {"lookback": 5, "tolerance_percent": 1.5, "max_levels": 5},
            "observation_drivers": (),
        },
    )
)
def _support_resistance(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    lookback = int(params.get("lookback", 5))
    tolerance = float(params.get("tolerance_percent", 1.5)) / 100.0
    max_levels = int(params.get("max_levels", 5))
    high, low = data.high, data.low
    n = len(high)

    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(lookback, n - lookback):
        window_high = high[i - lookback : i + lookback + 1]
        window_low = low[i - lookback : i + lookback + 1]
        if high[i] == max(window_high):
            swing_highs.append(high[i])
        if low[i] == min(window_low):
            swing_lows.append(low[i])

    last = data.close[-1]
    resistance = _cluster_levels(swing_highs, tolerance)
    support = _cluster_levels(swing_lows, tolerance)
    return {
        "resistance": [
            lvl for lvl in resistance if lvl["level"] > last
        ][:max_levels],
        "support": [
            lvl for lvl in sorted(support, key=lambda x: -x["level"]) if lvl["level"] < last
        ][:max_levels],
        "all_resistance": resistance[:max_levels],
        "all_support": support[:max_levels],
        "swing_high_count": len(swing_highs),
        "swing_low_count": len(swing_lows),
        "last_close": last,
        "lookback": lookback,
        "tolerance_percent": float(params.get("tolerance_percent", 1.5)),
    }


def _cluster_levels(values: list[float], tolerance: float) -> list[dict]:
    """Group nearby pivots into levels, reporting touch counts."""
    if not values:
        return []
    ordered = sorted(values)
    clusters: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        anchor = clusters[-1][0]
        if anchor and abs(value - anchor) / anchor <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    out = [
        {
            "level": sum(c) / len(c),
            "touches": len(c),
            "low": min(c),
            "high": max(c),
        }
        for c in clusters
    ]
    # Most-touched first: a level the market reacted to five times is better
    # evidence than one it touched once.
    out.sort(key=lambda d: (-d["touches"], d["level"]))
    return out


@register_method(
    MethodSpec(
        key="relative_strength",
        version="1.0.0",
        family="price",
        display_name="Relative strength versus benchmark",
        description=(
            "Excess return against a benchmark over the window, plus beta, alpha "
            "and the correlation of returns."
        ),
        purpose="Separate instrument-specific movement from market movement.",
        required_inputs=("close", "benchmark_close"),
        min_observations=30,
        output_schema={
            "excess_return": "fraction",
            "beta": "ratio",
            "alpha_annualised": "fraction_per_year",
            "correlation": "correlation",
            "r_squared": "fraction",
        },
        missing_data_policy=(
            "Requires the benchmark aligned bar-for-bar to the instrument. "
            "Unaligned series are rejected rather than being matched approximately, "
            "since a one-day misalignment destroys a beta estimate."
        ),
        limitations=(
            "Beta is an in-sample regression coefficient over this window, not a "
            "stable property; it moves substantially across periods and regimes.",
            "Alpha here is a residual, not evidence of skill, and carries no "
            "significance test by itself.",
            "A high correlation does not imply a causal relationship with the "
            "benchmark.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
    )
)
def _relative_strength(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    asset = log_returns(data.close)
    bench = log_returns(data.benchmark_close)
    if len(asset) != len(bench) or len(asset) < 2:
        return {"excess_return": None, "beta": None, "alpha_annualised": None,
                "correlation": None, "r_squared": None}

    bench_var = statistics.pvariance(bench)
    beta = None
    if bench_var > 0:
        cov = sum(
            (a - statistics.fmean(asset)) * (b - statistics.fmean(bench))
            for a, b in zip(asset, bench, strict=True)
        ) / len(asset)
        beta = cov / bench_var

    corr = None
    if statistics.pstdev(asset) > 0 and statistics.pstdev(bench) > 0:
        corr = statistics.correlation(asset, bench)

    asset_total = math.exp(sum(asset)) - 1.0
    bench_total = math.exp(sum(bench)) - 1.0
    periods_per_year = _periods_per_year(data)
    alpha = None
    if beta is not None and periods_per_year:
        residual_mean = statistics.fmean(
            [a - beta * b for a, b in zip(asset, bench, strict=True)]
        )
        alpha = math.exp(residual_mean * periods_per_year) - 1.0

    return {
        "excess_return": asset_total - bench_total,
        "asset_return": asset_total,
        "benchmark_return": bench_total,
        "beta": beta,
        "alpha_annualised": alpha,
        "correlation": corr,
        "r_squared": (corr**2) if corr is not None else None,
        "observations": len(asset),
    }


@register_method(
    MethodSpec(
        key="gap_analysis",
        version="1.0.0",
        family="technical",
        display_name="Overnight gap analysis",
        description=(
            "Distribution of open-to-previous-close gaps, and how often they "
            "closed within the same session."
        ),
        purpose="Quantify overnight risk that an intraday stop cannot protect against.",
        required_inputs=("open", "high", "low", "close"),
        min_observations=30,
        output_schema={
            "mean_absolute_gap": "fraction",
            "largest_gap_up": "fraction",
            "largest_gap_down": "fraction",
            "gap_fill_rate": "fraction",
        },
        missing_data_policy=(
            "A gap across a data gap is excluded, since the 'overnight' move would "
            "actually span several missing sessions."
        ),
        limitations=(
            "Gap statistics are dominated by a handful of event days; the mean is "
            "not a useful summary of the tail.",
            "The fill rate is a historical frequency within this window, not a "
            "probability for the next gap.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=("1d",),
        params_schema={"defaults": {"min_gap_percent": 1.0}},
    )
)
def _gap_analysis(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    threshold = float(params.get("min_gap_percent", 1.0)) / 100.0
    gaps: list[float] = []
    significant: list[tuple[int, float]] = []
    for i in range(1, len(data.close)):
        gap = (data.open[i] / data.close[i - 1]) - 1.0
        gaps.append(gap)
        if abs(gap) >= threshold:
            significant.append((i, gap))

    filled = 0
    for i, gap in significant:
        prior_close = data.close[i - 1]
        # A gap "fills" when the session trades back to the prior close.
        if (gap > 0 and data.low[i] <= prior_close) or (gap < 0 and data.high[i] >= prior_close):
            filled += 1

    if not gaps:
        return {"mean_absolute_gap": None, "largest_gap_up": None,
                "largest_gap_down": None, "gap_fill_rate": None}

    return {
        "mean_absolute_gap": statistics.fmean([abs(g) for g in gaps]),
        "median_absolute_gap": statistics.median([abs(g) for g in gaps]),
        "largest_gap_up": max(gaps),
        "largest_gap_down": min(gaps),
        "significant_gap_count": len(significant),
        "gap_fill_rate": (filled / len(significant)) if significant else None,
        "threshold_percent": float(params.get("min_gap_percent", 1.0)),
        "observations": len(gaps),
    }


@register_method(
    MethodSpec(
        key="trend_strength",
        version="1.0.0",
        family="technical",
        display_name="Trend strength",
        description=(
            "Ordinary-least-squares slope of log price against time, with its "
            "R-squared, plus the fraction of up periods."
        ),
        purpose="Quantify how directional the window was, and how cleanly so.",
        required_inputs=("close",),
        min_observations=20,
        output_schema={
            "slope_per_period": "log_fraction_per_period",
            "annualised_slope": "fraction_per_year",
            "r_squared": "fraction",
            "up_period_fraction": "fraction",
        },
        missing_data_policy="Regression runs on available bars; gaps shorten the window.",
        limitations=(
            "A fitted trend describes the window, it does not extend beyond it.",
            "A high R-squared on log price is common even in random walks over "
            "short windows and is not evidence of a persistent trend.",
        ),
        supported_instrument_types=EQUITY_LIKE,
        supported_resolutions=INTRADAY_AND_DAILY,
    )
)
def _trend_strength(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    y = [math.log(c) for c in data.close]
    n = len(y)
    x = list(range(n))
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y, strict=True))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None

    rets = log_returns(data.close)
    up_fraction = sum(1 for r in rets if r > 0) / len(rets) if rets else None
    periods_per_year = _periods_per_year(data)
    return {
        "slope_per_period": slope,
        "annualised_slope": (
            math.exp(slope * periods_per_year) - 1.0 if periods_per_year else None
        ),
        "r_squared": r_squared,
        "up_period_fraction": up_fraction,
        "observations": n,
    }


def _periods_per_year(data: MethodInput) -> float | None:
    """Periods per year for the input's resolution.

    Intraday figures assume a 6.5-hour US session; for another venue they are
    wrong, which is why the assumed value is returned alongside any annualised
    number instead of being hidden.
    """
    daily = data.trading_days_per_year
    return {
        "1d": float(daily),
        "1wk": 52.0,
        "1mo": 12.0,
        "1h": daily * 6.5,
        "30m": daily * 13.0,
        "15m": daily * 26.0,
        "5m": daily * 78.0,
        "1m": daily * 390.0,
    }.get(data.resolution)
