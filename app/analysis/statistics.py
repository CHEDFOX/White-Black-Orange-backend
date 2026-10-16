"""Statistical and econometric methods.

Everything here reports uncertainty alongside its estimate, and several methods
exist specifically to push back on over-reading a result:

* ``autocorrelation`` reports confidence bands so a reader can see whether a
  lag-1 coefficient of 0.05 is distinguishable from zero (on 60 observations it
  is not).
* ``multiple_testing`` exists because a p-value from the fiftieth variant of a
  strategy is not a p-value. It is applied to any family of tests the platform
  runs, and the family size is recorded.
* ``stationarity`` and ``change_point`` exist to detect when a single estimate
  over a long window is describing two different regimes averaged together.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from app.analysis.registry import MethodInput, MethodSpec, register_method
from app.analysis.technical import log_returns
from app.db.models.enums import ValidationStatus

ALL_RESOLUTIONS = ("1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")


@register_method(
    MethodSpec(
        key="return_distribution",
        version="1.0.0",
        family="statistics",
        display_name="Return distribution",
        description=(
            "Moments and quantiles of the log-return distribution, with a "
            "normality test and tail statistics."
        ),
        purpose=(
            "Describe the shape of the return distribution, because risk measures "
            "that assume normality understate tail losses in most instruments."
        ),
        required_inputs=("close",),
        min_observations=30,
        output_schema={
            "mean": "log_fraction",
            "std_dev": "log_fraction",
            "skewness": "dimensionless",
            "excess_kurtosis": "dimensionless",
            "jarque_bera_p": "p_value",
            "quantiles": "map_q_to_log_fraction",
        },
        missing_data_policy=(
            "Uses all available returns. Gaps mean some returns span more than "
            "one period, which inflates their magnitude; the gap count should be "
            "read alongside."
        ),
        limitations=(
            "Skewness and excess kurtosis need several hundred observations to be "
            "estimated with any precision; on 30 bars they are nearly meaningless.",
            "A Jarque-Bera rejection of normality says the distribution is not "
            "normal, not what it is instead.",
            "Empirical quantiles cannot describe the tail beyond the observed "
            "sample: a 1-in-500 loss is invisible in 250 observations.",
        ),
        supported_resolutions=ALL_RESOLUTIONS,
        params_schema={"defaults": {"quantiles": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]}},
    )
)
def _return_distribution(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    rets = np.asarray(log_returns(data.close), dtype=float)
    qs = [float(q) for q in params.get("quantiles", [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])]
    warnings: list[str] = []
    n = len(rets)

    if n < 200:
        warnings.append(
            f"skewness and excess kurtosis are estimated from {n} observations; "
            "both need several hundred to be meaningful and should be treated as "
            "indicative only"
        )

    jb_p = None
    if n >= 8:
        # Jarque-Bera is only valid asymptotically; below ~8 it is nonsense.
        _, jb_p = scipy_stats.jarque_bera(rets)

    return (
        {
            "observations": n,
            "mean": float(np.mean(rets)),
            "std_dev": float(np.std(rets, ddof=1)),
            "skewness": float(scipy_stats.skew(rets, bias=False)),
            "excess_kurtosis": float(scipy_stats.kurtosis(rets, fisher=True, bias=False)),
            "jarque_bera_p": float(jb_p) if jb_p is not None else None,
            "normality_rejected_at_5pct": (
                bool(jb_p < 0.05) if jb_p is not None else None
            ),
            "quantiles": {str(q): float(np.quantile(rets, q)) for q in qs},
            "min": float(np.min(rets)),
            "max": float(np.max(rets)),
            "positive_fraction": float(np.mean(rets > 0)),
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="autocorrelation",
        version="1.0.0",
        family="statistics",
        display_name="Return autocorrelation",
        description=(
            "Autocorrelation of returns at several lags with approximate 95% "
            "confidence bands, plus a Ljung-Box test for joint significance."
        ),
        purpose=(
            "Test whether past returns carry information about the next one -- the "
            "assumption behind every momentum and mean-reversion claim."
        ),
        required_inputs=("close",),
        min_observations=50,
        output_schema={
            "coefficients": "map_lag_to_correlation",
            "confidence_band": "correlation",
            "ljung_box_p": "p_value",
            "significant_lags": "list_of_lags",
        },
        missing_data_policy=(
            "Gaps break the lag structure; results over gapped windows are unreliable."
        ),
        limitations=(
            "The confidence band is the white-noise approximation 1.96/sqrt(n); it "
            "is too narrow when volatility clusters, which it almost always does.",
            "Autocorrelation in returns is typically economically tiny even when "
            "statistically detectable, and transaction costs usually exceed it.",
            "Testing many lags and reporting the significant ones is a multiple "
            "testing problem; use the Ljung-Box joint test for a verdict.",
        ),
        supported_resolutions=ALL_RESOLUTIONS,
        params_schema={"defaults": {"max_lag": 10}, "observation_drivers": ("max_lag",),
                       "window_padding": 40},
    )
)
def _autocorrelation(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    rets = np.asarray(log_returns(data.close), dtype=float)
    max_lag = min(int(params.get("max_lag", 10)), len(rets) // 4)
    n = len(rets)
    mean = float(np.mean(rets))
    centred = rets - mean
    denom = float(np.sum(centred**2))

    coefficients: dict[str, float] = {}
    for lag in range(1, max_lag + 1):
        num = float(np.sum(centred[lag:] * centred[:-lag]))
        coefficients[str(lag)] = (num / denom) if denom > 0 else 0.0

    band = 1.96 / math.sqrt(n)
    significant = [int(k) for k, v in coefficients.items() if abs(v) > band]

    # Ljung-Box: one joint test instead of max_lag separate ones.
    q_stat = n * (n + 2) * sum(
        (coefficients[str(lag)] ** 2) / (n - lag) for lag in range(1, max_lag + 1)
    )
    lb_p = float(scipy_stats.chi2.sf(q_stat, df=max_lag))

    return {
        "coefficients": coefficients,
        "confidence_band": band,
        "significant_lags": significant,
        "ljung_box_statistic": q_stat,
        "ljung_box_p": lb_p,
        "joint_independence_rejected_at_5pct": lb_p < 0.05,
        "observations": n,
        "max_lag": max_lag,
    }


@register_method(
    MethodSpec(
        key="stationarity",
        version="1.0.0",
        family="statistics",
        display_name="Stationarity (augmented Dickey-Fuller)",
        description=(
            "ADF test on log price and on returns, to check whether a series has "
            "a unit root."
        ),
        purpose=(
            "Decide whether a statistic estimated over the window describes one "
            "process or several averaged together."
        ),
        required_inputs=("close",),
        min_observations=60,
        output_schema={
            "price_adf_p": "p_value",
            "returns_adf_p": "p_value",
            "price_has_unit_root": "boolean",
        },
        missing_data_policy="Needs a contiguous window; gaps distort the lag structure.",
        limitations=(
            "Failing to reject a unit root is not evidence that one exists; the ADF "
            "test has low power, especially on short samples.",
            "A finding of non-stationarity in price is expected and uninformative; "
            "the interesting case is non-stationarity in returns.",
            "The test assumes a specific alternative and is sensitive to the lag "
            "order chosen.",
        ),
        supported_resolutions=("1d", "1wk", "1mo"),
        validation_status=ValidationStatus.UNIT_TESTED,
        cost_class="moderate",
    )
)
def _stationarity(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    from statsmodels.tsa.stattools import adfuller

    log_price = np.log(np.asarray(data.close, dtype=float))
    rets = np.asarray(log_returns(data.close), dtype=float)
    warnings: list[str] = []

    try:
        price_result = adfuller(log_price, autolag="AIC", result_object=False)
        price_p = float(price_result[1])
    except Exception as exc:  # pragma: no cover - numerical edge cases
        price_p = None
        warnings.append(f"ADF on log price could not be computed: {exc}")

    try:
        returns_result = adfuller(rets, autolag="AIC", result_object=False)
        returns_p = float(returns_result[1])
    except Exception as exc:  # pragma: no cover
        returns_p = None
        warnings.append(f"ADF on returns could not be computed: {exc}")

    return (
        {
            "price_adf_p": price_p,
            "returns_adf_p": returns_p,
            "price_has_unit_root": (price_p > 0.05) if price_p is not None else None,
            "returns_stationary": (returns_p < 0.05) if returns_p is not None else None,
            "observations": len(data.close),
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="volatility_clustering",
        version="1.0.0",
        family="statistics",
        display_name="Volatility clustering (ARCH effects)",
        description=(
            "Engle's ARCH LM test on squared returns, plus the autocorrelation of "
            "absolute returns."
        ),
        purpose=(
            "Detect whether volatility is predictable even when returns are not, "
            "which determines whether a constant-volatility risk model is adequate."
        ),
        required_inputs=("close",),
        min_observations=60,
        output_schema={"arch_lm_p": "p_value", "abs_return_autocorr_lag1": "correlation"},
        missing_data_policy="Requires a contiguous window.",
        limitations=(
            "Detecting ARCH effects says volatility is predictable in-sample; it "
            "does not establish that a GARCH forecast would beat a rolling "
            "standard deviation out of sample.",
            "The test is sensitive to outliers, and a single crash day can drive it.",
        ),
        supported_resolutions=("1d", "1h", "1wk"),
        cost_class="moderate",
        params_schema={"defaults": {"lags": 5}},
    )
)
def _volatility_clustering(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    from statsmodels.stats.diagnostic import het_arch

    rets = np.asarray(log_returns(data.close), dtype=float)
    lags = int(params.get("lags", 5))
    warnings: list[str] = []
    arch_p = None
    try:
        _, arch_p, _, _ = het_arch(rets, nlags=lags, result_object=False)
        arch_p = float(arch_p)
    except Exception as exc:  # pragma: no cover
        warnings.append(f"ARCH LM test could not be computed: {exc}")

    abs_rets = np.abs(rets)
    centred = abs_rets - abs_rets.mean()
    denom = float(np.sum(centred**2))
    lag1 = float(np.sum(centred[1:] * centred[:-1]) / denom) if denom > 0 else None

    return (
        {
            "arch_lm_p": arch_p,
            "arch_effects_present_at_5pct": (arch_p < 0.05) if arch_p is not None else None,
            "abs_return_autocorr_lag1": lag1,
            "lags": lags,
            "observations": len(rets),
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="change_point",
        version="1.0.0",
        family="statistics",
        display_name="Variance change-point detection",
        description=(
            "Scans for the split point that maximises the difference in return "
            "variance between the two halves, with an F-test on the split."
        ),
        purpose=(
            "Find where a window's behaviour changed, so a single statistic is not "
            "reported for two different regimes."
        ),
        required_inputs=("close",),
        min_observations=60,
        output_schema={
            "change_point_index": "index",
            "change_point_date": "iso_datetime",
            "variance_ratio": "ratio",
            "f_test_p": "p_value",
        },
        missing_data_policy="Requires a contiguous window.",
        limitations=(
            "The p-value is NOT corrected for the search over every candidate "
            "split point, so it overstates significance substantially. Treat it as "
            "a ranking score, not a hypothesis test.",
            "The method finds the single best split; a window with three regimes "
            "will be mis-described.",
            "An F-test on variance assumes normal returns, which they are not.",
        ),
        supported_resolutions=("1d", "1wk"),
        params_schema={"defaults": {"min_segment": 20}},
    )
)
def _change_point(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    rets = np.asarray(log_returns(data.close), dtype=float)
    min_segment = int(params.get("min_segment", 20))
    n = len(rets)
    if n < 2 * min_segment:
        return {
            "change_point_index": None,
            "change_point_date": None,
            "variance_ratio": None,
            "f_test_p": None,
        }

    best_idx = None
    best_ratio = 1.0
    for i in range(min_segment, n - min_segment):
        left = rets[:i]
        right = rets[i:]
        var_left = float(np.var(left, ddof=1))
        var_right = float(np.var(right, ddof=1))
        if var_left <= 0 or var_right <= 0:
            continue
        ratio = max(var_right / var_left, var_left / var_right)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_idx is None:
        return {
            "change_point_index": None,
            "change_point_date": None,
            "variance_ratio": None,
            "f_test_p": None,
        }

    left, right = rets[:best_idx], rets[best_idx:]
    var_left, var_right = float(np.var(left, ddof=1)), float(np.var(right, ddof=1))
    if var_right >= var_left:
        f_stat, df1, df2 = var_right / var_left, len(right) - 1, len(left) - 1
    else:
        f_stat, df1, df2 = var_left / var_right, len(left) - 1, len(right) - 1
    p_uncorrected = float(2 * scipy_stats.f.sf(f_stat, df1, df2))

    # Bonferroni over the candidate splits actually searched. Still optimistic,
    # but far closer to honest than the raw value.
    searched = n - 2 * min_segment
    p_corrected = min(1.0, p_uncorrected * max(searched, 1))

    ts = data.timestamps
    # +1 because returns are offset one bar from closes.
    date_idx = min(best_idx + 1, len(ts) - 1) if ts else None

    return (
        {
            "change_point_index": best_idx,
            "change_point_date": ts[date_idx].isoformat() if date_idx is not None else None,
            "variance_ratio": best_ratio,
            "volatility_before": math.sqrt(var_left),
            "volatility_after": math.sqrt(var_right),
            "f_test_p": p_uncorrected,
            "f_test_p_bonferroni": p_corrected,
            "candidate_splits_searched": searched,
            "observations": n,
        },
        [
            f"the reported p-value searched {searched} candidate split points; the "
            "Bonferroni-corrected value is the one to read"
        ],
    )


@register_method(
    MethodSpec(
        key="extreme_value",
        version="1.0.0",
        family="statistics",
        display_name="Tail risk (historical and Cornish-Fisher)",
        description=(
            "Historical and parametric value-at-risk and expected shortfall, with "
            "a Cornish-Fisher adjustment for skew and kurtosis."
        ),
        purpose="Estimate loss magnitudes in the tail, where a normal model fails.",
        required_inputs=("close",),
        min_observations=100,
        output_schema={
            "historical_var": "map_confidence_to_log_fraction",
            "historical_expected_shortfall": "map_confidence_to_log_fraction",
            "normal_var": "map_confidence_to_log_fraction",
            "cornish_fisher_var": "map_confidence_to_log_fraction",
        },
        missing_data_policy="Requires 100 returns; below that tail estimates are not reported.",
        limitations=(
            "Historical VaR cannot exceed the worst observed loss, so it "
            "systematically understates risk in a quiet sample.",
            "A normal VaR understates tail loss for essentially every equity.",
            "The Cornish-Fisher expansion is unreliable for large skew or kurtosis "
            "and can produce non-monotonic quantiles.",
            "All three are single-period measures and do not aggregate over time.",
        ),
        supported_resolutions=("1d", "1wk"),
        params_schema={"defaults": {"confidences": [0.95, 0.99]}},
    )
)
def _extreme_value(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    rets = np.asarray(log_returns(data.close), dtype=float)
    confidences = [float(c) for c in params.get("confidences", [0.95, 0.99])]
    mean = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    skew = float(scipy_stats.skew(rets, bias=False))
    kurt = float(scipy_stats.kurtosis(rets, fisher=True, bias=False))

    hist_var: dict[str, float] = {}
    hist_es: dict[str, float] = {}
    normal_var: dict[str, float] = {}
    cf_var: dict[str, float] = {}
    warnings: list[str] = []

    for c in confidences:
        alpha = 1.0 - c
        q = float(np.quantile(rets, alpha))
        hist_var[str(c)] = q
        tail = rets[rets <= q]
        hist_es[str(c)] = float(np.mean(tail)) if tail.size else q

        z = float(scipy_stats.norm.ppf(alpha))
        normal_var[str(c)] = mean + z * sd
        # Cornish-Fisher quantile expansion.
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * (skew**2) / 36
        )
        cf_var[str(c)] = mean + z_cf * sd

    if abs(skew) > 2 or kurt > 10:
        warnings.append(
            f"skew {skew:.2f} and excess kurtosis {kurt:.2f} are outside the range "
            "where the Cornish-Fisher expansion is reliable; prefer the historical "
            "figures"
        )
    warnings.append(
        f"historical VaR is bounded by the worst observed return in this sample "
        f"({float(np.min(rets)):.4f}); it cannot describe a loss larger than any seen"
    )

    return (
        {
            "historical_var": hist_var,
            "historical_expected_shortfall": hist_es,
            "normal_var": normal_var,
            "cornish_fisher_var": cf_var,
            "worst_observed_return": float(np.min(rets)),
            "skewness": skew,
            "excess_kurtosis": kurt,
            "observations": len(rets),
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="correlation_matrix",
        version="1.0.0",
        family="statistics",
        display_name="Correlation and dependence",
        description=(
            "Pearson and Spearman correlation between the instrument and each "
            "auxiliary series, with significance and tail-dependence estimates."
        ),
        purpose=(
            "Measure co-movement, including whether it strengthens in "
            "drawdowns -- which is when diversification is actually needed."
        ),
        required_inputs=("close",),
        min_observations=40,
        output_schema={
            "pearson": "map_series_to_correlation",
            "spearman": "map_series_to_correlation",
            "downside_correlation": "map_series_to_correlation",
        },
        missing_data_policy=(
            "Each auxiliary series must be aligned to the instrument's bars; "
            "mismatched lengths are skipped and named rather than truncated."
        ),
        limitations=(
            "Correlation is not causation, and a correlated pair can be driven by "
            "a third factor neither series represents.",
            "Pearson correlation measures linear dependence only.",
            "Correlation is unstable over time and typically rises in market "
            "stress, so an average figure understates joint-loss risk.",
        ),
        supported_resolutions=("1d", "1wk", "1mo"),
        params_schema={"defaults": {"downside_quantile": 0.25}},
    )
)
def _correlation_matrix(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    asset = np.asarray(log_returns(data.close), dtype=float)
    q = float(params.get("downside_quantile", 0.25))
    pearson: dict[str, float | None] = {}
    spearman: dict[str, float | None] = {}
    downside: dict[str, float | None] = {}
    p_values: dict[str, float | None] = {}
    warnings: list[str] = []

    candidates = dict(data.series)
    if data.benchmark_close:
        candidates["benchmark"] = data.benchmark_close

    for name, series in candidates.items():
        if len(series) != len(data.close):
            warnings.append(
                f"series {name!r} has {len(series)} points against "
                f"{len(data.close)} bars and was skipped rather than truncated"
            )
            pearson[name] = spearman[name] = downside[name] = None
            continue
        other = np.asarray(log_returns(series), dtype=float)
        if other.size != asset.size or asset.size < 3:
            pearson[name] = spearman[name] = downside[name] = None
            continue
        if np.std(other) == 0 or np.std(asset) == 0:
            pearson[name] = spearman[name] = downside[name] = None
            warnings.append(f"series {name!r} has zero variance; correlation undefined")
            continue

        r, p = scipy_stats.pearsonr(asset, other)
        rho, _ = scipy_stats.spearmanr(asset, other)
        pearson[name] = float(r)
        spearman[name] = float(rho)
        p_values[name] = float(p)

        # Correlation conditional on the asset being in its worst quantile.
        threshold = float(np.quantile(asset, q))
        mask = asset <= threshold
        if mask.sum() >= 5 and np.std(other[mask]) > 0 and np.std(asset[mask]) > 0:
            dr, _ = scipy_stats.pearsonr(asset[mask], other[mask])
            downside[name] = float(dr)
        else:
            downside[name] = None

    return (
        {
            "pearson": pearson,
            "spearman": spearman,
            "downside_correlation": downside,
            "p_values": p_values,
            "downside_quantile": q,
            "observations": int(asset.size),
        },
        warnings,
    )


@register_method(
    MethodSpec(
        key="multiple_testing",
        version="1.0.0",
        family="statistics",
        display_name="Multiple-testing correction",
        description=(
            "Applies Bonferroni and Benjamini-Hochberg corrections to a family of "
            "p-values."
        ),
        purpose=(
            "Keep a reported discovery honest. A p-value of 0.03 from the "
            "fiftieth variant tested is not evidence of anything."
        ),
        required_inputs=("scalar:n_tests",),
        min_observations=0,
        output_schema={
            "bonferroni": "list_of_p_values",
            "benjamini_hochberg": "list_of_p_values",
            "survives_bonferroni": "list_of_boolean",
        },
        missing_data_policy="Operates on supplied p-values; no market data required.",
        limitations=(
            "Bonferroni is conservative and loses power as the family grows.",
            "Benjamini-Hochberg controls the false discovery rate, not the "
            "family-wise error rate; they answer different questions.",
            "Neither corrects for tests run and never recorded, which is the "
            "larger problem in practice -- hence the experiment log.",
        ),
        supported_resolutions=ALL_RESOLUTIONS,
        params_schema={"defaults": {"alpha": 0.05, "p_values": []}},
    )
)
def _multiple_testing(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    p_values = [float(p) for p in params.get("p_values", [])]
    alpha = float(params.get("alpha", 0.05))
    n_tests = int(data.scalars.get("n_tests", len(p_values)))

    if not p_values:
        return {
            "bonferroni": [],
            "benjamini_hochberg": [],
            "survives_bonferroni": [],
            "survives_bh": [],
            "n_tests": n_tests,
            "alpha": alpha,
        }

    bonferroni = [min(1.0, p * n_tests) for p in p_values]

    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    bh = [0.0] * len(p_values)
    running_min = 1.0
    for rank in range(len(order) - 1, -1, -1):
        i = order[rank]
        adjusted = p_values[i] * n_tests / (rank + 1)
        running_min = min(running_min, adjusted, 1.0)
        bh[i] = running_min

    return {
        "bonferroni": bonferroni,
        "benjamini_hochberg": bh,
        "survives_bonferroni": [p <= alpha for p in bonferroni],
        "survives_bh": [p <= alpha for p in bh],
        "n_tests": n_tests,
        "alpha": alpha,
        "raw_p_values": p_values,
    }


@register_method(
    MethodSpec(
        key="seasonality",
        version="1.0.0",
        family="statistics",
        display_name="Calendar seasonality",
        description=(
            "Mean return by day of week and by month, with sample sizes and a "
            "multiple-testing-corrected significance test."
        ),
        purpose="Check whether calendar effects are present, and whether they survive correction.",
        required_inputs=("close",),
        min_observations=250,
        output_schema={
            "by_weekday": "map_to_mean_and_n",
            "by_month": "map_to_mean_and_n",
            "any_survives_correction": "boolean",
        },
        missing_data_policy=(
            "Needs at least a year of daily bars for monthly buckets to be populated."
        ),
        limitations=(
            "Calendar anomalies are the canonical data-mining trap: with 12 months "
            "and 5 weekdays, something will look significant by chance.",
            "Reported p-values are Bonferroni-corrected across all buckets tested; "
            "the uncorrected values are not reportable as evidence.",
            "Published seasonal effects have largely failed to persist "
            "out-of-sample after discovery.",
        ),
        supported_resolutions=("1d",),
    )
)
def _seasonality(data: MethodInput, params: dict[str, Any]) -> dict[str, Any]:
    rets = log_returns(data.close)
    ts = data.timestamps[1:]
    by_weekday: dict[int, list[float]] = {}
    by_month: dict[int, list[float]] = {}
    for t, r in zip(ts, rets, strict=True):
        by_weekday.setdefault(t.weekday(), []).append(r)
        by_month.setdefault(t.month, []).append(r)

    def summarise(buckets: dict[int, list[float]], labels: dict[int, str]) -> dict:
        out = {}
        for key, values in sorted(buckets.items()):
            if len(values) < 5:
                out[labels[key]] = {"mean": None, "n": len(values), "p_value": None}
                continue
            t_stat, p = scipy_stats.ttest_1samp(values, 0.0)
            out[labels[key]] = {
                "mean": statistics.fmean(values),
                "n": len(values),
                "t_statistic": float(t_stat),
                "p_value": float(p),
            }
        return out

    weekday_labels = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                      4: "Friday", 5: "Saturday", 6: "Sunday"}
    month_labels = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November",
        12: "December",
    }
    weekday = summarise(by_weekday, weekday_labels)
    month = summarise(by_month, month_labels)

    raw_p = [
        v["p_value"]
        for v in list(weekday.values()) + list(month.values())
        if v["p_value"] is not None
    ]
    n_tests = len(raw_p)
    corrected = [min(1.0, p * n_tests) for p in raw_p]
    for bucket in (weekday, month):
        for v in bucket.values():
            if v["p_value"] is not None:
                v["p_value_bonferroni"] = min(1.0, v["p_value"] * n_tests)

    return (
        {
            "by_weekday": weekday,
            "by_month": month,
            "tests_performed": n_tests,
            "min_corrected_p": min(corrected) if corrected else None,
            "any_survives_correction": bool(corrected and min(corrected) <= 0.05),
            "observations": len(rets),
        },
        [
            f"{n_tests} calendar buckets were tested; only the Bonferroni-corrected "
            "p-values are reportable as evidence"
        ],
    )
