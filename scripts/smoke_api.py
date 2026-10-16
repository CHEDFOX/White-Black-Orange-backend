"""API smoke test against a running server with seeded data.

    PORT=8099 python -m scripts.smoke_api

Exercises every endpoint group and prints what came back, so a reviewer can
see the platform's actual behaviour rather than take a test suite's word for it.
"""

from __future__ import annotations

import json
import os
import sys

import httpx

BASE = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/api/v1"
client = httpx.Client(base_url=BASE, timeout=180.0, trust_env=False)

failures: list[str] = []


def call(method: str, path: str, **kwargs):
    try:
        response = client.request(method, path, **kwargs)
    except Exception as exc:
        failures.append(f"{method} {path}: {type(exc).__name__}: {exc}")
        return None
    if response.status_code >= 400:
        failures.append(f"{method} {path} -> {response.status_code}: {response.text[:300]}")
        return None
    return response.json()


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


heading("SEARCH")
data = call("GET", "/instruments/search", params={"q": "a", "only_with_prices": True})
if data:
    print(f"{data['returned']} results for 'a'")
    for r in data["results"]:
        print(
            f"  {r['symbol']:6s} {r['exchange_code']:6s} "
            f"{r['instrument_type']:8s} {r['name']}"
        )

heading("INSTRUMENT DETAIL: AAPL")
data = call("GET", "/instruments/AAPL")
if data:
    print(f"{data['symbol']} | {data['name']} | {data['exchange_name']} | {data['currency']}")
    print(f"sector={data['sector']} cik={data['cik']} tz={data['timezone']}")
    print(f"resolutions={data['supported_resolutions']} bases={data['supported_adjustment_bases']}")
    print(f"freshness: {data['freshness']['note']} (stale={data['freshness']['is_stale']})")
    for c in data["coverage"]:
        print(
            f"  coverage: {c['data_kind']:12s} {c['resolution']!s:4s} {c['source']:10s} "
            f"n={c['observation_count']:6d} "
            f"{(c['first_observation'] or '')[:10]} -> {(c['last_observation'] or '')[:10]}"
        )
    for caveat in data["analysis_caveats"]:
        print(f"  caveat: {caveat}")

heading("CANDLES (daily, total-return)")
data = call(
    "GET", "/market-data/AAPL/candles",
    params={"resolution": "1d", "adjustment_basis": "total_return",
            "start": "2024-01-01T00:00:00Z", "end": "2024-01-15T00:00:00Z"},
)
if data:
    print(f"{data['count']} candles, currency {data['currency']}")
    for c in data["candles"][:5]:
        print(
            f"  {c['t'][:10]}  O {c['o']:>10}  H {c['h']:>10}  "
            f"L {c['l']:>10}  C {c['c']:>10}  V {c['v']}"
        )
    print(f"attribution: {[a['key'] for a in data['attribution']]}")
    for a in data["attribution"]:
        if a["attribution_text"]:
            print(f"  '{a['attribution_text']}' official_api={a['official_api']} "
                  f"redistribution={a['redistribution_allowed']}")
    for d in data["disclosures"]:
        print(f"  disclosure: {d}")

heading("CANDLES aggregated to weekly from stored daily")
data = call(
    "GET", "/market-data/AAPL/candles",
    params={"resolution": "1wk", "start": "2024-01-01T00:00:00Z", "end": "2024-03-01T00:00:00Z"},
)
if data:
    print(f"{data['count']} weekly candles, aggregated_from={data['aggregated_from']}")
    for c in data["candles"][:4]:
        print(
            f"  {c['t'][:10]}  O {c['o']:>10}  H {c['h']:>10}  "
            f"L {c['l']:>10}  C {c['c']:>10}"
        )
    for d in data["disclosures"]:
        print(f"  disclosure: {d}")

heading("CANDLES at a resolution that cannot be built (minute from daily)")
data = call("GET", "/market-data/AAPL/candles", params={"resolution": "5m"})
if data:
    print(f"count={data['count']}")
    for d in data["disclosures"]:
        print(f"  disclosure: {d}")

heading("LATEST PRICE")
data = call("GET", "/market-data/AAPL/latest")
if data:
    print(f"{data['symbol']} {data['price']} {data['currency']} at {data['at']}")
    print(f"  note: {data['note']}")

heading("PAST ANALYSIS: 2022 drawdown year")
data = call(
    "GET", "/analysis/AAPL/past",
    params={"start": "2022-01-01T00:00:00Z", "end": "2022-12-31T00:00:00Z"},
)
if data:
    print(f"{data['symbol']} {data['period_start'][:10]} -> {data['period_end'][:10]} "
          f"({data['observations']} bars, cutoff {data['as_of'][:10]})")
    for key in ("returns", "realized_volatility", "drawdown", "regime_classification",
                "relative_strength"):
        m = data["methods"].get(key)
        if not m:
            continue
        if not m["ok"]:
            print(f"  {key:22s} UNAVAILABLE: {m['unavailable_reason']}")
            continue
        interesting = {
            k: v for k, v in m["values"].items()
            if isinstance(v, (int, float, str)) and k not in {"observations", "periods"}
        }
        print(f"  {key:22s} v{m['version']} n={m['observations_used']}")
        for k, v in list(interesting.items())[:5]:
            print(f"      {k} = {v}")
    print("\n  DISCLOSURES:")
    for d in data["disclosures"]:
        print(f"    - {d}")

heading("PRESENT ANALYSIS")
data = call("GET", "/analysis/AAPL/present", params={"lookback_days": 400})
if data:
    print(f"{data['observations']} bars to {data['period_end'][:10]}")
    for key in ("moving_averages", "rsi", "macd", "bollinger_bands", "atr",
                "support_resistance"):
        m = data["methods"].get(key)
        if not m or not m["ok"]:
            continue
        print(f"  {key}:")
        for k, v in list(m["values"].items())[:4]:
            print(f"      {k} = {json.dumps(v)[:120]}")

heading("FUNDAMENTALS (point-in-time)")
data = call("GET", "/analysis/AAPL/fundamentals")
if data:
    for key in ("profitability", "balance_sheet_strength", "cash_flow_quality",
                "valuation_ratios", "growth_trajectory"):
        m = data["methods"].get(key)
        if not m:
            continue
        if not m["ok"]:
            print(f"  {key:24s} UNAVAILABLE: {m['unavailable_reason']}")
            continue
        print(f"  {key}:")
        for k, v in m["values"].items():
            if isinstance(v, (int, float)) and v is not None:
                print(f"      {k} = {v:,.4f}" if abs(v) < 1e6 else f"      {k} = {v:,.0f}")
        for w in m["warnings"]:
            print(f"      warning: {w}")

heading("FUNDAMENTALS AS OF 2022 (restatement check)")
data = call("GET", "/analysis/AAPL/fundamentals", params={"as_of": "2022-06-30T00:00:00Z"})
if data:
    m = data["methods"].get("profitability")
    if m and m["ok"]:
        print(f"  revenue as known on 2022-06-30: {m['values'].get('revenue'):,.0f}")
        print(f"  net margin: {m['values'].get('net_margin')}")
    v = data["methods"].get("valuation_ratios")
    if v and v["ok"]:
        print(f"  P/E as of then: {v['values'].get('price_to_earnings')}")

heading("RISK")
data = call("GET", "/analysis/AAPL/risk")
if data:
    m = data["methods"].get("risk_profile")
    if m and m["ok"]:
        for k in ("annualised_volatility", "var_95", "var_99",
                  "expected_shortfall_95", "max_drawdown", "atr_percent"):
            print(f"  {k:26s} {m['values'].get(k)}")
        print(f"  gap_risk: {json.dumps(m['values'].get('gap_risk'))[:200]}")
        print(f"  disclaimer: {m['values'].get('disclaimer')[:120]}...")
    s = data["methods"].get("position_sizing")
    if s and s["ok"]:
        print(f"  volatility-target weight: {s['values'].get('volatility_target_weight')}")

heading("TRADE STRUCTURE")
data = call(
    "POST", "/analysis/AAPL/trade-structure",
    json={"direction": "long", "stop_atr_multiple": 2.0,
          "target_atr_multiples": [2, 4], "account_value": 100000},
)
if data:
    m = data["methods"].get("trade_structure")
    if m and m["ok"]:
        s = m["values"]["structure"]
        print(f"  entry {s['entry']:.2f} stop {s['stop']:.2f} "
              f"({s['stop_distance_percent']:.2%}, {s['stop_distance_atr']} ATR)")
        targets = [round(t, 2) for t in s["targets"]]
        ratios = [round(r, 2) for r in s["risk_reward_ratios"]]
        print(f"  targets {targets} R:R {ratios}")
        print(f"  position size {s['position_size']:.1f} shares at 1% risk")
        print(f"  stop_jump_probability {m['values'].get('stop_jump_probability')}")
        for w in m["warnings"]:
            print(f"  warning: {w}")

heading("BUBBLE ASSESSMENT")
data = call("GET", "/analysis/TSLA/bubble")
if data:
    m = data["methods"].get("bubble_assessment")
    if m and m["ok"]:
        v = m["values"]
        print(f"  vulnerability: {v['vulnerability_score']:.1f}/100 ({v['vulnerability_band']})")
        print(f"  triggered {v['signals_triggered']} of {v['signals_evaluated']} evaluated "
              f"({v['signals_total']} total)")
        for s in v["signals"]:
            mark = {"triggered": "YES", "not_triggered": " no", "unavailable": " --"}[s["status"]]
            print(f"    [{mark}] {s['display_name']}")
            if s["status"] == "triggered":
                print(f"           {s['evidence']}")
            elif s["status"] == "unavailable":
                print(f"           unavailable: {s['unavailable_reason']}")
        print(f"  {v['no_crash_prediction']}")
        for w in m["warnings"]:
            print(f"  warning: {w}")

heading("FORECAST MODELS")
data = call("GET", "/forecasts/models")
if data:
    for m in data:
        print(f"  {m['key']:18s} v{m['version']} baseline={m['is_baseline']!s:5s} "
              f"{m['lifecycle']:12s} horizons={','.join(m['supported_horizons'])}")

heading("FORECAST with walk-forward validation")
data = call("POST", "/forecasts/AAPL", json={"model_key": "random_walk", "horizon": "1mo",
                                             "include_validation": True})
if data:
    print(f"  {data['model_key']} v{data['model_version']} {data['horizon_label']}")
    print(f"  cutoff {data['information_cutoff'][:10]} resolves {data['resolves_at'][:10]}")
    print(f"  point estimate {data['point_estimate']} prob_positive {data['prob_positive']}")
    print(f"  base price {data['base_price']:.2f}")
    print("  price levels:")
    for q, p in sorted(data["price_levels"].items(), key=lambda kv: float(kv[0])):
        print(f"      q{q}: {p:.2f}")
    if data.get("validation", {}).get("available"):
        mt = data["validation"]["metrics"]
        print(f"  VALIDATION ({mt['n_forecasts']} out-of-sample forecasts):")
        print(f"      CRPS {mt['mean_crps']:.5f}  MAE {mt['mean_absolute_error']:.5f}")
        print(f"      calibration: {mt['calibration_verdict']}  PIT p={mt['pit_uniformity_p']}")
        print(f"      coverage: {mt['interval_coverage']}")
        for w in mt["warnings"]:
            print(f"      warning: {w}")
    print(f"  disclaimer: {data['disclaimer'][:150]}...")

heading("MODEL COMPARISON (does anything beat the random walk?)")
data = call("GET", "/forecasts/AAPL/compare", params={"horizon": "1mo", "lookback_days": 2000})
if data:
    rows = []
    for r in data["results"]:
        if not r.get("available"):
            print(f"  {r['model_key']:18s} unavailable: {r['reason'][:80]}")
            continue
        mt = r["metrics"]
        rows.append((r["model_key"], mt["n_forecasts"], mt["mean_crps"],
                     mt.get("skill_vs_baseline"), mt["calibration_verdict"]))
    rows.sort(key=lambda row: (row[3] if row[3] is not None else -1), reverse=True)
    print(f"  {'model':18s} {'n':>4s} {'CRPS':>9s} {'skill':>8s}  calibration")
    for key, n, crps, skill, cal in rows:
        skill_s = "baseline" if skill is None else f"{skill:+.4f}"
        print(f"  {key:18s} {n:4d} {crps:9.5f} {skill_s:>8s}  {cal}")
    print(f"\n  {data['note']}")

heading("HISTORICAL REPORT")
data = call(
    "GET", "/reports/AAPL/historical",
    params={"start": "2022-01-01T00:00:00Z", "end": "2022-12-31T00:00:00Z"},
)
if data:
    print(f"  {data['title']}")
    print(f"  {data['summary']}")
    print(f"  reproduction key: {data['reproduction_key'][:32]}...")
    print(f"  claim kinds: {data['claim_kind_counts']}")
    print(f"  narrative: {data['narrative_status']}")
    print(f"\n  FINDINGS ({len(data['findings'])}):")
    for f in data["findings"]:
        print(f"\n    [{f['claim_kind']}] ({f['section']})")
        print(f"    {f['conclusion']}")
        if f.get("uncertainty"):
            print(f"      uncertainty: {f['uncertainty'][:160]}")
        for c in f["calculations"][:3]:
            method = f"{c['method_key']}@{c['method_version']}"
            print(f"      calc: {c['label']} = {c['value']} [{method}]")
        for e in f["evidence"][:2]:
            print(f"      {e['stance']}: {e['description'][:140]}")
    print(f"\n  LIMITATIONS ({len(data['limitations'])}):")
    for lim in data["limitations"]:
        print(f"    - {lim}")

heading("DATA SOURCES")
data = call("GET", "/data-sources")
if data:
    for s in data["sources"]:
        flags = (
            f"active={s['active']!s:5s} "
            f"pit={s.get('supports_point_in_time')!s:5s} "
            f"official={s['official_api']!s:5s} "
            f"redistribute={s['redistribution_allowed']!s:5s}"
        )
        print(f"  {s['key']:14s} {flags}")
        if s.get("unavailable_reason"):
            print(f"      unavailable: {s['unavailable_reason']}")
        for c in (s.get("caveats") or [])[:2]:
            if c:
                print(f"      caveat: {c[:120]}")

heading("METHODS REGISTRY")
data = call("GET", "/methods")
if data:
    print(f"  {data['count']} methods: {data['families']}")

heading("LIMITATIONS")
data = call("GET", "/limitations")
if data:
    for group, items in data.items():
        print(f"  {group}:")
        for item in items:
            print(f"    - {item}")

heading("RESULT")
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("all endpoints responded successfully")
