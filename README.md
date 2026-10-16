# Stock Intelligence Platform — Backend

An evidence-driven stock intelligence and trader guidance API.

**Operating principle:** data before intelligence, validation before
prediction, evidence before recommendations.

Three conventions hold across every endpoint, and they are what the code is
organised around:

- **Every conclusion carries an epistemic label.** A computed return is a
  `deterministic_calculation`; a regime label is a `model_estimate`; a question
  the data cannot answer is `unknown` and is returned rather than omitted.
- **Missing data is reported, never defaulted.** Nothing is interpolated,
  forward-filled or coerced to zero. A method that lacks the data to run
  returns the reason it could not.
- **Analysis carries its own limitations.** Responses include `disclosures` or
  `limitations`; a client that shows a figure can show these too.

> Research and educational tooling. Not personalised investment advice, not a
> recommendation to buy or sell, and no trade execution. See
> [docs/LIMITATIONS.md](docs/LIMITATIONS.md) and `GET /api/v1/limitations`.

## Quick start

No signup required — the defaults use real data.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

createdb sip
alembic upgrade head
python -m scripts.seed              # real prices, fundamentals, macro
./scripts/start_api.sh              # http://localhost:8000/docs
```

Or with Docker:

```bash
docker compose up --build
docker compose run --rm api python -m scripts.seed
```

Set `SEC_USER_AGENT` to something with your contact details — the SEC throttles
anonymous clients.

## What the seed loads

Real data from four live sources, verified working:

```
5 instruments      AAPL, MSFT, TSLA, AMZN, VTI (the EODHD demo token's set)
15,050 price bars  6 years, daily, raw and total-return bases
74 corporate actions  splits and dividends, with declaration dates
19,345 financial facts  SEC XBRL, with per-filing restatement history
30,325 macro observations  12 FRED series
```

## Try it

```bash
# What is loaded
curl localhost:8000/api/v1/instruments | jq '.instruments[].symbol'

# Point-in-time: Apple's revenue as it was known at two different dates
for year in 2022 2024; do
  curl -s "localhost:8000/api/v1/analysis/AAPL/fundamentals?as_of=${year}-06-30T00:00:00Z" \
    | jq -r "\"as of ${year}: \" + (.methods.profitability.values.revenue | tostring)"
done
# as of 2022: 365817000000   <- FY2021, the then-latest annual figure
# as of 2024: 383285000000   <- FY2023

# A historical reconstruction, cut off at the end of its own window
curl -s "localhost:8000/api/v1/analysis/AAPL/past?start=2022-01-01T00:00:00Z&end=2022-12-31T00:00:00Z" \
  | jq '{return: .methods.returns.values.total_return,
         volatility: .methods.realized_volatility.values.annualised_volatility,
         drawdown: .methods.drawdown.values.max_drawdown,
         beta: .methods.relative_strength.values.beta,
         disclosures: .disclosures | length}'

# Does any forecasting model beat a random walk?
curl -s "localhost:8000/api/v1/forecasts/AAPL/compare?horizon=1mo" \
  | jq '.results[] | select(.available) | {model: .model_key, skill: .metrics.skill_vs_baseline}'

# An evidence-linked report
curl -s "localhost:8000/api/v1/reports/AAPL/historical?start=2022-01-01T00:00:00Z&end=2022-12-31T00:00:00Z" \
  | jq '{summary, claims: .claim_kind_counts, key: .reproduction_key}'
```

A full smoke run over every endpoint group:

```bash
python -m scripts.smoke_api
```

## Data sources

| Source | Serves | Point-in-time? | Credential |
|---|---|---|---|
| **EODHD** | prices, splits, dividends, fundamentals | prices yes, fundamentals no | none (`demo`) or `EODHD_API_KEY` |
| **SEC EDGAR** | fundamentals, filings, 10.4k instrument universe | **yes, genuinely** | none; `SEC_USER_AGENT` required |
| **FRED** | 12 macro series | **only with `FRED_API_KEY`** | none, or a free key |
| **Yahoo** | prices | no | none; unofficial, fallback only |
| **fixture** | synthetic prices | yes | tests only, never served |

Details and caveats: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## API

29 endpoints at `/api/v1`, documented at `/docs`.

| Group | Endpoints |
|---|---|
| Instruments | list, search, detail with coverage and freshness |
| Market data | candles (with one-directional aggregation), latest stored close |
| Analysis | past, present, statistical, fundamentals, risk, bubble, trade-structure, custom |
| Forecasts | models, horizons, create, compare, history, single record |
| Reports | historical, current |
| System | health, methods, data-sources, limitations |
| Watchlists | CRUD |

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | module layout, data flow, the four clocks, why Decimal |
| [POINT_IN_TIME.md](docs/POINT_IN_TIME.md) | how each leak is prevented, and where it does not hold |
| [DATA_SOURCES.md](docs/DATA_SOURCES.md) | providers, licensing, what is not ingested |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | what the platform does not know |
| [OPERATIONS.md](docs/OPERATIONS.md) | running it, scheduled jobs, CI, backups |
| [TASKS.md](docs/TASKS.md) | milestone tracker, honest status |

## Tests

```bash
pytest                   # 354 tests: unit + database, no network
pytest -m network        # 18 live provider contract tests
pytest -m database       # the PostgreSQL-backed subset
```

Network tests are excluded by default: a run must not fail because a provider
is rate-limiting, and a provider's data changing must not look like a
regression in our code.

The tests worth reading first, because they are the ones that make the
platform's claims checkable rather than asserted:

- `tests/integration/test_point_in_time.py` — a bar is invisible before its
  session closes; a query before a correction returns the original value;
  backfilled rows stay visible to historical queries.
- `tests/integration/test_fundamentals_pit.py` — a restatement filed in 2023 is
  invisible to a 2022 analysis; a 90-day quarter never merges with a 272-day
  cumulative.
- `tests/integration/test_forecast_immutability.py` — a stored forecast cannot
  be edited, deleted, re-scored, or have its cutoff moved.
- `tests/unit/test_forecasting.py` — Gaussian data fed to a Gaussian model
  comes back calibrated; extrapolated drift gains no edge on a random walk.
- `tests/unit/test_statistics.py` — a random walk shows no significant
  autocorrelation, and seventeen calendar buckets of noise produce no seasonal
  effect that survives correction.
