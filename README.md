# Stock Intelligence Platform — Backend

An evidence-driven stock intelligence and trader guidance API.

**Operating principle:** data before intelligence, validation before prediction,
evidence before recommendations. Every number a client receives is traceable to
a stored observation and a versioned calculation, and every estimate is labelled
as an estimate.

> This is research and educational tooling, not personalised investment advice,
> and it does not execute trades. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md)
> for what the platform does *not* know.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # defaults work with no signup

createdb sip                     # or use docker compose
alembic upgrade head
python -m scripts.seed_instruments   # registers instruments from SEC's universe
uvicorn app.main:app --reload
```

Then `http://localhost:8000/docs`.

## Data sources

Configured out of the box, with no API key required:

| Source | Serves | Point-in-time? | Notes |
|---|---|---|---|
| **EODHD** | prices, splits, dividends, fundamentals | prices yes, fundamentals no | Official API. Default token `demo` covers a fixed handful of symbols; set `EODHD_API_KEY` for full coverage. |
| **SEC EDGAR** | fundamentals, filings, US instrument universe | **yes** | Each XBRL fact carries the filing that reported it, so restatement history is preserved. |
| **FRED** | macro series | **only with `FRED_API_KEY`** | Without a key, only the latest revision is available and the platform says so. |
| **Yahoo (chart)** | prices | no | Unofficial fallback, not licensed for redistribution, aggressively rate-limited. |
| **fixture** | synthetic prices | yes | Deterministic test data, always labelled `synthetic`. |

Details and licensing posture: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module layout and data flow
- [docs/POINT_IN_TIME.md](docs/POINT_IN_TIME.md) — the four clocks and how look-ahead bias is prevented
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) — providers, licensing, coverage
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — what is not known or not supported
- [docs/TASKS.md](docs/TASKS.md) — milestone tracker

## Tests

```bash
pytest                       # unit tests, no network or database needed
pytest -m database           # requires PostgreSQL
pytest -m network            # performs live provider calls
```
