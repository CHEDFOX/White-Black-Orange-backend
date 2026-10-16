# Task tracker

Status as of the latest commit. Honest state, not aspiration.

## Milestone 1 — Working foundation

| Item | Status |
|---|---|
| Repository inspection (both repos were empty; nothing to preserve) | done |
| Configuration via environment, no secrets in code | done |
| 44-table versioned schema with PIT columns on every observation | done |
| Alembic migration applied against PostgreSQL 16 | done |
| Provider abstraction (`MarketDataProvider`, normalised dataclasses) | done |
| EODHD adapter, verified live (prices, splits, dividends, fundamentals, quote) | done |
| Yahoo chart adapter, verified live | done |
| SEC EDGAR adapter, verified live (facts + restatement history + 10.4k universe) | done |
| FRED adapter, verified live; refuses to fake a vintage without a key | done |
| Deterministic synthetic fixture provider with injectable defects | done |
| Provider registry with credential gating | done |
| Instrument search API | in progress |
| Architecture + limitations documentation | in progress |

## Milestone 2 — Reliable market data
| Item | Status |
|---|---|
| Validation suite (OHLC, duplicates, gaps, outliers, alignment) | pending |
| Idempotent ingestion with revision handling | pending |
| Chart + resampling API | pending |
| Coverage and freshness reporting | pending |

## Milestone 3 — Historical intelligence
Pending.

## Milestone 4 — Scientific analysis
Pending.

## Milestone 5 — Forecasting and risk
Pending.

## Milestone 6 — Unified guidance
Pending.

## Milestone 7 — Production readiness
Pending.

## Known blockers / environment notes
- `stooq.com` is unreachable from the build environment (connection reset by the
  egress policy), so no Stooq adapter is included.
- True macro vintages need a free `FRED_API_KEY`; the keyless CSV endpoint
  accepts `vintage_date` and silently ignores it (verified).
- Yahoo rate-limits per IP after a short burst; it is a fallback, not a primary.
