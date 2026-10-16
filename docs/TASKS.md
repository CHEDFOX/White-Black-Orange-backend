# Task tracker

Honest state as of the latest commit. "Done" means implemented, tested, and
verified against real data or a real database — not merely scaffolded.

## Milestone 1 — Working foundation ✅

| Item | Status |
|---|---|
| Repository inspection (both repos were empty; nothing to preserve) | done |
| Environment-driven configuration, no secrets in code | done |
| 44-table versioned schema with PIT columns on every observation | done |
| Alembic migrations, applied against PostgreSQL 16 | done |
| Provider abstraction with normalised dataclasses | done |
| EODHD adapter — verified live (prices, actions, fundamentals, quote) | done |
| SEC EDGAR adapter — verified live (facts, restatements, 10.4k universe) | done |
| FRED adapter — verified live; refuses to fake a vintage without a key | done |
| Yahoo adapter — verified live, kept as a documented fallback | done |
| Deterministic synthetic provider with injectable defects | done |
| Provider registry with credential gating | done |
| Instrument search and listing API | done |

## Milestone 2 — Reliable market data ✅

| Item | Status |
|---|---|
| Validation: OHLC structure, duplicates, gaps, outliers, alignment | done, 37 tests |
| Split-shaped outlier detection with a diagnostic hint | done |
| Idempotent ingestion; revisions are append-only | done, 33 tests |
| Chart API with one-directional aggregation | done |
| Coverage and freshness tracking and reporting | done |
| End-to-end verified: 15,050 real bars, two bases, zero gaps | done |

## Milestone 3 — Historical intelligence ✅

| Item | Status |
|---|---|
| Point-in-time query layer with one audited predicate | done, 16 tests |
| SEC fundamentals with per-filing restatement history | done, 16 tests |
| FRED macro with vintage support and honest disclosure | done |
| Corporate actions dated from announcement, not ex-date | done |
| Filing-derived event timeline | done |
| Historical report with source attribution | done |
| Verified: AAPL revenue reads correctly at four different cutoffs | done |

## Milestone 4 — Scientific analysis ✅

| Item | Status |
|---|---|
| Versioned method registry with declared inputs and limitations | done |
| 9 technical, 4 price, 9 statistical, 5 fundamental, 2 regime methods | done |
| 4 risk methods and the bubble assessment | done |
| Multiple-testing correction applied where the platform tests a family | done |
| Regime estimation with explicit thresholds and a runner-up label | done |
| Scale-free similarity feature vector, versioned | done |
| Experiment-log table | schema only — not wired to an API |

## Milestone 5 — Forecasting and risk ✅

| Item | Status |
|---|---|
| Five baseline models, each declaring assumptions and invalidation | done |
| Forecast immutability enforced three ways | done, 20 tests |
| CRPS, PIT, interval coverage, calibration verdicts | done |
| Walk-forward evaluation with recorded execution assumptions | done |
| Scheduled evaluation job — verified scoring a real forecast | done |
| Scenario analysis with historical frequencies | done |
| Risk profile, position sizing, trade-structure arithmetic | done |
| Multi-signal bubble assessment, 10 signals | done |

## Milestone 6 — Unified guidance ✅

| Item | Status |
|---|---|
| Analysis orchestrator returning results *and* explained absences | done |
| Evidence-linked report generator with mandatory claim labels | done, 31 tests |
| Reproduction key over inputs and method versions | done |
| LLM abstraction with a number-fabrication guard | done |
| 29-endpoint versioned REST API | done |

## Milestone 7 — Production readiness 🟡

| Item | Status |
|---|---|
| Dockerfile (multi-stage, unprivileged, healthcheck) | done |
| docker-compose with ordered migrations | done |
| GitHub Actions CI: lint, tests, typecheck, Docker, migration-drift check | done |
| Live provider contract tests, scheduled separately | done, 18 tests |
| Structured logging with request ids | done |
| Documentation: architecture, PIT, sources, limitations, operations | done |
| Error-monitoring integration | not wired up |
| Distributed tracing | not done |
| Redis-backed rate limiter for multi-instance | not done — current one is per-process and says so |
| Automated backups | deployment-specific; policy documented |
| Broader market coverage | limited by provider contracts, not by code |

## Frontend ✅

| Item | Status |
|---|---|
| Search, listing, instrument overview | done |
| Past / Present / Future / Risk / Bubble / Report views | done |
| Canvas candlestick chart with volume, gaps, crosshair, forecast fan | done |
| Claim labels, explained absences, required disclosures | done |
| All 11 pages verified rendering real data | done |
| 29 tests, typecheck and lint clean, production build succeeds | done |

## Not built, and why

- **Index constituents, point-in-time.** The peer-group schema is time-ranged
  and ready; the data needs a licensed index feed. Without it a historical
  index comparison would be survivorship-biased, so none is offered.
- **Options, short interest, positioning, flows.** Four bubble signals report
  unavailable as a result, and the assessment says a low score may reflect
  missing data rather than low risk.
- **News ingestion and sentiment.** The schema (articles, entity links,
  extracted claims with verbatim passages) exists. No provider is configured,
  so no sentiment is computed and no claim about investor psychology is made.
- **Nonlinear ML models.** No evidence yet that nonlinearity helps on this
  data. Adding PyTorch before the evidence would be complexity substituting
  for performance.
- **Live streaming.** No licensed real-time feed; the quote endpoint is
  explicitly the last stored close rather than a live price.
- **Trade execution.** Out of scope by design.

## Environment notes

- `stooq.com` is unreachable from the build environment (connection reset by
  the egress policy), so no Stooq adapter is included.
- True macro vintages need a free `FRED_API_KEY`; the keyless CSV endpoint
  accepts `vintage_date` and silently ignores it (verified).
- Yahoo rate-limits per IP after a short burst, which is why it is a fallback
  with a low default limit.
