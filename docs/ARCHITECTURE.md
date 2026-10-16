# Architecture

## Shape

One modular application, not a mesh of services. The module boundaries are
drawn where the *rules* change, not where a deployment boundary might one day
be convenient:

```
app/
  core/          configuration, structured logging, domain errors, time and calendars
  db/            SQLAlchemy models, 44 tables, PIT columns on every observation
  providers/     provider abstraction and adapters (EODHD, Yahoo, SEC, FRED, fixture)
  ingestion/     validation, idempotent writes, revision handling, coverage tracking
  pit/           the point-in-time query layer -- one audited predicate
  analysis/      versioned method registry and the methods themselves
  forecasting/   models, scoring, walk-forward evaluation, immutable storage
  reporting/     findings, evidence, report generation, the guarded LLM layer
  services/      orchestration: database reads -> analysis inputs -> bundles
  api/           versioned REST endpoints and request/response schemas
  jobs/          Celery tasks: refresh, gap detection, forecast evaluation
```

The dependency direction is strictly one way: `api → services → analysis/
forecasting/reporting → pit → db`, with `providers` feeding `ingestion` only.
**Nothing in `analysis/` imports a provider or a database session.** That is
what makes the methods deterministic and unit-testable, and it is why a
licensed vendor can replace EODHD without touching a feature module.

## The data flow

```
provider adapter          normalised dataclasses, Decimal prices
      ↓
ingestion/validation      structural checks, duplicates, gaps, outliers
      ↓                   findings persisted, defects excluded, nothing invented
db (append-only)          four clocks per observation + a revision chain
      ↓
pit/queries               available_at <= T, newest revision only
      ↓
services/data_access      SeriesBundle: data + coverage + gaps + quality issues
      ↓
analysis/registry         run_method(): refuses below its minimum, returns
      ↓                   values + warnings + limitations + input fingerprint
services/analysis_service AnalysisBundle: results *and* explained absences
      ↓
reporting/generator       Findings, each with a mandatory epistemic label
      ↓
api                       JSON with disclosures attached
```

Each arrow narrows what the next stage is allowed to assume. By the time a
number reaches the API it carries the method version that produced it, a hash
of its exact inputs, the observation count behind it, and the limitations of
the method.

## The four clocks

Every observation table carries four timestamps, because conflating them is how
a financial dataset leaks the future:

| Column | Meaning |
|---|---|
| `event_time` | when the thing happened (quarter end, session open, rate decision) |
| `published_at` | when the publisher first released it |
| `available_at` | when *this platform* could first have acted on it |
| `ingested_at` | when our pipeline wrote the row — operational only, never a filter |

Point-in-time queries filter on `available_at` and nothing else. Filtering on
`ingested_at` would make a backfill written today hide 2019 data from a 2019
analysis; filtering on `event_time` would make a quarter's revenue knowable
before it was filed.

Revisions are append-only. A corrected value is a new row with `revision + 1`
and its own `available_at`; the superseded row keeps `superseded_at`. A query
dated before the correction therefore still returns the original figure.

See [POINT_IN_TIME.md](POINT_IN_TIME.md).

## Why prices are Decimal

Prices and volumes are `Decimal` from the moment a provider's JSON is parsed
(`json.loads(..., parse_float=Decimal)`) to the moment they leave the API (as
strings). They become `float` exactly once, at the NumPy boundary in
`services/data_access`, and never convert back.

A float close of `182.63` is `182.63000000000000255795384873636066913604736328125`.
Over a reconciliation against an exchange statement, or a sum of a million
volumes, that noise shows up. The uniqueness key on `price_bars` includes the
adjustment basis for the same reason: a raw and a total-return bar for the same
instant are different observations and must not overwrite each other.

## Keys that are not obvious

**Instruments** are keyed on an internal UUID. A ticker is not a global key:
`BMW` is a different instrument on XETR and on the US OTC market, and `FB`
referred to Meta before 2022. Tickers are time-ranged rows, provider symbols
live in a side table, and `resolve_symbol_at` resolves a symbol *as of* an
instant.

**Fiscal periods** are keyed on `(issuer, period_end, period_kind)` where the
kind is derived from the reported duration. Two earlier approaches both failed
in production:

- Keying on SEC's `fy`/`fp` fields is wrong, because those describe the
  *filing*: FY2021 revenue restated in the FY2023 10-K arrives tagged
  `fy=2023`, which filed a 2021 figure under 2023.
- Keying on `(fiscal_year, fiscal_period)` collides, because a 10-Q reports a
  discrete quarter *and* a year-to-date cumulative under the same `fp`. A
  90-day quarter and a 272-day nine-month total merged into one period, so a
  "quarterly revenue" could silently be a nine-month figure.

`period_end` and the duration are facts in the payload, so keying on them is
unambiguous. `fiscal_year` is retained as a display label only.

## Where floats, estimates and prose are allowed

Three boundaries are drawn deliberately:

- **Decimal → float** happens once, in `services/data_access`, where NumPy and
  SciPy need it. Nothing converts back.
- **Estimates** are labelled at the point of creation. A `ClaimKind` is a
  required field on every `Finding`, so an unlabelled conclusion cannot be
  constructed.
- **Generated prose** is additive and validated. The LLM receives findings as
  structured data, and any number in its output that is not derivable from
  those findings causes the narrative to be **discarded**. With
  `LLM_PROVIDER=null` nothing is called and reports are complete.

## Technology choices and their reasons

| Choice | Why |
|---|---|
| PostgreSQL | window functions for revision selection, JSONB for payloads, exact `NUMERIC` |
| No TimescaleDB | the query pattern is "a window of one series", which a composite B-tree index serves well. Partitioning becomes worthwhile at a scale this does not have, and it is a deployment cost paid up front. |
| SQLAlchemy 2 + Alembic | a typed ORM, and migrations that CI checks for drift against the models |
| FastAPI + Pydantic | typed request and response schemas, generated OpenAPI |
| Celery + Redis | the scheduled work is coarse and idempotent; a heavier orchestrator would buy nothing |
| NumPy/SciPy/statsmodels | established implementations of the statistical tests, so the engine is not reimplementing an ADF test |
| No PyTorch | no model here has demonstrated that nonlinearity helps. Adding the dependency before the evidence would be complexity as a substitute for performance. |
| Canvas chart, hand-written | prices must stay exact decimal strings through to the readout; gaps must be drawn as breaks; direction must be encoded without relying on colour |

## Scaling, when it is needed

The honest current state: a single API process, a single worker, one
PostgreSQL. What it would take to grow, in the order the bottlenecks would
actually appear:

1. **Read volume** → cache chart responses keyed on
   `(instrument, resolution, basis, window, as_of)`. Analysis responses are
   already deterministic for a fixed cutoff, so they are cacheable without
   invalidation logic.
2. **Instrument count** → the refresh fan-out is already per-instrument tasks;
   add workers.
3. **Bar count** → partition `price_bars` by `event_time` range. The schema is
   written so this is a migration, not a rewrite.
4. **Multi-instance rate limiting** → the in-process limiter in `api/deps.py`
   is honest about being per-process; replace it with the Redis-backed
   equivalent.

None of this is built, because none of it is measured. Optimising an
unmeasured bottleneck would add infrastructure the platform cannot yet justify.
