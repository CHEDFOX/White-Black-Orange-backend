# Operations

## Running it

### Docker Compose

```bash
cp .env.example .env            # defaults work with no signup
docker compose up --build
docker compose run --rm api python -m scripts.seed
```

Migrations run in a one-shot `migrate` service that the API waits on, so two
API replicas cannot apply them concurrently. The API's healthcheck polls
`/api/v1/health`.

### Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
createdb sip
alembic upgrade head
python -m scripts.seed
./scripts/start_api.sh           # or: uvicorn app.main:app --reload
```

`scripts/start_api.sh` stops an existing instance by PID file rather than by
matching the process list — a `pkill -f "uvicorn app.main"` pattern also
matches the shell running the pkill, and `ss -ltnp` hides other users' PIDs
without privileges.

## Configuration

Everything is environment-driven; see `.env.example`. The settings that change
behaviour most:

| Variable | Effect |
|---|---|
| `MARKET_DATA_PROVIDERS` | comma-separated, priority order. A provider whose credential is unmet is skipped and the reason is reported by `/data-sources`. |
| `EODHD_API_KEY` | `demo` serves a fixed symbol set; a real key unlocks full coverage |
| `FRED_API_KEY` | **turns on real macro vintages.** Without it, macro analysis uses revised figures and discloses that per series. |
| `SEC_USER_AGENT` | required by the SEC; anonymous clients are throttled or blocked |
| `LLM_PROVIDER` | `null` by default. The platform is fully functional and reports contain no generated prose. |
| `REQUIRE_API_KEY` | off by default so a local checkout works; set it with `API_KEYS` for any deployment |
| `DEFAULT_BENCHMARK_TICKER` | `VTI`, because the EODHD demo token serves `VTI.US` but not `SPY.US` |

**No credential ever reaches the browser.** The frontend calls the backend from
Server Components only, so the backend address and any API key stay server-side.

## Scheduled jobs

A Celery worker and beat scheduler run five tasks. All are idempotent, because
a retried ingestion must not duplicate a bar or invent a revision.

| Task | Schedule | Purpose |
|---|---|---|
| `refresh_all_prices` | 22:30 UTC | incremental refresh with a deliberate overlap, so a provider's correction to a recent session is picked up as a revision rather than missed |
| `evaluate_due_forecasts` | 23:15 UTC | scores forecasts whose horizon has closed. **This is the job that makes "forecasts are evaluated without hindsight" true in a running system.** |
| `detect_recent_gaps` | 23:30 UTC | re-checks recent sessions against the trading calendar |
| `record_provider_health` | every 30 min | so `/data-sources` reports a measured state rather than an assumption |
| `refresh_macro` | 12:15 UTC | re-fetches the curated macro set |

The worker is optional. Without it the platform is read-only over whatever has
been seeded, which is a legitimate deployment.

`evaluate_due_forecasts` deliberately *skips* a forecast whose horizon has
closed on the calendar but whose data has not arrived. Scoring against a
partial window would flatter or punish the model unfairly, so it waits.

## Observability

Structured JSON logs via `structlog`, with a request id bound per request and
echoed in `X-Request-ID`. Every request logs its path, status and duration.

`InsufficientDataError` logs at `info`, not `error`: it is the designed outcome
when the data does not support an analysis, and alerting on it would train
operators to ignore the channel.

`GET /api/v1/health` reports database reachability and row counts.
`GET /api/v1/data-sources?probe=true` performs a live reachability check —
off by default because it consumes provider rate limit.

## CI

`.github/workflows/ci.yml` runs lint, tests against a real PostgreSQL service,
a typecheck, and a Docker build. Two checks are worth calling out:

**Migration drift.** CI autogenerates a revision against the migrated database
and fails if it is non-empty. A model changed without a migration passes every
test and then breaks a real deployment; this is what catches it.

**Network tests are excluded from the default run** and scheduled separately.
A build must not fail because a provider is rate-limiting, and a provider's
data changing must not look like a regression. The scheduled job runs them
weekdays after the US close, which is when a provider shape change would show
up.

## Backups and recovery

Not automated here; the deployment target decides. What matters for this
schema:

- **`price_bars`, `financial_facts`, `economic_observations` are append-only.**
  A point-in-time restore to any instant is coherent, because nothing is
  updated in place.
- **`forecasts` must never be restored selectively.** The immutability
  guarantee depends on the table being append-only; restoring a subset would
  let historical performance be curated.
- Everything else is re-derivable: `computed_features`, `regime_estimates`,
  `historical_episodes` and `generated_reports` are caches of deterministic
  computations over the observation tables, and can be dropped and rebuilt.

A sensible policy is a nightly base backup plus WAL archiving, with the
observation tables as the thing that must survive.

## Known operational gaps

Stated rather than discovered later:

- The rate limiter is per-process. A multi-instance deployment needs the
  Redis-backed equivalent.
- No distributed tracing.
- No error-monitoring integration wired up; the structured logs carry what a
  collector would need.
- Secrets come from the environment. A real deployment should use a secret
  manager.
- The incremental refresh covers recent sessions only. If a provider silently
  back-adjusts a historical series, nothing currently schedules the full
  re-fetch that would detect it as a revision.
