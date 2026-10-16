# Data sources

Four real sources, all verified working, plus a synthetic provider for tests.
Each entry states what it provides, its licensing posture, and what it cannot
do. The same information is served from `GET /api/v1/data-sources`, so a client
can display the caveats alongside the data.

## EODHD — primary market data

**Serves:** daily and intraday OHLCV, raw and adjusted; splits and dividends
with declaration dates; fundamentals; a delayed quote.

**Why it is the default:** an officially documented REST API with a published
free tier, which makes its licensing posture far clearer than a scraped browser
endpoint. The literal token `demo` serves a fixed symbol set, so a fresh
checkout has real market data with no signup.

**Demo token coverage:** `AAPL.US`, `MSFT.US`, `TSLA.US`, `AMZN.US`, `VTI.US`,
`BTC-USD.CC`, `EURUSD.FOREX`. Anything else returns 403, and the adapter says
so rather than reporting a bare failure. Set `EODHD_API_KEY` for full coverage.

**Point-in-time:** prices yes, by a property of the data — a bar is immutable
once its session closes. Fundamentals no: the endpoint returns latest-restated
figures, which is why SEC EDGAR carries that workload instead.

**Licensing:** free and paid tiers; redistribution requires a commercial
agreement. Attribution required.

**Caveats:**
- Adjusted closes use EODHD's own methodology; the raw series is stored
  alongside so any adjustment can be recomputed.
- Daily bar timestamps are derived from the exchange calendar — the API returns
  only a date.
- The `demo` token's symbol limit, above.

## SEC EDGAR — fundamentals, filings, instrument universe

**Serves:** XBRL company facts with per-filing provenance; filing metadata with
exact acceptance timestamps; a 10,444-symbol CIK↔ticker universe.

**Why it carries fundamentals:** `companyfacts` returns every reported value
*together with the filing that reported it and the date it was filed*, so a
restated period appears once per filing. Selecting the newest row whose filing
date precedes a cutoff yields the figure a reader genuinely had. That is a
property no commercial fundamentals feed in this price bracket offers.

**Point-in-time:** yes, genuinely. See
[POINT_IN_TIME.md](POINT_IN_TIME.md#fundamentals--genuinely-pit-correct).

**Licensing:** US government work, public domain. The SEC requires a
descriptive `User-Agent` with real contact details and a request rate at or
below 10/s; both are enforced in the adapter. Set `SEC_USER_AGENT`.

**Caveats:**
- SEC registrants only: US-listed issuers and foreign private issuers that file
  with the SEC. Non-US listings are not covered.
- XBRL facts begin around 2009; earlier filings are text-only.
- Issuers use different `us-gaap` concepts for the same economic quantity, so
  the concept mapping is explicit, ordered by preference, and incomplete rather
  than guessed.

## FRED / ALFRED — macro series

**Serves:** twelve curated series across rates, inflation, employment, growth,
risk, credit, currency and commodities, each with its documented publication
cadence.

**Point-in-time:** **only with `FRED_API_KEY`.** Without one, the keyless CSV
endpoint serves the latest revision and silently ignores `vintage_date`
(verified). The adapter raises rather than serving a revised figure as a
vintage, and every stored series carries `has_vintages=False` so each PIT read
discloses it. A free key turns on real ALFRED vintages.

**Licensing:** mostly US federal government works in the public domain; some
third-party series carry their own terms. Attribution required.

## Yahoo Finance chart endpoint — fallback only

**Serves:** daily and intraday OHLCV, splits and dividends.

**Status:** `official_api=False`, `redistribution_allowed=False`. It is an
undocumented endpoint used by Yahoo's own web client. It requires no
credential, which makes it usable for development, and it is **not a licensed
redistribution feed**.

**Rate limiting:** aggressive and per-IP. It returned 429 after a short burst
during development, which is why the default limit is 5 requests/minute and
why it is listed second.

**Use a licensed provider in production.** Add a key and list the provider
ahead of `yahoo_chart` in `MARKET_DATA_PROVIDERS`; the analysis engines do not
change.

## Fixture provider — tests only

Deterministic synthetic data from a seeded PRNG, with injectable gaps,
duplicates and structurally broken bars. Its data-source row is created with
`classification='synthetic'`, and `exclude_synthetic` keeps it out of every
production read — asserted by
`test_synthetic_bars_are_excluded_from_production_reads`.

It exists because a validator only ever shown clean data is untested. The
defects it injects are what the validation suite catches.

## Not reachable from this build

- **Stooq** — the egress policy in the build environment resets the connection.
  No adapter is included; writing one against an unreachable host would be
  untested code.

## Not ingested at all

These are absences with consequences, so they are listed rather than left to be
discovered:

| Missing | What it costs |
|---|---|
| Options chains | the bubble engine's options signal reports unavailable; implied-vs-realised volatility cannot be computed |
| Short interest, positioning, flows | three more speculation signals unavailable |
| Index constituents, point-in-time | a historical index comparison would be survivorship-biased, so none is offered |
| Analyst estimates | no beat/miss reconstruction |
| News text | event timelines hold filings and corporate actions only; no sentiment |
| Trades and quotes (L1/L2) | no spread or microstructure measures |

The bubble assessment reports each unavailable signal with its reason and
excludes it from the score's denominator, so a low score from six evaluable
signals is never presented as a low score from ten.

## Adding a provider

Implement `MarketDataProvider` in `app/providers/`, declare a
`ProviderCapabilities` (including an honest `supports_point_in_time`), register
it in `app/providers/registry.py` with a credential check, and add it to
`MARKET_DATA_PROVIDERS`. Nothing in `analysis/` or `forecasting/` changes.

A contract test in `tests/integration/test_providers_live.py` is expected. It
should assert the *shape* of the response, not its values: "the adjusted series
is continuous across a known split" is a contract worth protecting, while "the
close on this date is 182.63" is a brittle assertion against a vendor's
back-adjustment.
