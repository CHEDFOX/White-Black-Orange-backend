# Point-in-time correctness

The platform's central claim is that it can reconstruct what was *knowable* at
a past instant. This document says how that is implemented, what it guarantees,
and — importantly — where it does not hold.

## The problem

A financial dataset leaks the future in several distinct ways, and each needs a
different defence:

| Leak | What it looks like | Defence here |
|---|---|---|
| **Look-ahead** | using today's close in a morning analysis | `available_at` is the bar's close plus provider lag |
| **Restatement** | using a 2024-restated FY2021 revenue in a 2022 analysis | one row per (period, filing); newest visible revision wins |
| **Revision** | using today's revised GDP to "predict" 2008 | vintages where available, and a disclosure where not |
| **Survivorship** | analysing today's index members over a past decade | time-ranged peer-group and ticker membership |
| **Crawl-time** | a 2020 article "available" in 2020 because a crawler indexed it in 2024 | news filters on `published_at`, never `ingested_at` |
| **Ticker reuse** | resolving `FB` to nothing because it is now `META` | time-ranged ticker rows, resolved as of the cutoff |

## The four clocks

```
event_time      ──  when it happened in the world
published_at    ──  when the publisher first released it
available_at    ──  when THIS PLATFORM could first have acted on it   ← the only filter
ingested_at     ──  when our pipeline wrote the row                   ← never a filter
```

`ingested_at` is the trap. It is the timestamp a naive implementation reaches
for, and filtering on it means a backfill run today makes 2019 data invisible
to a 2019 analysis. The test
`test_backfilled_rows_are_visible_to_historical_queries` asserts against
exactly that.

## The predicate

Every point-in-time read goes through `app/pit/availability.py`. The predicate
is two clauses, and both are required:

```python
and_(
    model.available_at <= as_of.cutoff,          # not published yet → invisible
    or_(model.superseded_at.is_(None),           # not yet replaced
        model.superseded_at > as_of.cutoff),     # or replaced after our cutoff
)
```

Filtering on availability alone is not enough: it returns *every* vintage
published before the cutoff, so a series would contain duplicates. Revision
selection uses a window function partitioned by the entity and period, ordered
by `available_at DESC, revision DESC`, taking the first row. A correlated
subquery would be simpler to read and degrades badly once a series has millions
of rows with several vintages each.

`AsOf` wraps the cutoff in its own type so it cannot be confused with an
ordinary timestamp argument, rejects naive datetimes, and carries the
disclosures a report must repeat.

## What each source actually guarantees

This is where honesty matters more than completeness.

### Prices — PIT-safe, by a property of the data

A daily bar is immutable once its session closes. `available_at` is the bar's
close plus the provider's delivery lag (8 hours for EODHD EOD, documented as an
assumption rather than a measurement). A provider *correction* to a recent
session is picked up by the overlapping incremental refresh and stored as a new
revision, so the original remains queryable at its original cutoff.

The limitation: EODHD serves only the current revision of a series, so the
platform records when *it* first saw a value, not what EODHD published on a
past date. For bars that distinction is immaterial. For anything revisable it
is not, which is why fundamentals come from a different source.

### Fundamentals — genuinely PIT-correct

SEC EDGAR's `companyfacts` returns every reported value *with the filing that
reported it and the date it was filed*. A fiscal period therefore appears once
per filing that restated it. `available_at` comes from the filing's acceptance
timestamp — the exact instant it became public — and falls back to end of the
filing day when the submissions index omits it, which is conservative.

Verified against real data: Apple's revenue reads

| Cutoff | Revenue returned | Which year that is |
|---|---|---|
| 2021-06-30 | 274,515,000,000 | FY2020, the then-latest annual figure |
| 2022-06-30 | 365,817,000,000 | FY2021 |
| 2023-06-30 | 394,328,000,000 | FY2022 |
| 2024-06-30 | 383,285,000,000 | FY2023 |

Every figure matches the filings, and each cutoff returns what an analyst
genuinely had.

The limitation: XBRL facts begin around 2009, so earlier filings are text-only
and not ingested. Coverage is SEC registrants only.

### Macro — PIT **only with a FRED API key**

Without a key the platform reads `fredgraph.csv`, which serves the latest
revision. That endpoint accepts a `vintage_date` parameter and **silently
ignores it** — verified by requesting GDPC1 as of 2020-05-01 and receiving
today's rebased figures. Rather than serve those as a vintage, the adapter
**raises**:

```
UnsupportedError: a point-in-time vintage of UNRATE as of 2020-04-15 was
requested, but no FRED_API_KEY is configured. The keyless CSV endpoint returns
only the latest revision; serving it as a vintage would be a look-ahead bug.
```

Where macro values *are* stored without vintages, `available_at` is estimated
from each series' documented publication lag, `publication_known` is set to
`False`, and `EconomicSeries.has_vintages` is `False` — so every PIT read of
that series returns a coverage note saying the figures are revised rather than
as-published. With a key, the adapter uses ALFRED realtime windows and the
vintages are real.

### News and events — PIT by construction, but thin

Events filter on their own `published_at`. Corporate actions take
`available_at` from the *announcement* date where the provider supplies one: a
split announced on 1 June with an ex-date of 1 July was actionable information
in June, and dating it to July would understate what the market knew. Where no
announcement date exists, the ex-date is used and the ingestion report says so.

The limitation: no news provider is configured in this build, so event
timelines contain filings and corporate actions only.

## The tests that make this checkable

`tests/integration/test_point_in_time.py` (16 tests) and
`tests/integration/test_fundamentals_pit.py` (16 tests) assert the invariants
directly, each named for the specific leak it prevents:

- a bar is invisible before its session closes;
- a query before a correction returns the original value;
- three vintages of one bar collapse to one row;
- the middle vintage is selected correctly for a cutoff between corrections;
- backfilled rows are visible to historical queries;
- a restatement filed in 2023 is invisible to a 2022 analysis;
- each period contributes exactly one row at any cutoff;
- synthetic fixture data never reaches a production read;
- a renamed ticker resolves to the same instrument in both eras;
- `assert_no_future_rows` checks the invariant against the database itself
  rather than trusting that every call site remembered the predicate.

## What this does not guarantee

- **Index constituents.** No index-membership data is ingested, so a historical
  index comparison would be survivorship-biased. The peer-group schema is
  time-ranged and ready for it; the data is not there.
- **Analyst estimates.** None ingested, so a "beat or miss" cannot be
  reconstructed.
- **Intraday availability.** Bar availability uses a documented lag assumption,
  not a measured feed latency. For a strategy sensitive to seconds, that
  assumption is too coarse.
- **Provider back-adjustment.** If a provider silently restates a historical
  price series, the overlapping refresh only covers recent sessions. A full
  re-fetch would detect it as a revision; nothing currently schedules one.
- **Original-release macro without a key.** As above.
