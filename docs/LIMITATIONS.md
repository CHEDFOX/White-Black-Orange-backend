# Limitations

What this platform does not know, cannot do, or does only approximately. It is
served from `GET /api/v1/limitations` as well, so a client cannot display
analysis without being able to show the caveats.

The ordering is deliberate: the things most likely to mislead come first.

## Forecasting

**No model here has demonstrated profitable predictive performance.** Five
models are registered and every one is a baseline. Three (`random_walk`,
`ewma_volatility`, `student_t`) have zero drift by construction and make no
directional call at all; their value is as a benchmark and as a risk-interval
estimate. `historical_drift` exists to demonstrate why extrapolating a sample
mean return is a mistake — it reports the standard error that makes its own
drift statistically indistinguishable from zero. `seasonal_naive` is a control
expected to perform badly.

A forecast is a probability distribution under stated assumptions, not a
prediction. The interval matters more than the central estimate.

**Walk-forward windows overlap**, so the forecasts in an evaluation are not
independent and the effective sample is smaller than the count. The limitation
is reported with an effective-sample estimate alongside it.

**Backtests are gross of execution costs** unless the recorded assumptions say
otherwise, and `WalkForwardConfig` defaults both cost and slippage to zero
precisely so that choice is visible rather than buried. A hypothetical backtest
is never actual trading performance.

**Square-root-of-time scaling** assumes serially uncorrelated returns and
constant volatility over the horizon. Both fail at horizons beyond a few
months, and each model warns when used there.

## Data

**Coverage is what has been ingested, not a global universe.** Five instruments
in the seeded state. `GET /instruments` is the honest answer to "what can I
analyse here?"

**Macro series carry no revision vintages** without a `FRED_API_KEY`, so
historical macro analysis uses revised figures. Each affected series discloses
this on every point-in-time read.

**No options, short-interest, positioning, flow or index-constituent data.**
Four of the ten bubble signals therefore report unavailable, and they include
some of the strongest: options positioning, leverage, margin debt, retail
flows. A low vulnerability score may reflect missing data rather than low risk,
and the assessment says so.

**No news or filings text analysis.** Event timelines contain corporate actions
and filing metadata only. No sentiment is computed anywhere, so no claim about
investor psychology is made.

**Intraday depth depends entirely on the configured provider.** The seeded
state holds daily bars only, and a request for minute bars is refused with an
explanation rather than approximated from daily data.

**Bar availability uses a documented lag assumption**, not a measured feed
latency. For a strategy sensitive to seconds, that assumption is too coarse.

## Analysis

**Statistical relationships are not causal claims.** An event preceding a price
movement is not evidence that it caused it. The platform reports events and
price movements separately and labels any proposed cause as a hypothesis.

**Regime labels are estimates from an explicit rule**, not observations. The
thresholds are conventions chosen for interpretability, not parameters
estimated from data. A different rule disagrees near the boundaries, and the
output reports the runner-up classification so a reader can see how close the
call was.

**Reported p-values are corrected for the tests the platform performed in that
family**, not for tests performed and discarded elsewhere. The experiment log
exists for that larger problem; it is not yet wired into the API.

**The change-point scan's raw p-value is badly optimistic** because it searches
every candidate split. The Bonferroni-corrected value is the one reported as
evidence, and the raw one is labelled as a ranking score.

**Similarity between historical windows is measured on a specific feature
set** and has no established predictive value. Visually similar charts do not
imply similar outcomes. The feature-set version is stored with every vector so
a match is never made across incompatible versions.

**Support and resistance levels are descriptive clusters of past pivots.** A
level with two touches is not statistically distinguishable from coincidence,
and a different lookback gives different levels.

**Trading calendars** come from `pandas_market_calendars` where a mapping
exists, and fall back to a weekday-only approximation otherwise. The
`coverage_note` on every calendar states which case applies, and an
unrecognised exchange produces a visible caveat rather than a silent
approximation.

## Speculative-risk ("bubble") assessment

**It detects conditions associated with past speculative episodes.** It cannot
distinguish a bubble from a justified re-rating, and most instruments that
trigger several signals do not crash.

**No crash date and no crash probability is produced.** The output is a
*vulnerability* estimate plus the conditions that would strengthen or weaken
it. Overvaluation can persist for years, and an episode can end in stagnation
or in fundamentals catching up rather than in a decline.

**The weights are a documented design choice, not a fitted model.** Two
reasonable people would weight these signals differently. The formula is
disclosed in the response.

**Every historical bubble comparison is survivorship-biased**: the episodes we
remember are the ones that ended dramatically.

## Fundamentals

**Accounting definitions differ between issuers and jurisdictions**, so margins
and ratios are not strictly comparable across companies.

**ROE is inflated by leverage** and is not reported at all when equity is zero
or negative, because the ratio is undefined rather than large.

**Invested capital, free cash flow and enterprise value** are each defined
differently by every data vendor. The definitions used are returned alongside
the figures.

**The share count is the diluted weighted average for the reporting period**,
not the count outstanding today, so a multiple computed from a current price
mixes two dates. This is disclosed in the method's limitations.

**Operating leases, pension obligations and off-balance-sheet commitments** are
not captured and can be material.

## Regulatory and operational

**This is research and educational tooling, not personalised investment
advice.** It makes no recommendations and takes no account of any individual's
circumstances, objectives or risk tolerance. Every risk method carries
`disclaimer_code='educational_research'`.

**No trade execution exists**, and none would be added without separate review,
explicit user authorisation and independent risk controls.

**The rate limiter is per-process.** A multi-instance deployment needs the
Redis-backed equivalent; the current one is honest about its scope in its
docstring rather than pretending to be global.

**Authentication is a shared API key, optional by default.** Adequate for a
single-tenant deployment, and explicitly not an authorisation system. The
watchlist endpoints identify a caller by an `X-Subject` header, which is a
convenience, not a security boundary.

**Yahoo data is not licensed for redistribution.** If the Yahoo adapter is
active, the data-source endpoint and every report's attribution say so.
