"""Live provider contract tests.

Marked ``network`` and excluded from the default run, for two reasons: a build
must not fail because a third-party provider is rate-limiting, and a provider's
*data* changing must not look like a regression in our code.

What these assert is the **shape** of each response, not its values. "AAPL's
close on a given day is 182.63" would be a brittle assertion against a vendor's
back-adjustment; "the EOD payload still carries a parseable date and four
positive prices, and the adjusted series is still continuous across a known
split" is the contract we actually depend on. When a provider changes that
shape, the adapter breaks silently in production, and this is the test that
catches it.
"""

from __future__ import annotations

import datetime as dt
import decimal
import itertools
import os

import pytest

from app.core.timeutil import UTC
from app.db.models.enums import AdjustmentBasis, CorporateActionType, InstrumentType
from app.providers.base import ProviderFetchRequest

pytestmark = [pytest.mark.network, pytest.mark.slow]


@pytest.fixture(scope="module")
def eodhd():
    from app.providers.eodhd import EodhdProvider

    provider = EodhdProvider()
    yield provider
    provider.close()


@pytest.fixture(scope="module")
def sec():
    if not os.environ.get("SEC_USER_AGENT"):
        pytest.skip(
            "SEC_USER_AGENT is not set; the SEC requires a descriptive agent with "
            "contact details and throttles anonymous clients"
        )
    from app.providers.sec_edgar import SecEdgarProvider

    provider = SecEdgarProvider()
    yield provider
    provider.close()


@pytest.fixture(scope="module")
def fred():
    from app.providers.fred import FredProvider

    provider = FredProvider()
    yield provider
    provider.close()


class TestEodhdContract:
    def test_the_provider_is_reachable(self, eodhd):
        status = eodhd.health_check()
        assert status.healthy, f"EODHD health check failed: {status.detail}"

    def test_instrument_metadata_still_carries_what_we_read(self, eodhd):
        instrument = eodhd.resolve_instrument("AAPL.US")
        assert instrument is not None
        assert instrument.symbol == "AAPL"
        assert instrument.currency == "USD"
        assert instrument.exchange_code
        assert instrument.instrument_type in set(InstrumentType)
        # The CIK is what links an instrument to its SEC filings.
        assert (instrument.raw or {}).get("cik"), (
            "the fundamentals payload no longer carries a CIK, so instruments "
            "can no longer be linked to SEC filings"
        )

    def test_eod_bars_parse_to_exact_decimals(self, eodhd):
        batch = eodhd.fetch_bars(
            ProviderFetchRequest(
                symbol="AAPL.US",
                start=dt.datetime(2024, 1, 2, tzinfo=UTC),
                end=dt.datetime(2024, 1, 12, tzinfo=UTC),
            )
        )
        assert len(batch.bars) >= 5
        for bar in batch.bars:
            # Decimal, not float: a float close carries binary noise into the
            # ledger users reconcile against an exchange.
            assert isinstance(bar.close, decimal.Decimal)
            assert bar.high >= bar.low
            assert min(bar.open, bar.high, bar.low, bar.close) > 0
            assert bar.event_time.tzinfo is not None
            assert bar.bar_close_time > bar.event_time

    def test_daily_timestamps_land_on_the_session_not_midnight(self, eodhd):
        """A bar dated midnight UTC sits on the wrong day for a US session."""
        batch = eodhd.fetch_bars(
            ProviderFetchRequest(
                symbol="AAPL.US",
                start=dt.datetime(2024, 3, 4, tzinfo=UTC),
                end=dt.datetime(2024, 3, 8, tzinfo=UTC),
            )
        )
        assert batch.bars
        for bar in batch.bars:
            assert bar.event_time.hour in {13, 14}, (
                f"a US session open should be 13:30 or 14:30 UTC depending on DST, "
                f"got {bar.event_time.isoformat()}"
            )

    def test_the_adjusted_series_is_continuous_across_a_known_split(self, eodhd):
        """Apple split 4-for-1 on 2020-08-31.

        On the raw basis the close drops roughly 75% overnight. On the adjusted
        basis it must not: a jump there means the adjustment is being applied
        wrongly, which silently corrupts every return in the series.
        """
        request = {
            "symbol": "AAPL.US",
            "start": dt.datetime(2020, 8, 27, tzinfo=UTC),
            "end": dt.datetime(2020, 9, 2, tzinfo=UTC),
        }
        raw = eodhd.fetch_bars(
            ProviderFetchRequest(**request, adjustment_basis=AdjustmentBasis.RAW)
        )
        adjusted = eodhd.fetch_bars(
            ProviderFetchRequest(
                **request, adjustment_basis=AdjustmentBasis.TOTAL_RETURN
            )
        )
        assert len(raw.bars) >= 4 and len(adjusted.bars) >= 4

        def biggest_move(bars) -> float:
            moves = [
                abs(float(b.close / a.close) - 1.0)
                for a, b in itertools.pairwise(bars)
            ]
            return max(moves)

        assert biggest_move(raw.bars) > 0.5, (
            "the raw series should show the ~75% split drop; if it does not, the "
            "provider has started serving pre-adjusted data under the raw basis"
        )
        assert biggest_move(adjusted.bars) < 0.15, (
            "the adjusted series jumped across the split, so the adjustment is "
            "being applied incorrectly"
        )

    def test_corporate_actions_still_carry_announcement_dates(self, eodhd):
        """The announcement date is what makes a split point-in-time correct."""
        actions = eodhd.fetch_corporate_actions(
            "AAPL.US", dt.date(2020, 1, 1), dt.date(2020, 12, 31)
        )
        assert actions

        splits = [a for a in actions if a.action_type == CorporateActionType.SPLIT]
        assert splits, "the 2020 4-for-1 split is missing from the action feed"
        split = splits[0]
        assert split.ex_date == dt.date(2020, 8, 31)
        assert split.ratio_numerator == 4
        assert split.ratio_denominator == 1

        dividends = [
            a for a in actions if a.action_type == CorporateActionType.CASH_DIVIDEND
        ]
        assert dividends
        assert any(d.announced_on is not None for d in dividends), (
            "no dividend carries a declaration date any more, so corporate "
            "actions can no longer be dated by when they became knowable"
        )

    def test_an_unsupported_resolution_is_refused_not_substituted(self, eodhd):
        from app.core.errors import UnsupportedError

        with pytest.raises(UnsupportedError):
            eodhd.fetch_bars(
                ProviderFetchRequest(
                    symbol="AAPL.US",
                    start=dt.datetime(2024, 1, 2, tzinfo=UTC),
                    end=dt.datetime(2024, 1, 5, tzinfo=UTC),
                    resolution="15m",
                )
            )

    def test_the_demo_token_reports_its_own_limits(self, eodhd):
        """A 403 on the demo token must explain itself, not just fail."""
        if not eodhd.is_demo:
            pytest.skip("a real EODHD key is configured, so there is no demo limit")
        from app.core.errors import ProviderError

        with pytest.raises(ProviderError) as exc:
            eodhd.fetch_bars(
                ProviderFetchRequest(
                    symbol="IBM.US",
                    start=dt.datetime(2024, 1, 2, tzinfo=UTC),
                    end=dt.datetime(2024, 1, 5, tzinfo=UTC),
                )
            )
        assert "demo" in str(exc.value).lower()


class TestSecContract:
    def test_the_provider_is_reachable(self, sec):
        status = sec.health_check()
        assert status.healthy, f"SEC health check failed: {status.detail}"

    def test_the_ticker_universe_is_still_served(self, sec):
        universe = sec.fetch_ticker_universe()
        assert len(universe) > 5000, (
            f"the SEC ticker file returned only {len(universe)} rows; the format "
            "may have changed"
        )
        tickers = {t for t, _, _ in universe}
        assert "AAPL" in tickers

    def test_filings_still_carry_exact_acceptance_timestamps(self, sec):
        """The acceptance instant is what ``available_at`` is set from."""
        company = sec.fetch_company(320193)
        assert company.name
        assert company.cik == "0000320193"
        assert company.filings

        annual = [f for f in company.filings if f.form == "10-K"]
        assert annual, "no 10-K in the recent filings window"
        assert any(f.accepted_at is not None for f in annual), (
            "no filing carries an acceptanceDateTime any more, so facts can no "
            "longer be dated to the instant they became public"
        )
        assert annual[0].url and annual[0].url.startswith("https://www.sec.gov/")

    def test_a_restated_period_still_appears_once_per_filing(self, sec):
        """The restatement history is the point-in-time signal.

        If the SEC ever collapsed companyfacts to one value per period, the
        platform could no longer tell what a figure looked like before a
        restatement, and this test is how we would find out.
        """
        facts = sec.fetch_company_facts(320193, concepts=("revenue",))
        assert facts

        by_period: dict[dt.date, list] = {}
        for fact in facts:
            if fact.period_type != "duration":
                continue
            by_period.setdefault(fact.period_end, []).append(fact)

        repeated = [
            (period, group) for period, group in by_period.items() if len(group) > 1
        ]
        assert repeated, (
            "every period is now reported by exactly one filing, so no "
            "restatement history remains in the payload"
        )

        _, group = repeated[0]
        filed_dates = {f.filed_date for f in group}
        assert len(filed_dates) > 1, (
            "a period's repeated facts all share one filing date, so they are "
            "duplicates rather than vintages"
        )

    def test_facts_carry_the_units_the_mapper_expects(self, sec):
        facts = sec.fetch_company_facts(
            320193, concepts=("revenue", "eps_diluted", "shares_diluted")
        )
        units = {f.concept: f.unit for f in facts}
        if "revenue" in units:
            assert units["revenue"] == "USD"
        if "eps_diluted" in units:
            assert units["eps_diluted"].endswith("/shares")
        if "shares_diluted" in units:
            assert units["shares_diluted"] == "shares"


class TestFredContract:
    def test_the_provider_is_reachable(self, fred):
        status = fred.health_check()
        assert status.healthy, f"FRED health check failed: {status.detail}"

    def test_a_daily_series_parses(self, fred):
        observations = fred.fetch_observations(
            "DGS10", start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31)
        )
        assert len(observations) >= 15
        for observation in observations:
            assert observation.unit == "percent"
            assert observation.event_time.tzinfo is not None
            # Availability must never precede the period the figure describes.
            assert observation.available_at >= observation.event_time

    def test_a_missing_observation_is_none_not_zero(self, fred):
        """FRED writes "." for a missing value. It is missing, not zero."""
        # A window containing a US holiday, when the series has no print.
        observations = fred.fetch_observations(
            "DGS10", start=dt.date(2023, 12, 20), end=dt.date(2024, 1, 5)
        )
        assert observations
        values = [o.value for o in observations]
        assert all(v is None or v > 0 for v in values), (
            "a zero appeared in a Treasury yield series, which almost certainly "
            "means a missing value was coerced"
        )

    def test_a_vintage_request_without_a_key_is_refused(self, fred):
        """Serving the latest revision as a vintage would be a look-ahead bug."""
        if fred.has_vintages:
            pytest.skip("a FRED API key is configured, so vintages are available")
        from app.core.errors import UnsupportedError

        with pytest.raises(UnsupportedError, match="look-ahead"):
            fred.fetch_observations("UNRATE", as_of=dt.date(2020, 4, 15))

    def test_the_capability_flag_matches_the_key_state(self, fred):
        assert fred.capabilities.supports_point_in_time == fred.has_vintages
        if not fred.has_vintages:
            assert any("FRED_API_KEY" in c for c in fred.capabilities.caveats)
