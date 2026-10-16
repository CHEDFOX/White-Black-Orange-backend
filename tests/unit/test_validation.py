"""Validation suite tests.

These exercise the validators against deliberately broken input, because a
validator only ever shown clean data is untested.
"""

from __future__ import annotations

import datetime as dt
import decimal

from app.core.timeutil import UTC
from app.db.models.enums import DataQualityKind, DataQualitySeverity
from app.ingestion.validation import (
    check_bar_structure,
    check_timestamp_alignment,
    check_zero_volume,
    find_duplicate_timestamps,
    find_outlier_returns,
    find_session_gaps,
    reconcile_bars,
    validate_bars,
)
from app.providers.base import ProviderFetchRequest
from tests.conftest import make_bar


class TestBarStructure:
    def test_clean_bar_passes(self):
        assert check_bar_structure(make_bar(day=dt.date(2023, 3, 1))) is None

    def test_high_below_low_is_fatal(self):
        bar = make_bar(day=dt.date(2023, 3, 1), high="98.00", low="101.00")
        finding = check_bar_structure(bar)
        assert finding is not None
        assert finding.kind == DataQualityKind.OHLC_INCONSISTENT
        assert finding.is_fatal

    def test_negative_price_is_fatal(self):
        bar = make_bar(day=dt.date(2023, 3, 1), low="-1.00")
        finding = check_bar_structure(bar)
        assert finding is not None
        assert finding.kind == DataQualityKind.NEGATIVE_PRICE
        assert finding.is_fatal

    def test_zero_price_is_rejected(self):
        # A zero price is not a cheap stock, it is a missing value that someone
        # defaulted. It must never enter a return calculation.
        bar = make_bar(day=dt.date(2023, 3, 1), low="0.00")
        finding = check_bar_structure(bar)
        assert finding is not None
        assert finding.kind == DataQualityKind.NEGATIVE_PRICE

    def test_close_outside_range_is_fatal(self):
        bar = make_bar(day=dt.date(2023, 3, 1), high="101.00", low="99.00", close="105.00")
        finding = check_bar_structure(bar)
        assert finding is not None
        assert "outside the bar range" in finding.detail

    def test_open_outside_range_is_fatal(self):
        bar = make_bar(day=dt.date(2023, 3, 1), open_="90.00", high="101.00", low="99.00")
        finding = check_bar_structure(bar)
        assert finding is not None
        assert finding.is_fatal

    def test_null_volume_is_allowed(self):
        # "No reported volume" is a legitimate observation; only a negative
        # number is structurally wrong.
        bar = make_bar(day=dt.date(2023, 3, 1), volume=None)
        assert check_bar_structure(bar) is None

    def test_negative_volume_is_fatal(self):
        bar = make_bar(day=dt.date(2023, 3, 1), volume="-5")
        finding = check_bar_structure(bar)
        assert finding is not None
        assert finding.is_fatal


class TestZeroVolume:
    def test_zero_volume_flat_bar_is_informational(self):
        bar = make_bar(
            day=dt.date(2023, 3, 1), open_="100.00", high="100.00", low="100.00",
            close="100.00", volume="0",
        )
        finding = check_zero_volume(bar)
        assert finding is not None
        assert finding.severity == DataQualitySeverity.INFO

    def test_zero_volume_with_range_is_a_warning(self):
        bar = make_bar(day=dt.date(2023, 3, 1), volume="0")
        finding = check_zero_volume(bar)
        assert finding is not None
        assert finding.severity == DataQualitySeverity.WARNING

    def test_null_volume_is_not_zero_volume(self):
        assert check_zero_volume(make_bar(day=dt.date(2023, 3, 1), volume=None)) is None


class TestDuplicates:
    def test_identical_duplicate_is_informational(self):
        bar = make_bar(day=dt.date(2023, 3, 1))
        findings = find_duplicate_timestamps([bar, bar])
        assert len(findings) == 1
        assert findings[0].severity == DataQualitySeverity.INFO

    def test_conflicting_duplicate_is_an_error(self):
        a = make_bar(day=dt.date(2023, 3, 1), close="100.50")
        b = make_bar(day=dt.date(2023, 3, 1), close="102.75")
        findings = find_duplicate_timestamps([a, b])
        assert len(findings) == 1
        assert findings[0].is_fatal
        assert "100.50" in findings[0].detail and "102.75" in findings[0].detail

    def test_no_duplicates_in_clean_series(self):
        bars = [make_bar(day=dt.date(2023, 3, d)) for d in (1, 2, 3)]
        assert find_duplicate_timestamps(bars) == []


class TestOutliers:
    def test_split_shaped_drop_is_flagged_with_a_split_hint(self):
        # An unadjusted 4:1 split looks like a 75% loss. The message must say
        # so, because treating it as volatility corrupts every downstream
        # volatility and drawdown estimate.
        bars = [
            make_bar(day=dt.date(2023, 3, 1), open_="400.00", high="405.00", low="395.00", close="400.00"),
            make_bar(day=dt.date(2023, 3, 2), open_="100.00", high="102.00", low="99.00", close="100.00"),
        ]
        findings = find_outlier_returns(bars)
        assert len(findings) == 1
        assert findings[0].kind == DataQualityKind.OUTLIER_RETURN
        assert "4:1" in findings[0].detail

    def test_normal_volatility_is_not_flagged(self):
        bars = [
            make_bar(day=dt.date(2023, 3, 1), close="100.00"),
            make_bar(day=dt.date(2023, 3, 2), open_="100.00", high="106.00", low="99.00", close="105.00"),
        ]
        assert find_outlier_returns(bars) == []

    def test_outlier_threshold_is_configurable(self):
        bars = [
            make_bar(day=dt.date(2023, 3, 1), close="100.00"),
            make_bar(day=dt.date(2023, 3, 2), open_="100.00", high="112.00", low="99.00", close="111.00"),
        ]
        assert find_outlier_returns(bars, threshold=0.40) == []
        assert len(find_outlier_returns(bars, threshold=0.05)) == 1


class TestSessionGaps:
    def test_missing_sessions_are_found_and_merged(self, calendar):
        days = [dt.date(2023, 5, d) for d in (10, 11, 12, 17, 18, 19)]
        bars = [make_bar(day=d) for d in days]
        gaps = find_session_gaps(
            bars,
            calendar,
            start=dt.datetime(2023, 5, 10, tzinfo=UTC),
            end=dt.datetime(2023, 5, 19, 23, 59, tzinfo=UTC),
        )
        assert len(gaps) == 1, "15th and 16th are consecutive and must merge into one gap"
        assert gaps[0].expected_sessions == 2
        assert gaps[0].missing_dates == (dt.date(2023, 5, 15), dt.date(2023, 5, 16))

    def test_weekends_are_not_gaps(self, calendar):
        # 2023-05-12 is a Friday, 2023-05-15 the following Monday.
        bars = [make_bar(day=dt.date(2023, 5, 12)), make_bar(day=dt.date(2023, 5, 15))]
        gaps = find_session_gaps(
            bars,
            calendar,
            start=dt.datetime(2023, 5, 12, tzinfo=UTC),
            end=dt.datetime(2023, 5, 15, 23, 59, tzinfo=UTC),
        )
        assert gaps == []

    def test_utc_window_start_does_not_pull_in_the_previous_session(self, calendar):
        """Regression: a UTC midnight boundary must not reach back a session.

        00:00 UTC on 2023-05-10 is 20:00 on 2023-05-09 in New York. Deriving
        the window from local dates therefore reported 2023-05-09 as a missing
        session even though the caller's window began on the 10th.
        """
        days = [dt.date(2023, 5, d) for d in (10, 11, 12, 15, 16, 17, 18, 19)]
        bars = [make_bar(day=d) for d in days]
        gaps = find_session_gaps(
            bars,
            calendar,
            start=dt.datetime(2023, 5, 10, tzinfo=UTC),
            end=dt.datetime(2023, 5, 19, 23, 59, tzinfo=UTC),
        )
        assert gaps == [], "every session in the window has a bar"

    def test_window_selects_sessions_by_instant_not_local_date(self, calendar):
        from app.ingestion.validation import expected_sessions_in_window

        sessions = expected_sessions_in_window(
            calendar,
            dt.datetime(2023, 5, 10, tzinfo=UTC),
            dt.datetime(2023, 5, 19, 23, 59, tzinfo=UTC),
        )
        assert sessions[0] == dt.date(2023, 5, 10)
        assert sessions[-1] == dt.date(2023, 5, 19)

    def test_exchange_holiday_is_not_a_gap(self, calendar):
        # 2023-07-04 is Independence Day; the NYSE calendar must not report it
        # as missing data.
        assert not calendar.is_trading_day(dt.date(2023, 7, 4))
        bars = [make_bar(day=dt.date(2023, 7, 3)), make_bar(day=dt.date(2023, 7, 5))]
        gaps = find_session_gaps(
            bars,
            calendar,
            start=dt.datetime(2023, 7, 3, tzinfo=UTC),
            end=dt.datetime(2023, 7, 5, 23, 59, tzinfo=UTC),
        )
        assert gaps == []


class TestTimestampAlignment:
    def test_aligned_daily_bars_pass(self, calendar):
        bars = [make_bar(day=dt.date(2023, 3, d)) for d in (1, 2, 3)]
        findings = check_timestamp_alignment(bars, "1d", calendar)
        assert [f for f in findings if f.severity != DataQualitySeverity.INFO] == []

    def test_weekend_dated_daily_bar_is_flagged(self, calendar):
        # 2023-03-04 is a Saturday.
        bar = make_bar(day=dt.date(2023, 3, 4))
        findings = check_timestamp_alignment([bar], "1d", calendar)
        assert any(f.kind == DataQualityKind.TIMESTAMP_MISALIGNED for f in findings)

    def test_misaligned_intraday_bar_is_flagged(self, calendar):
        bar = make_bar(day=dt.date(2023, 3, 1), resolution="5m")
        off = bar.event_time + dt.timedelta(seconds=37)
        shifted = type(bar)(
            event_time=off,
            bar_close_time=off + dt.timedelta(minutes=5),
            open=bar.open, high=bar.high, low=bar.low, close=bar.close,
            volume=bar.volume, currency=bar.currency, resolution="5m",
            adjustment_basis=bar.adjustment_basis,
        )
        findings = check_timestamp_alignment([shifted], "5m", calendar)
        assert any(f.kind == DataQualityKind.TIMESTAMP_MISALIGNED for f in findings)


class TestValidateBars:
    def test_broken_bars_are_excluded_but_recorded(self, calendar):
        good = make_bar(day=dt.date(2023, 3, 1))
        broken = make_bar(day=dt.date(2023, 3, 2), high="90.00", low="110.00")
        result = validate_bars(
            [good, broken], resolution="1d", calendar=calendar
        )
        assert result.accepted == [good], "a structurally broken bar must not be stored"
        assert result.rejected_count == 1
        assert result.errors(), "the defect must still be recorded, not silently dropped"

    def test_rejected_bar_does_not_contaminate_outlier_detection(self, calendar):
        # The broken middle bar would create two huge spurious returns if it
        # reached the outlier pass.
        bars = [
            make_bar(day=dt.date(2023, 3, 1), close="100.00"),
            make_bar(day=dt.date(2023, 3, 2), open_="100.00", high="1.00", low="900.00", close="500.00"),
            make_bar(day=dt.date(2023, 3, 3), open_="100.00", high="102.00", low="99.00", close="101.00"),
        ]
        result = validate_bars(bars, resolution="1d", calendar=calendar)
        assert result.accepted_count == 2
        assert [f for f in result.findings if f.kind == DataQualityKind.OUTLIER_RETURN] == []

    def test_exact_duplicates_collapse_to_one_row(self, calendar):
        bar = make_bar(day=dt.date(2023, 3, 1))
        result = validate_bars([bar, bar], resolution="1d", calendar=calendar)
        assert result.accepted_count == 1

    def test_conflicting_duplicates_keep_one_and_reject_the_rest(self, calendar):
        a = make_bar(day=dt.date(2023, 3, 1), close="100.00")
        b = make_bar(day=dt.date(2023, 3, 1), close="200.00")
        result = validate_bars([a, b], resolution="1d", calendar=calendar)
        assert result.accepted_count == 1
        assert result.rejected_count == 1

    def test_gap_reporting_is_opt_in_and_accurate(self, calendar):
        days = [dt.date(2023, 5, d) for d in (10, 11, 12, 17, 18)]
        bars = [make_bar(day=d) for d in days]
        result = validate_bars(
            bars,
            resolution="1d",
            calendar=calendar,
            check_gaps_between=(
                dt.datetime(2023, 5, 10, tzinfo=UTC),
                dt.datetime(2023, 5, 18, 23, 59, tzinfo=UTC),
            ),
        )
        missing = [f for f in result.findings if f.kind == DataQualityKind.MISSING_SESSION]
        assert len(missing) == 1
        assert "2023-05-15" in missing[0].detail

    def test_nothing_is_invented_to_fill_a_gap(self, calendar):
        days = [dt.date(2023, 5, d) for d in (10, 11, 17, 18)]
        bars = [make_bar(day=d) for d in days]
        result = validate_bars(
            bars,
            resolution="1d",
            calendar=calendar,
            check_gaps_between=(
                dt.datetime(2023, 5, 10, tzinfo=UTC),
                dt.datetime(2023, 5, 18, 23, 59, tzinfo=UTC),
            ),
        )
        # The core guarantee: validation reports holes, it does not patch them.
        assert result.accepted_count == 4


class TestFixtureProviderDefects:
    """The fixture provider's injected defects must be caught by the validators."""

    def test_injected_broken_bar_is_rejected(self, fixture_provider, calendar):
        batch = fixture_provider.fetch_bars(
            ProviderFetchRequest(
                symbol="SYNTHBAD",
                start=dt.datetime(2023, 6, 1, tzinfo=UTC),
                end=dt.datetime(2023, 6, 30, tzinfo=UTC),
                resolution="1d",
            )
        )
        result = validate_bars(list(batch.bars), resolution="1d", calendar=calendar)
        assert result.rejected_count >= 1
        kinds = {f.kind for f in result.errors()}
        assert DataQualityKind.OHLC_INCONSISTENT in kinds

    def test_injected_duplicate_is_detected(self, fixture_provider, calendar):
        batch = fixture_provider.fetch_bars(
            ProviderFetchRequest(
                symbol="SYNTHBAD",
                start=dt.datetime(2023, 6, 1, tzinfo=UTC),
                end=dt.datetime(2023, 6, 30, tzinfo=UTC),
                resolution="1d",
            )
        )
        findings = find_duplicate_timestamps(list(batch.bars))
        assert findings, "the fixture injects a duplicate on 2023-06-15"

    def test_injected_gap_is_detected(self, fixture_provider, calendar):
        batch = fixture_provider.fetch_bars(
            ProviderFetchRequest(
                symbol="SYNTHGAP",
                start=dt.datetime(2023, 3, 1, tzinfo=UTC),
                end=dt.datetime(2023, 3, 31, tzinfo=UTC),
                resolution="1d",
            )
        )
        gaps = find_session_gaps(
            list(batch.bars),
            calendar,
            start=dt.datetime(2023, 3, 1, tzinfo=UTC),
            end=dt.datetime(2023, 3, 31, tzinfo=UTC),
        )
        assert len(gaps) == 1
        assert gaps[0].expected_sessions == 3


class TestReconciliation:
    def test_agreeing_providers_produce_no_findings(self):
        bars = [make_bar(day=dt.date(2023, 3, 1))]
        assert reconcile_bars(bars, bars, left_source="a", right_source="b") == []

    def test_disagreement_is_reported_not_resolved(self):
        a = [make_bar(day=dt.date(2023, 3, 1), close="100.00")]
        b = [make_bar(day=dt.date(2023, 3, 1), close="105.00")]
        findings = reconcile_bars(a, b, left_source="eodhd", right_source="yahoo")
        assert findings
        assert findings[0].kind == DataQualityKind.PROVIDER_DISAGREEMENT
        # Crucially: the function reports, it does not pick a winner.
        assert "eodhd" in findings[0].context and "yahoo" in findings[0].context

    def test_tiny_rounding_differences_are_tolerated(self):
        a = [make_bar(day=dt.date(2023, 3, 1), close="100.00")]
        b = [make_bar(day=dt.date(2023, 3, 1), close="100.01")]
        assert reconcile_bars(
            a, b, left_source="a", right_source="b",
            tolerance=decimal.Decimal("0.005"),
        ) == []
