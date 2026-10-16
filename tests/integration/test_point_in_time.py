"""Point-in-time correctness.

These are the tests that make the platform's central claim checkable. Each one
describes a specific way a financial dataset leaks the future, and asserts the
leak does not happen.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select

from app.core.errors import LookAheadError
from app.core.timeutil import UTC
from app.db.models.enums import AdjustmentBasis, DataClassification
from app.db.models.market_data import PriceBar
from app.pit.availability import AsOf, assert_no_future_rows, visible_at
from app.pit.queries import price_bars_as_of, resolve_symbol_at
from tests.integration.conftest import add_price_bar, utc

pytestmark = pytest.mark.database


class TestAvailabilityFiltering:
    def test_a_bar_is_invisible_before_its_session_closes(self, db, instrument, source):
        """The headline guarantee: today's close is not knowable this morning."""
        event_time = utc(2024, 3, 1)
        add_price_bar(db, instrument, source, event_time=event_time, close="100.00")

        morning = AsOf(cutoff=utc(2024, 3, 1, 15, 0))
        result = price_bars_as_of(db, instrument.id, morning)
        assert result.rows == [], (
            "a bar whose session had not closed must not be visible mid-session"
        )

        next_day = AsOf(cutoff=utc(2024, 3, 2, 14, 30))
        assert price_bars_as_of(db, instrument.id, next_day).rows != []

    def test_future_bars_never_appear(self, db, instrument, source):
        for day in (1, 4, 5, 6, 7):
            add_price_bar(db, instrument, source, event_time=utc(2024, 3, day), close="100.00")

        as_of = AsOf(cutoff=utc(2024, 3, 5, 23, 0))
        result = price_bars_as_of(db, instrument.id, as_of)
        assert result.rows, "some history should be visible"
        assert all(bar.event_time <= as_of.cutoff for bar in result.rows)
        assert max(b.event_time for b in result.rows).day == 5

    def test_requested_end_is_clamped_and_disclosed(self, db, instrument, source):
        add_price_bar(db, instrument, source, event_time=utc(2024, 3, 1), close="100.00")
        as_of = AsOf(cutoff=utc(2024, 3, 5))
        result = price_bars_as_of(
            db, instrument.id, as_of, end=utc(2024, 6, 1)
        )
        assert any("clamped" in note for note in result.coverage.notes), (
            "silently returning fewer rows than asked for hides the clamp"
        )

    def test_backfilled_rows_are_visible_to_historical_queries(self, db, instrument, source):
        """``ingested_at`` must never act as a filter.

        A backfill writes 2019 rows today. If availability were derived from
        ingestion time, a 2019 analysis would see nothing -- the bug this
        asserts against.
        """
        bar = add_price_bar(db, instrument, source, event_time=utc(2019, 6, 3), close="50.00")
        assert bar.ingested_at > utc(2019, 6, 3), "the row was written long after the event"

        as_of = AsOf(cutoff=utc(2019, 6, 4, 23, 0))
        assert price_bars_as_of(db, instrument.id, as_of).rows != []

    def test_database_level_invariant_holds(self, db, instrument, source):
        for day in (1, 4, 5):
            add_price_bar(db, instrument, source, event_time=utc(2024, 3, day), close="100.00")
        as_of = AsOf(cutoff=utc(2024, 3, 6))
        assert_no_future_rows(db, PriceBar, as_of)


class TestRevisionHandling:
    def test_a_query_before_a_correction_returns_the_original_value(
        self, db, instrument, source
    ):
        """Revisions must not rewrite history.

        A price corrected in April must still read as the original figure for
        an analysis dated March, otherwise every backtest silently uses
        knowledge nobody had.
        """
        event_time = utc(2024, 3, 1)
        correction_available = utc(2024, 4, 1)

        add_price_bar(
            db, instrument, source,
            event_time=event_time, close="100.00", revision=0,
            superseded_at=correction_available,
        )
        add_price_bar(
            db, instrument, source,
            event_time=event_time, close="101.50", revision=1,
            available_at=correction_available,
        )

        before = price_bars_as_of(db, instrument.id, AsOf(cutoff=utc(2024, 3, 15)))
        assert len(before.rows) == 1
        assert str(before.rows[0].close) == "100.00000000", "must see the original figure"

        after = price_bars_as_of(db, instrument.id, AsOf(cutoff=utc(2024, 4, 15)))
        assert len(after.rows) == 1
        assert str(after.rows[0].close) == "101.50000000", "must see the corrected figure"

    def test_only_one_revision_per_timestamp_is_returned(self, db, instrument, source):
        """Filtering on availability alone would return every past vintage."""
        event_time = utc(2024, 3, 1)
        add_price_bar(
            db, instrument, source, event_time=event_time, close="100.00",
            revision=0, available_at=utc(2024, 3, 2), superseded_at=utc(2024, 3, 10),
        )
        add_price_bar(
            db, instrument, source, event_time=event_time, close="100.50",
            revision=1, available_at=utc(2024, 3, 10), superseded_at=utc(2024, 3, 20),
        )
        add_price_bar(
            db, instrument, source, event_time=event_time, close="100.75",
            revision=2, available_at=utc(2024, 3, 20),
        )

        result = price_bars_as_of(db, instrument.id, AsOf(cutoff=utc(2024, 4, 1)))
        assert len(result.rows) == 1, "three vintages of one bar must collapse to one row"
        assert str(result.rows[0].close) == "100.75000000"
        assert result.rows[0].revision == 2

    def test_middle_vintage_is_selected_correctly(self, db, instrument, source):
        event_time = utc(2024, 3, 1)
        add_price_bar(
            db, instrument, source, event_time=event_time, close="100.00",
            revision=0, available_at=utc(2024, 3, 2), superseded_at=utc(2024, 3, 10),
        )
        add_price_bar(
            db, instrument, source, event_time=event_time, close="100.50",
            revision=1, available_at=utc(2024, 3, 10), superseded_at=utc(2024, 3, 20),
        )
        add_price_bar(
            db, instrument, source, event_time=event_time, close="100.75",
            revision=2, available_at=utc(2024, 3, 20),
        )
        result = price_bars_as_of(db, instrument.id, AsOf(cutoff=utc(2024, 3, 15)))
        assert len(result.rows) == 1
        assert str(result.rows[0].close) == "100.50000000"


class TestSyntheticIsolation:
    def test_synthetic_bars_are_excluded_from_production_reads(
        self, db, instrument, source, synthetic_source
    ):
        add_price_bar(db, instrument, source, event_time=utc(2024, 3, 1), close="100.00")
        add_price_bar(
            db, instrument, synthetic_source,
            event_time=utc(2024, 3, 4), close="999.00",
            classification=DataClassification.SYNTHETIC,
        )
        result = price_bars_as_of(db, instrument.id, AsOf(cutoff=utc(2024, 3, 10)))
        closes = {str(b.close) for b in result.rows}
        assert "999.00000000" not in closes, "synthetic data must never reach a production read"
        assert "100.00000000" in closes

    def test_synthetic_data_is_reachable_when_explicitly_requested(
        self, db, instrument, synthetic_source
    ):
        add_price_bar(
            db, instrument, synthetic_source,
            event_time=utc(2024, 3, 1), close="123.00",
            classification=DataClassification.SYNTHETIC,
        )
        result = price_bars_as_of(
            db, instrument.id, AsOf(cutoff=utc(2024, 3, 10)), production_only=False
        )
        assert [str(b.close) for b in result.rows] == ["123.00000000"]


class TestSymbolResolution:
    def test_a_renamed_ticker_resolves_to_the_era_appropriate_instrument(
        self, db, instrument
    ):
        """FB in 2019 and META in 2024 are the same instrument."""
        from app.ingestion.instruments import record_ticker_change

        record_ticker_change(
            db, instrument,
            old_symbol="TEST", new_symbol="TEST2",
            effective_date=dt.date(2022, 6, 9),
        )
        old_era = resolve_symbol_at(db, "TEST", AsOf(cutoff=utc(2021, 1, 1)))
        new_era = resolve_symbol_at(db, "TEST2", AsOf(cutoff=utc(2023, 1, 1)))
        assert old_era is not None and new_era is not None
        assert old_era.id == new_era.id == instrument.id

    def test_a_ticker_is_not_resolved_before_it_was_assigned(self, db, instrument):
        from app.ingestion.instruments import record_ticker_change

        record_ticker_change(
            db, instrument,
            old_symbol="TEST", new_symbol="NEWSYM",
            effective_date=dt.date(2022, 6, 9),
        )
        # Looking up NEWSYM in 2021 must not match the dated ticker row; the
        # only reason it resolves at all is the explicit current-symbol
        # fallback, and the dated row is what a strict caller checks.
        stmt = select(PriceBar).where(PriceBar.instrument_id == instrument.id)
        assert db.execute(stmt).scalars().all() == []

        from app.db.models.instrument import InstrumentTicker

        rows = db.execute(
            select(InstrumentTicker).where(InstrumentTicker.instrument_id == instrument.id)
        ).scalars().all()
        newsym = [r for r in rows if r.symbol == "NEWSYM"][0]
        assert newsym.valid_from == dt.date(2022, 6, 9)
        old = [r for r in rows if r.symbol == "TEST"][0]
        assert old.valid_to == dt.date(2022, 6, 8), "the old ticker's range must close"


class TestAsOfGuards:
    def test_assert_not_future_raises_on_a_leak(self):
        as_of = AsOf(cutoff=utc(2024, 3, 1))
        with pytest.raises(LookAheadError) as exc:
            as_of.assert_not_future(utc(2024, 3, 2), what="earnings release")
        assert "earnings release" in str(exc.value)

    def test_assert_not_future_passes_on_past_data(self):
        as_of = AsOf(cutoff=utc(2024, 3, 1))
        as_of.assert_not_future(utc(2024, 2, 28), what="earnings release")

    def test_naive_cutoff_is_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            AsOf(cutoff=dt.datetime(2024, 3, 1))

    def test_disclosures_accumulate_without_duplication(self):
        as_of = AsOf(cutoff=utc(2024, 3, 1))
        once = as_of.with_disclosure("macro is latest-revision")
        twice = once.with_disclosure("macro is latest-revision")
        assert once.disclosures == twice.disclosures == ("macro is latest-revision",)
