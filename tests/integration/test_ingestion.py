"""Ingestion behaviour: idempotency, revisions, validation gating, coverage."""

from __future__ import annotations

import dataclasses
import datetime as dt
import decimal

import pytest
from sqlalchemy import func, select

from app.core.timeutil import UTC
from app.db.models.enums import AdjustmentBasis, DataQualityKind, JobStatus
from app.db.models.instrument import InstrumentCoverage
from app.db.models.market_data import DataGap, DataQualityIssue, PriceBar
from app.ingestion.prices import ingest_bar_batch
from app.pit.availability import AsOf
from app.pit.queries import price_bars_as_of
from app.providers.base import BarBatch, ProviderBar, ProviderInstrument
from tests.conftest import make_bar

pytestmark = pytest.mark.database


def provider_instrument() -> ProviderInstrument:
    return ProviderInstrument(
        provider_symbol="TEST",
        symbol="TEST",
        name="Test Instrument",
        exchange_code="XNYS",
        currency="USD",
        mic="XNYS",
        timezone="America/New_York",
    )


def batch_from(bars: list[ProviderBar], **kwargs) -> BarBatch:
    defaults = {
        "instrument": provider_instrument(),
        "resolution": "1d",
        "adjustment_basis": AdjustmentBasis.TOTAL_RETURN,
        "bars": tuple(bars),
        "availability_lag": dt.timedelta(hours=1),
    }
    defaults.update(kwargs)
    return BarBatch(**defaults)


def clean_bars(days: list[int], *, month: int = 3, closes: list[str] | None = None):
    out = []
    for i, day in enumerate(days):
        close = closes[i] if closes else f"{100 + i}.00"
        out.append(
            make_bar(
                day=dt.date(2024, month, day),
                open_=close,
                high=str(decimal.Decimal(close) + 1),
                low=str(decimal.Decimal(close) - 1),
                close=close,
                adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
            )
        )
    return out


class TestIdempotency:
    def test_reingesting_identical_data_writes_nothing_new(self, db, instrument, source):
        bars = clean_bars([1, 4, 5])
        batch = batch_from(bars)

        first = ingest_bar_batch(db, instrument=instrument, data_source=source, batch=batch)
        assert first.bars_inserted == 3
        assert first.bars_unchanged == 0

        second = ingest_bar_batch(db, instrument=instrument, data_source=source, batch=batch)
        assert second.bars_inserted == 0
        assert second.bars_unchanged == 3, (
            "a retried ingestion must be a no-op, not a new vintage of every bar"
        )

        total = db.execute(
            select(func.count()).select_from(PriceBar).where(
                PriceBar.instrument_id == instrument.id
            )
        ).scalar_one()
        assert total == 3

    def test_repeated_ingestion_does_not_duplicate_volume(self, db, instrument, source):
        bars = clean_bars([1])
        batch = batch_from(bars)
        for _ in range(3):
            ingest_bar_batch(db, instrument=instrument, data_source=source, batch=batch)
        rows = db.execute(
            select(PriceBar).where(PriceBar.instrument_id == instrument.id)
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].volume == decimal.Decimal(1_000_000)


class TestRevisions:
    def test_a_changed_value_creates_a_revision_not_an_update(self, db, instrument, source):
        original = clean_bars([1], closes=["100.00"])
        ingest_bar_batch(
            db, instrument=instrument, data_source=source, batch=batch_from(original)
        )

        corrected = clean_bars([1], closes=["101.25"])
        report = ingest_bar_batch(
            db, instrument=instrument, data_source=source, batch=batch_from(corrected)
        )
        assert report.bars_revised == 1
        assert report.bars_inserted == 0

        rows = db.execute(
            select(PriceBar)
            .where(PriceBar.instrument_id == instrument.id)
            .order_by(PriceBar.revision)
        ).scalars().all()
        assert len(rows) == 2, "the original must be retained, not overwritten"
        assert rows[0].revision == 0 and str(rows[0].close) == "100.00000000"
        assert rows[0].superseded_at is not None
        assert rows[1].revision == 1 and str(rows[1].close) == "101.25000000"
        assert rows[1].superseded_at is None

    def test_a_revision_is_disclosed_in_the_report(self, db, instrument, source):
        ingest_bar_batch(
            db, instrument=instrument, data_source=source,
            batch=batch_from(clean_bars([1], closes=["100.00"])),
        )
        report = ingest_bar_batch(
            db, instrument=instrument, data_source=source,
            batch=batch_from(clean_bars([1], closes=["101.25"])),
        )
        assert any("revised" in note for note in report.notes)

    def test_the_pre_revision_value_stays_queryable(self, db, instrument, source):
        ingest_bar_batch(
            db, instrument=instrument, data_source=source,
            batch=batch_from(clean_bars([1], closes=["100.00"])),
        )
        row = db.execute(
            select(PriceBar).where(PriceBar.instrument_id == instrument.id)
        ).scalars().one()
        before_correction = row.available_at + dt.timedelta(minutes=1)

        ingest_bar_batch(
            db, instrument=instrument, data_source=source,
            batch=batch_from(clean_bars([1], closes=["101.25"])),
        )
        result = price_bars_as_of(db, instrument.id, AsOf(cutoff=before_correction))
        assert len(result.rows) == 1
        assert str(result.rows[0].close) == "100.00000000"


class TestValidationGating:
    def test_a_broken_bar_is_not_stored_but_is_recorded(self, db, instrument, source):
        good = clean_bars([1])[0]
        broken = make_bar(
            day=dt.date(2024, 3, 4), open_="100.00", high="90.00", low="110.00",
            close="100.00", adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
        )
        report = ingest_bar_batch(
            db, instrument=instrument, data_source=source, batch=batch_from([good, broken])
        )
        assert report.bars_inserted == 1
        assert report.bars_rejected == 1
        assert report.status == JobStatus.PARTIAL

        stored = db.execute(
            select(func.count()).select_from(PriceBar).where(
                PriceBar.instrument_id == instrument.id
            )
        ).scalar_one()
        assert stored == 1

        issues = db.execute(
            select(DataQualityIssue).where(
                DataQualityIssue.instrument_id == instrument.id,
                DataQualityIssue.kind == DataQualityKind.OHLC_INCONSISTENT,
            )
        ).scalars().all()
        assert issues, "the defect must be persisted so a report can disclose it"

    def test_a_batch_of_only_broken_bars_fails(self, db, instrument, source):
        broken = make_bar(
            day=dt.date(2024, 3, 4), open_="100.00", high="90.00", low="110.00",
            close="100.00", adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
        )
        report = ingest_bar_batch(
            db, instrument=instrument, data_source=source, batch=batch_from([broken])
        )
        assert report.status == JobStatus.FAILED
        assert report.bars_inserted == 0

    def test_outlier_findings_are_persisted(self, db, instrument, source):
        bars = [
            make_bar(day=dt.date(2024, 3, 1), open_="400.00", high="401.00", low="399.00",
                     close="400.00", adjustment_basis=AdjustmentBasis.TOTAL_RETURN),
            make_bar(day=dt.date(2024, 3, 4), open_="100.00", high="101.00", low="99.00",
                     close="100.00", adjustment_basis=AdjustmentBasis.TOTAL_RETURN),
        ]
        ingest_bar_batch(
            db, instrument=instrument, data_source=source, batch=batch_from(bars)
        )
        issues = db.execute(
            select(DataQualityIssue).where(
                DataQualityIssue.kind == DataQualityKind.OUTLIER_RETURN
            )
        ).scalars().all()
        assert issues
        assert "4:1" in issues[0].detail, "the split hint must survive into storage"


class TestGapRecording:
    def test_missing_sessions_are_recorded_as_gaps(self, db, instrument, source):
        # 2024-03-06 and 03-07 omitted from a Mon-Fri window.
        bars = clean_bars([4, 5, 8])
        batch = batch_from(
            bars,
            requested_start=dt.datetime(2024, 3, 4, tzinfo=UTC),
            requested_end=dt.datetime(2024, 3, 8, 23, 59, tzinfo=UTC),
        )
        report = ingest_bar_batch(
            db, instrument=instrument, data_source=source, batch=batch
        )
        assert report.gaps_recorded == 1
        gap = db.execute(
            select(DataGap).where(DataGap.instrument_id == instrument.id)
        ).scalars().one()
        assert gap.expected_sessions == 2

    def test_gaps_are_not_duplicated_on_reingest(self, db, instrument, source):
        batch = batch_from(
            clean_bars([4, 5, 8]),
            requested_start=dt.datetime(2024, 3, 4, tzinfo=UTC),
            requested_end=dt.datetime(2024, 3, 8, 23, 59, tzinfo=UTC),
        )
        ingest_bar_batch(db, instrument=instrument, data_source=source, batch=batch)
        ingest_bar_batch(db, instrument=instrument, data_source=source, batch=batch)
        count = db.execute(
            select(func.count()).select_from(DataGap).where(
                DataGap.instrument_id == instrument.id
            )
        ).scalar_one()
        assert count == 1

    def test_nothing_is_interpolated_across_a_gap(self, db, instrument, source):
        batch = batch_from(
            clean_bars([4, 5, 8]),
            requested_start=dt.datetime(2024, 3, 4, tzinfo=UTC),
            requested_end=dt.datetime(2024, 3, 8, 23, 59, tzinfo=UTC),
        )
        ingest_bar_batch(db, instrument=instrument, data_source=source, batch=batch)
        stored = db.execute(
            select(PriceBar.event_time).where(PriceBar.instrument_id == instrument.id)
        ).scalars().all()
        assert len(stored) == 3, "a gap is reported, never filled"


class TestAdjustmentBasisIsolation:
    def test_raw_and_adjusted_series_coexist(self, db, instrument, source):
        raw = [
            make_bar(day=dt.date(2024, 3, 1), close="400.00", open_="400.00",
                     high="401.00", low="399.00", adjustment_basis=AdjustmentBasis.RAW)
        ]
        adjusted = [
            make_bar(day=dt.date(2024, 3, 1), close="100.00", open_="100.00",
                     high="100.25", low="99.75",
                     adjustment_basis=AdjustmentBasis.TOTAL_RETURN)
        ]
        ingest_bar_batch(
            db, instrument=instrument, data_source=source,
            batch=batch_from(raw, adjustment_basis=AdjustmentBasis.RAW),
        )
        ingest_bar_batch(
            db, instrument=instrument, data_source=source,
            batch=batch_from(adjusted, adjustment_basis=AdjustmentBasis.TOTAL_RETURN),
        )
        rows = db.execute(
            select(PriceBar).where(PriceBar.instrument_id == instrument.id)
        ).scalars().all()
        assert len(rows) == 2, (
            "a raw and an adjusted bar for the same instant are different "
            "observations and must not overwrite each other"
        )
        bases = {r.adjustment_basis for r in rows}
        assert bases == {AdjustmentBasis.RAW, AdjustmentBasis.TOTAL_RETURN}

    def test_a_query_returns_only_the_requested_basis(self, db, instrument, source):
        ingest_bar_batch(
            db, instrument=instrument, data_source=source,
            batch=batch_from(
                [make_bar(day=dt.date(2024, 3, 1), close="400.00", open_="400.00",
                          high="401.00", low="399.00",
                          adjustment_basis=AdjustmentBasis.RAW)],
                adjustment_basis=AdjustmentBasis.RAW,
            ),
        )
        ingest_bar_batch(
            db, instrument=instrument, data_source=source,
            batch=batch_from(
                [make_bar(day=dt.date(2024, 3, 1), close="100.00", open_="100.00",
                          high="100.25", low="99.75",
                          adjustment_basis=AdjustmentBasis.TOTAL_RETURN)],
                adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
            ),
        )
        as_of = AsOf(cutoff=dt.datetime(2024, 3, 10, tzinfo=UTC))
        raw_rows = price_bars_as_of(
            db, instrument.id, as_of, adjustment_basis=AdjustmentBasis.RAW
        ).rows
        assert [str(r.close) for r in raw_rows] == ["400.00000000"]


class TestVolumeSemantics:
    def test_null_volume_is_stored_as_null_not_zero(self, db, instrument, source):
        bar = make_bar(
            day=dt.date(2024, 3, 1), volume=None,
            adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
        )
        ingest_bar_batch(
            db, instrument=instrument, data_source=source, batch=batch_from([bar])
        )
        row = db.execute(
            select(PriceBar).where(PriceBar.instrument_id == instrument.id)
        ).scalars().one()
        assert row.volume is None, (
            "'no reported volume' must not be coerced to 'zero shares traded'"
        )

    def test_zero_volume_is_stored_as_zero(self, db, instrument, source):
        bar = make_bar(
            day=dt.date(2024, 3, 1), volume="0", open_="100.00", high="100.00",
            low="100.00", close="100.00", adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
        )
        ingest_bar_batch(
            db, instrument=instrument, data_source=source, batch=batch_from([bar])
        )
        row = db.execute(
            select(PriceBar).where(PriceBar.instrument_id == instrument.id)
        ).scalars().one()
        assert row.volume == 0


class TestCoverageTracking:
    def test_coverage_is_recorded_after_ingestion(self, db, instrument, source):
        ingest_bar_batch(
            db, instrument=instrument, data_source=source, batch=batch_from(clean_bars([1, 4, 5]))
        )
        coverage = db.execute(
            select(InstrumentCoverage).where(
                InstrumentCoverage.instrument_id == instrument.id,
                InstrumentCoverage.data_kind == "prices",
            )
        ).scalars().one()
        assert coverage.observation_count == 3
        assert coverage.first_observation is not None
        assert coverage.last_observation is not None
        assert coverage.resolution == "1d"


class TestPrecision:
    def test_prices_round_trip_exactly(self, db, instrument, source):
        """A price must survive storage without binary-float drift."""
        bar = make_bar(
            day=dt.date(2024, 3, 1), open_="182.63", high="184.95", low="182.11",
            close="183.79", adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
        )
        ingest_bar_batch(
            db, instrument=instrument, data_source=source, batch=batch_from([bar])
        )
        row = db.execute(
            select(PriceBar).where(PriceBar.instrument_id == instrument.id)
        ).scalars().one()
        assert row.close == decimal.Decimal("183.79")
        assert row.open == decimal.Decimal("182.63")
