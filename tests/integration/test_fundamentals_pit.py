"""Point-in-time fundamentals: period identity and restatement handling.

These tests encode two bugs that reached a running system before being caught,
so that neither can return:

1. SEC's ``fy``/``fp`` fields describe the *filing*, not the period. FY2021
   revenue restated in the FY2023 10-K arrives tagged ``fy=2023, fp=FY``.
   Keying a period on those fields filed a 2021 figure under 2023.
2. A 10-Q reports a discrete quarter *and* a year-to-date cumulative figure
   under the same ``fp``. Keying on ``(fiscal_year, fiscal_period)`` merged a
   90-day quarter with a 272-day nine-month total into one period, so a
   "quarterly revenue" could silently be a nine-month figure.
"""

from __future__ import annotations

import datetime as dt
import decimal

import pytest
from sqlalchemy import select

from app.core.timeutil import UTC
from app.db.models.fundamentals import FinancialFact, FiscalPeriod
from app.ingestion.fundamentals import (
    ingest_facts,
    period_kind_for,
    supersede_older_revisions,
)
from app.pit.availability import AsOf
from app.pit.queries import financial_facts_as_of
from app.providers.sec_edgar import SecFact
from app.services.data_access import load_fundamentals

pytestmark = pytest.mark.database


def fact(
    concept: str,
    value: str,
    *,
    period_start: dt.date | None,
    period_end: dt.date,
    filed: dt.date,
    accession: str,
    fy: int | None = None,
    fp: str | None = "FY",
    unit: str = "USD",
    form: str = "10-K",
) -> SecFact:
    return SecFact(
        concept=concept,
        source_concept=f"Test{concept}",
        value=decimal.Decimal(value),
        unit=unit,
        period_start=period_start,
        period_end=period_end,
        period_type="duration" if period_start else "instant",
        fiscal_year=fy,
        fiscal_period=fp,
        form=form,
        accession=accession,
        filed_date=filed,
        accepted_at=dt.datetime.combine(filed, dt.time(21, 0), tzinfo=UTC),
    )


class TestPeriodKindClassification:
    def test_an_annual_duration_is_classified_fy(self):
        kind, days = period_kind_for(dt.date(2022, 9, 25), dt.date(2023, 9, 30))
        assert kind == "FY"
        assert days == 370

    def test_a_discrete_quarter_is_classified_q(self):
        kind, days = period_kind_for(dt.date(2023, 10, 1), dt.date(2023, 12, 30))
        assert kind == "Q"
        assert days == 90

    def test_a_year_to_date_half_is_not_a_quarter(self):
        """The collision that corrupted quarterly figures."""
        kind, days = period_kind_for(dt.date(2023, 10, 1), dt.date(2024, 3, 30))
        assert kind == "H"
        assert days == 181

    def test_a_nine_month_cumulative_is_not_a_quarter(self):
        kind, days = period_kind_for(dt.date(2023, 10, 1), dt.date(2024, 6, 29))
        assert kind == "9M"
        assert days == 272

    def test_an_instant_fact_has_no_duration(self):
        kind, days = period_kind_for(None, dt.date(2023, 9, 30))
        assert kind == "PIT"
        assert days is None


class TestPeriodIdentity:
    def test_a_restated_fact_files_under_its_own_period_not_the_filings(self, db, issuer, source):
        """The headline bug: a 2021 figure must not be filed under 2023."""
        facts = [
            # FY2021 as first reported, tagged fy=2021.
            fact("revenue", "365817000000", period_start=dt.date(2020, 9, 27),
                 period_end=dt.date(2021, 9, 25), filed=dt.date(2021, 10, 29),
                 accession="0000320193-21-000105", fy=2021),
            # The same FY2021 period repeated in the FY2023 10-K, tagged fy=2023.
            fact("revenue", "365817000000", period_start=dt.date(2020, 9, 27),
                 period_end=dt.date(2021, 9, 25), filed=dt.date(2023, 11, 3),
                 accession="0000320193-23-000106", fy=2023),
            # FY2023 itself, also tagged fy=2023.
            fact("revenue", "383285000000", period_start=dt.date(2022, 9, 25),
                 period_end=dt.date(2023, 9, 30), filed=dt.date(2023, 11, 3),
                 accession="0000320193-23-000106", fy=2023),
        ]
        ingest_facts(db, issuer=issuer, facts=facts, data_source=source)

        periods = db.execute(
            select(FiscalPeriod).where(FiscalPeriod.issuer_id == issuer.id)
            .order_by(FiscalPeriod.period_end)
        ).scalars().all()
        assert len(periods) == 2, (
            "two distinct annual periods, not three facts collapsed by filing year"
        )
        assert [p.period_end for p in periods] == [
            dt.date(2021, 9, 25), dt.date(2023, 9, 30)
        ]
        assert all(p.period_kind == "FY" for p in periods)

        # The FY2021 period carries both filings' versions of its figure.
        fy2021 = periods[0]
        rows = db.execute(
            select(FinancialFact).where(FinancialFact.fiscal_period_id == fy2021.id)
        ).scalars().all()
        assert len(rows) == 2, "the restatement history must be preserved"
        assert {r.revision for r in rows} == {0, 1}

    def test_a_quarter_and_a_cumulative_period_do_not_collide(self, db, issuer, source):
        """A 90-day quarter and a 272-day cumulative must stay separate."""
        facts = [
            fact("revenue", "119575000000", period_start=dt.date(2023, 10, 1),
                 period_end=dt.date(2023, 12, 30), filed=dt.date(2024, 2, 2),
                 accession="q1", fy=2024, fp="Q1", form="10-Q"),
            fact("revenue", "210328000000", period_start=dt.date(2023, 10, 1),
                 period_end=dt.date(2024, 3, 30), filed=dt.date(2024, 5, 3),
                 accession="q2", fy=2024, fp="Q2", form="10-Q"),
            fact("revenue", "296098000000", period_start=dt.date(2023, 10, 1),
                 period_end=dt.date(2024, 6, 29), filed=dt.date(2024, 8, 2),
                 accession="q3", fy=2024, fp="Q3", form="10-Q"),
        ]
        ingest_facts(db, issuer=issuer, facts=facts, data_source=source)
        periods = db.execute(
            select(FiscalPeriod).where(FiscalPeriod.issuer_id == issuer.id)
            .order_by(FiscalPeriod.period_end)
        ).scalars().all()
        assert len(periods) == 3
        assert [p.period_kind for p in periods] == ["Q", "H", "9M"]
        assert [p.duration_days for p in periods] == [90, 181, 272]

    def test_a_balance_sheet_fact_joins_the_flow_period_sharing_its_end_date(
        self, db, issuer, source
    ):
        """Otherwise ROE divides a full-year profit by another period's balance."""
        facts = [
            fact("revenue", "383285000000", period_start=dt.date(2022, 9, 25),
                 period_end=dt.date(2023, 9, 30), filed=dt.date(2023, 11, 3),
                 accession="fy23"),
            fact("net_income", "96995000000", period_start=dt.date(2022, 9, 25),
                 period_end=dt.date(2023, 9, 30), filed=dt.date(2023, 11, 3),
                 accession="fy23"),
            # Instant: no period_start.
            fact("stockholders_equity", "62146000000", period_start=None,
                 period_end=dt.date(2023, 9, 30), filed=dt.date(2023, 11, 3),
                 accession="fy23"),
        ]
        ingest_facts(db, issuer=issuer, facts=facts, data_source=source)
        periods = db.execute(
            select(FiscalPeriod).where(FiscalPeriod.issuer_id == issuer.id)
        ).scalars().all()
        assert len(periods) == 1, "the balance sheet must not create its own period"

        rows = db.execute(
            select(FinancialFact).where(FinancialFact.fiscal_period_id == periods[0].id)
        ).scalars().all()
        assert {r.concept for r in rows} == {
            "revenue", "net_income", "stockholders_equity"
        }

    def test_an_instant_fact_with_no_matching_period_is_skipped_not_misfiled(
        self, db, issuer, source
    ):
        facts = [
            fact("stockholders_equity", "50000000000", period_start=None,
                 period_end=dt.date(2019, 3, 31), filed=dt.date(2019, 5, 1),
                 accession="orphan"),
        ]
        written, _ = ingest_facts(db, issuer=issuer, facts=facts, data_source=source)
        assert written == 0, (
            "an orphan balance-sheet fact must not be attached to an unrelated period"
        )


class TestRestatementPointInTime:
    def _ingest_restated(self, db, issuer, source):
        facts = [
            fact("revenue", "274515000000", period_start=dt.date(2019, 9, 29),
                 period_end=dt.date(2020, 9, 26), filed=dt.date(2020, 10, 30),
                 accession="fy20"),
            fact("revenue", "365817000000", period_start=dt.date(2020, 9, 27),
                 period_end=dt.date(2021, 9, 25), filed=dt.date(2021, 10, 29),
                 accession="fy21"),
            # FY2021 restated with a different number in a later filing.
            fact("revenue", "365900000000", period_start=dt.date(2020, 9, 27),
                 period_end=dt.date(2021, 9, 25), filed=dt.date(2023, 11, 3),
                 accession="fy23"),
        ]
        ingest_facts(db, issuer=issuer, facts=facts, data_source=source)
        supersede_older_revisions(db, issuer.id)

    def test_a_cutoff_before_the_restatement_sees_the_original_figure(
        self, db, issuer, source
    ):
        self._ingest_restated(db, issuer, source)
        result = financial_facts_as_of(
            db, issuer.id, AsOf(cutoff=dt.datetime(2022, 6, 30, tzinfo=UTC)),
            concepts=("revenue",),
        )
        values = {str(r.value) for r in result.rows}
        assert "365817000000.0000000000" in values, "the as-first-reported figure"
        assert "365900000000.0000000000" not in values, (
            "a restatement filed in 2023 must be invisible to a 2022 analysis"
        )

    def test_a_cutoff_after_the_restatement_sees_the_revised_figure(
        self, db, issuer, source
    ):
        self._ingest_restated(db, issuer, source)
        result = financial_facts_as_of(
            db, issuer.id, AsOf(cutoff=dt.datetime(2024, 6, 30, tzinfo=UTC)),
            concepts=("revenue",),
        )
        fy2021 = [
            r for r in result.rows
            if r.event_time.date() == dt.date(2021, 9, 25)
        ]
        assert len(fy2021) == 1, "one revision per period, the one in force"
        assert str(fy2021[0].value) == "365900000000.0000000000"

    def test_each_period_contributes_exactly_one_row_at_any_cutoff(
        self, db, issuer, source
    ):
        self._ingest_restated(db, issuer, source)
        for year in (2021, 2022, 2023, 2024):
            result = financial_facts_as_of(
                db, issuer.id, AsOf(cutoff=dt.datetime(year, 6, 30, tzinfo=UTC)),
                concepts=("revenue",),
            )
            ends = [r.event_time.date() for r in result.rows]
            assert len(ends) == len(set(ends)), (
                f"as of {year} a period appears more than once, so a vintage leaked"
            )


class TestLoadFundamentals:
    def test_the_latest_annual_period_is_selected_per_cutoff(
        self, db, instrument, issuer, source
    ):
        facts = []
        for end_year, revenue in (
            (2020, "274515000000"), (2021, "365817000000"),
            (2022, "394328000000"), (2023, "383285000000"),
        ):
            filed = dt.date(end_year, 10, 30)
            facts.append(
                fact("revenue", revenue,
                     period_start=dt.date(end_year - 1, 9, 29),
                     period_end=dt.date(end_year, 9, 26),
                     filed=filed, accession=f"fy{end_year}")
            )
        ingest_facts(db, issuer=issuer, facts=facts, data_source=source)
        supersede_older_revisions(db, issuer.id)

        expected = {
            2021: 274515000000.0,
            2022: 365817000000.0,
            2023: 394328000000.0,
            2024: 383285000000.0,
        }
        for year, revenue in expected.items():
            snapshot = load_fundamentals(
                db, instrument, as_of=AsOf(cutoff=dt.datetime(year, 6, 30, tzinfo=UTC))
            )
            assert snapshot.scalars["revenue"] == pytest.approx(revenue), (
                f"as of {year}-06-30 the then-latest annual revenue should be {revenue:,.0f}"
            )

    def test_quarterly_periods_are_excluded_from_an_annual_snapshot(
        self, db, instrument, issuer, source
    ):
        facts = [
            fact("revenue", "383285000000", period_start=dt.date(2022, 9, 25),
                 period_end=dt.date(2023, 9, 30), filed=dt.date(2023, 11, 3),
                 accession="fy23"),
            # A quarter ending later would otherwise win the "latest period" race.
            fact("revenue", "119575000000", period_start=dt.date(2023, 10, 1),
                 period_end=dt.date(2023, 12, 30), filed=dt.date(2024, 2, 2),
                 accession="q1", fp="Q1", form="10-Q"),
        ]
        ingest_facts(db, issuer=issuer, facts=facts, data_source=source)
        supersede_older_revisions(db, issuer.id)
        snapshot = load_fundamentals(
            db, instrument, as_of=AsOf(cutoff=dt.datetime(2024, 6, 30, tzinfo=UTC)),
            annual_only=True,
        )
        assert snapshot.scalars["revenue"] == pytest.approx(383285000000.0), (
            "an annual snapshot must not pick up a quarterly figure"
        )

    def test_a_growth_series_only_uses_periods_present_in_every_concept(
        self, db, instrument, issuer, source
    ):
        facts = [
            fact("revenue", "100", period_start=dt.date(2021, 1, 1),
                 period_end=dt.date(2021, 12, 31), filed=dt.date(2022, 2, 1),
                 accession="a"),
            fact("revenue", "120", period_start=dt.date(2022, 1, 1),
                 period_end=dt.date(2022, 12, 31), filed=dt.date(2023, 2, 1),
                 accession="b"),
            # net_income exists for only one of the two years.
            fact("net_income", "10", period_start=dt.date(2022, 1, 1),
                 period_end=dt.date(2022, 12, 31), filed=dt.date(2023, 2, 1),
                 accession="b"),
        ]
        ingest_facts(db, issuer=issuer, facts=facts, data_source=source)
        supersede_older_revisions(db, issuer.id)
        snapshot = load_fundamentals(
            db, instrument, as_of=AsOf(cutoff=dt.datetime(2024, 1, 1, tzinfo=UTC))
        )
        assert "revenue_by_period" in snapshot.series
        assert "net_income_by_period" not in snapshot.series, (
            "a series with a hole would compare non-adjacent periods as if adjacent"
        )

    def test_a_growth_series_keeps_the_latest_unbroken_run(
        self, db, instrument, issuer, source
    ):
        # TSLA's shape: revenue is absent for the earliest years and present
        # for every year since. Discarding the concept entirely threw away
        # three usable growth observations and left the bubble engine's
        # divergence signal permanently unevaluable.
        facts = [
            fact("net_income", "5", period_start=dt.date(2019, 1, 1),
                 period_end=dt.date(2019, 12, 31), filed=dt.date(2020, 2, 1),
                 accession="a"),
            fact("net_income", "7", period_start=dt.date(2020, 1, 1),
                 period_end=dt.date(2020, 12, 31), filed=dt.date(2021, 2, 1),
                 accession="b"),
        ]
        for year, value in ((2021, "100"), (2022, "120"), (2023, "150")):
            facts.append(
                fact("revenue", value, period_start=dt.date(year, 1, 1),
                     period_end=dt.date(year, 12, 31),
                     filed=dt.date(year + 1, 2, 1), accession=f"r{year}")
            )
            facts.append(
                fact("net_income", "9", period_start=dt.date(year, 1, 1),
                     period_end=dt.date(year, 12, 31),
                     filed=dt.date(year + 1, 2, 1), accession=f"r{year}")
            )
        ingest_facts(db, issuer=issuer, facts=facts, data_source=source)
        supersede_older_revisions(db, issuer.id)
        snapshot = load_fundamentals(
            db, instrument, as_of=AsOf(cutoff=dt.datetime(2025, 1, 1, tzinfo=UTC))
        )

        assert snapshot.series["revenue_by_period"] == [100.0, 120.0, 150.0], (
            "the three adjacent years are usable even though earlier years "
            "report no revenue"
        )
        assert len(snapshot.series["net_income_by_period"]) == 5
        assert any("shortened" in note for note in snapshot.notes), (
            "a shortened series must say so; silently serving fewer periods "
            "than the caller asked for is the omission this platform forbids"
        )

    def test_a_hole_is_never_bridged(self, db, instrument, issuer, source):
        # The run must stop at the gap, not skip over it: 2021 and 2023 are
        # not adjacent, and treating them as adjacent would report a two-year
        # change as one year's growth.
        for year, value in ((2021, "100"), (2023, "150"), (2024, "160")):
            ingest_facts(
                db,
                issuer=issuer,
                facts=[
                    fact("revenue", value, period_start=dt.date(year, 1, 1),
                         period_end=dt.date(year, 12, 31),
                         filed=dt.date(year + 1, 2, 1), accession=f"r{year}")
                ],
                data_source=source,
            )
        supersede_older_revisions(db, issuer.id)
        snapshot = load_fundamentals(
            db, instrument, as_of=AsOf(cutoff=dt.datetime(2026, 1, 1, tzinfo=UTC))
        )

        assert snapshot.series["revenue_by_period"] == [150.0, 160.0], (
            "2021 sits the far side of a missing 2022 and must be excluded"
        )

    def test_an_instrument_with_no_issuer_reports_why(self, db, instrument):
        instrument.issuer_id = None
        db.flush()
        snapshot = load_fundamentals(
            db, instrument, as_of=AsOf(cutoff=dt.datetime(2024, 1, 1, tzinfo=UTC))
        )
        assert snapshot.is_empty
        assert any("no issuer linked" in n for n in snapshot.notes)
