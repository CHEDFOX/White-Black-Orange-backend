"""Normalising XBRL tags to concepts across a tag change.

An issuer does not use one XBRL tag for its whole life. Tesla reported revenue
as ``Revenues`` through FY2020 and as
``RevenueFromContractWithCustomerExcludingAssessedTax`` from FY2021. Reading
only the first tag that matches therefore truncates history at the changeover
and, worse, truncates it *silently*: a concept missing from a period is
indistinguishable from one the issuer never reported.

This cost TSLA its FY2019 and FY2020 revenue, which in turn left the bubble
engine's fundamental-divergence signal permanently unevaluable.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.providers.sec_edgar import SecEdgarProvider

PREFERRED = "RevenueFromContractWithCustomerExcludingAssessedTax"
FALLBACK = "Revenues"


def row(year: int, value: float, *, accession: str | None = None) -> dict:
    return {
        "val": value,
        "start": f"{year}-01-01",
        "end": f"{year}-12-31",
        "filed": f"{year + 1}-02-15",
        "accn": accession or f"acc-{year}",
        "form": "10-K",
        "fy": year,
        "fp": "FY",
    }


@pytest.fixture
def provider(monkeypatch):
    """A provider whose only network call is replaced by a fixed payload."""
    created = SecEdgarProvider.__new__(SecEdgarProvider)
    created._data_client = object()
    created._www_client = object()
    return created


def facts_from(provider, monkeypatch, us_gaap: dict) -> list:
    monkeypatch.setattr(
        provider, "_json", lambda client, path: {"facts": {"us-gaap": us_gaap}}, raising=False
    )
    monkeypatch.setattr(SecEdgarProvider, "pad_cik", staticmethod(lambda cik: "0000000000"))
    return provider.fetch_company_facts(1, concepts=("revenue",))


class TestTagChange:
    def test_a_tag_change_mid_history_does_not_truncate_the_series(
        self, provider, monkeypatch
    ):
        facts = facts_from(
            provider,
            monkeypatch,
            {
                PREFERRED: {"units": {"USD": [row(2021, 53.8), row(2022, 81.5)]}},
                FALLBACK: {"units": {"USD": [row(2019, 24.6), row(2020, 31.5)]}},
            },
        )

        years = sorted(f.period_end.year for f in facts)
        assert years == [2019, 2020, 2021, 2022], (
            "the older tag's years must survive; stopping at the first tag that "
            "matched is what silently dropped FY2019 and FY2020"
        )

    def test_the_preferred_tag_wins_when_one_filing_carries_both(
        self, provider, monkeypatch
    ):
        facts = facts_from(
            provider,
            monkeypatch,
            {
                PREFERRED: {"units": {"USD": [row(2021, 53.8, accession="same")]}},
                FALLBACK: {"units": {"USD": [row(2021, 99.9, accession="same")]}},
            },
        )

        assert len(facts) == 1, "one period in one filing is one fact, not two"
        assert facts[0].source_concept == PREFERRED
        assert float(facts[0].value) == pytest.approx(53.8)

    def test_the_same_period_in_a_later_filing_survives_as_a_revision(
        self, provider, monkeypatch
    ):
        facts = facts_from(
            provider,
            monkeypatch,
            {
                PREFERRED: {
                    "units": {
                        "USD": [
                            row(2021, 53.8, accession="original"),
                            row(2021, 54.1, accession="restatement"),
                        ]
                    }
                }
            },
        )

        assert len(facts) == 2, (
            "a restatement is the point-in-time signal itself and must not be "
            "collapsed into the figure that superseded it"
        )
        assert sorted(float(f.value) for f in facts) == pytest.approx([53.8, 54.1])

    def test_a_concept_reported_under_one_tag_only_is_unaffected(
        self, provider, monkeypatch
    ):
        facts = facts_from(
            provider,
            monkeypatch,
            {FALLBACK: {"units": {"USD": [row(2020, 31.5)]}}},
        )

        assert len(facts) == 1
        assert facts[0].source_concept == FALLBACK
        assert facts[0].period_start == dt.date(2020, 1, 1)
