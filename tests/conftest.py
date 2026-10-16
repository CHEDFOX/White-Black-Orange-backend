"""Shared test fixtures.

Every price series used in a unit test comes from the fixture provider or an
explicit hand-built bar list, never from a live provider. Tests that touch the
network or a database are marked and excluded from the default run.
"""

from __future__ import annotations

import datetime as dt
import decimal

import pytest

from app.core.calendars import get_calendar
from app.core.timeutil import UTC
from app.db.models.enums import AdjustmentBasis
from app.providers.base import ProviderBar, ProviderFetchRequest
from app.providers.fixture import FixtureProvider, SyntheticSpec


@pytest.fixture
def calendar():
    return get_calendar("XNYS")


@pytest.fixture
def fixture_provider() -> FixtureProvider:
    return FixtureProvider()


@pytest.fixture
def daily_bars(fixture_provider: FixtureProvider) -> list[ProviderBar]:
    batch = fixture_provider.fetch_bars(
        ProviderFetchRequest(
            symbol="SYNTH",
            start=dt.datetime(2023, 1, 3, tzinfo=UTC),
            end=dt.datetime(2023, 12, 29, tzinfo=UTC),
            resolution="1d",
        )
    )
    return list(batch.bars)


def make_bar(
    *,
    day: dt.date,
    open_: str = "100.00",
    high: str = "101.00",
    low: str = "99.00",
    close: str = "100.50",
    volume: str | None = "1000000",
    resolution: str = "1d",
    adjustment_basis: str = AdjustmentBasis.RAW,
    currency: str = "USD",
) -> ProviderBar:
    """Build one bar with exact decimals, for precise assertions."""
    cal = get_calendar("XNYS")
    session = cal.session(day)
    open_utc = (
        session.open_utc
        if session
        else dt.datetime.combine(day, dt.time(14, 30), tzinfo=UTC)
    )
    close_utc = (
        session.close_utc
        if session
        else dt.datetime.combine(day, dt.time(21, 0), tzinfo=UTC)
    )
    return ProviderBar(
        event_time=open_utc,
        bar_close_time=close_utc,
        open=decimal.Decimal(open_),
        high=decimal.Decimal(high),
        low=decimal.Decimal(low),
        close=decimal.Decimal(close),
        volume=decimal.Decimal(volume) if volume is not None else None,
        currency=currency,
        resolution=resolution,
        adjustment_basis=adjustment_basis,
    )


@pytest.fixture
def bar_factory():
    return make_bar


@pytest.fixture
def gap_spec() -> SyntheticSpec:
    return SyntheticSpec(
        symbol="GAPTEST",
        seed=1,
        skip_dates=frozenset({dt.date(2023, 5, 15), dt.date(2023, 5, 16)}),
    )
