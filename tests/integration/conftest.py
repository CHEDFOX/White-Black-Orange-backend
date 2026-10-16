"""Database fixtures.

Tests here run against a real PostgreSQL instance because the behaviour under
test -- window functions for revision selection, unique constraints, JSONB --
is database behaviour, and SQLite would validate a different system.

``SIP_TEST_DATABASE_URL`` selects the database; the tests are skipped rather
than failing when none is reachable.
"""

from __future__ import annotations

import datetime as dt
import decimal
import os
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.timeutil import UTC
from app.db.models import Base
from app.db.models.enums import AdjustmentBasis, DataClassification, InstrumentType
from app.db.models.instrument import Instrument, InstrumentTicker, Issuer
from app.db.models.provider import DataSource

TEST_DB_URL = os.environ.get(
    "SIP_TEST_DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/sip_test",
)


@pytest.fixture(scope="session")
def engine():
    try:
        eng = create_engine(TEST_DB_URL, future=True)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not reachable at {TEST_DB_URL}: {exc}")
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine) -> Iterator[Session]:
    """A session wrapped in a transaction that is always rolled back.

    Each test therefore sees a clean database without paying for a schema
    rebuild per test.
    """
    connection = engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(bind=connection, future=True, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def source(db: Session) -> DataSource:
    src = DataSource(
        key="test_source",
        name="Test Source",
        kind="market_data",
        classification=DataClassification.PRODUCTION,
        attribution_required=False,
        official_api=True,
    )
    db.add(src)
    db.flush()
    return src


@pytest.fixture
def synthetic_source(db: Session) -> DataSource:
    src = DataSource(
        key="test_fixture",
        name="Test Fixture Source",
        kind="synthetic",
        classification=DataClassification.SYNTHETIC,
        attribution_required=False,
        official_api=False,
    )
    db.add(src)
    db.flush()
    return src


@pytest.fixture
def issuer(db: Session) -> Issuer:
    iss = Issuer(name="Test Issuer Inc.", cik="0000000001", sector="Technology")
    db.add(iss)
    db.flush()
    return iss


@pytest.fixture
def instrument(db: Session, issuer: Issuer) -> Instrument:
    inst = Instrument(
        symbol="TEST",
        exchange_code="XNYS",
        mic="XNYS",
        name="Test Instrument",
        instrument_type=InstrumentType.EQUITY,
        currency="USD",
        timezone="America/New_York",
        calendar_code="XNYS",
        issuer_id=issuer.id,
        listed_on=dt.date(2000, 1, 3),
    )
    db.add(inst)
    db.flush()
    db.add(
        InstrumentTicker(
            instrument_id=inst.id,
            symbol="TEST",
            exchange_code="XNYS",
            valid_from=dt.date(2000, 1, 3),
            valid_to=None,
        )
    )
    db.flush()
    return inst


def add_price_bar(
    db: Session,
    instrument: Instrument,
    source: DataSource,
    *,
    event_time: dt.datetime,
    close: str,
    available_at: dt.datetime | None = None,
    revision: int = 0,
    superseded_at: dt.datetime | None = None,
    resolution: str = "1d",
    adjustment_basis: str = AdjustmentBasis.TOTAL_RETURN,
    classification: str = DataClassification.PRODUCTION,
):
    from app.db.models.market_data import PriceBar

    close_d = decimal.Decimal(close)
    bar_close = event_time + dt.timedelta(hours=6, minutes=30)
    bar = PriceBar(
        id=uuid.uuid4(),
        instrument_id=instrument.id,
        resolution=resolution,
        adjustment_basis=adjustment_basis,
        data_source_id=source.id,
        event_time=event_time,
        bar_close_time=bar_close,
        published_at=bar_close,
        available_at=available_at or (bar_close + dt.timedelta(hours=1)),
        revision=revision,
        superseded_at=superseded_at,
        open=close_d,
        high=close_d + decimal.Decimal("1"),
        low=close_d - decimal.Decimal("1"),
        close=close_d,
        volume=decimal.Decimal(1_000_000),
        currency="USD",
        classification=classification,
    )
    db.add(bar)
    db.flush()
    return bar


def utc(year: int, month: int, day: int, hour: int = 14, minute: int = 30) -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, tzinfo=UTC)
