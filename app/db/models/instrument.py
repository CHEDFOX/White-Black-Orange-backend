"""Instrument registry.

Deliberately provider-agnostic. A ticker is *not* a global key: ``BMW`` is a
different instrument on XETR and on the US OTC market, and ``FB`` referred to
Meta before 2022 and to nothing afterwards. The internal UUID is the only
stable handle; tickers are time-ranged rows and provider symbols live in a
side table.
"""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.models.enums import InstrumentType
from app.db.types import JSONType


class Issuer(Base, TimestampMixin):
    """The legal entity behind one or more instruments."""

    __tablename__ = "issuers"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    legal_name: Mapped[str | None] = mapped_column(String(300))
    country: Mapped[str | None] = mapped_column(String(2))
    lei: Mapped[str | None] = mapped_column(String(20), unique=True)
    cik: Mapped[str | None] = mapped_column(String(10), unique=True, index=True)
    sector: Mapped[str | None] = mapped_column(String(100), index=True)
    industry: Mapped[str | None] = mapped_column(String(150), index=True)
    sic_code: Mapped[str | None] = mapped_column(String(8))
    website: Mapped[str | None] = mapped_column(String(300))
    description: Mapped[str | None] = mapped_column(Text)
    fiscal_year_end_month: Mapped[int | None] = mapped_column(Integer)

    instruments: Mapped[list["Instrument"]] = relationship(back_populates="issuer")


class Instrument(Base, TimestampMixin):
    __tablename__ = "instruments"
    __table_args__ = (
        UniqueConstraint("symbol", "exchange_code", name="uq_instruments_symbol_exchange_code"),
        Index("ix_instruments_search", "symbol", "name"),
        Index("ix_instruments_type_active", "instrument_type", "is_active"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    # Current canonical symbol and venue. Historical symbols live in
    # ``InstrumentTicker``; this pair is only the "as of now" convenience view.
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    exchange_code: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    mic: Mapped[str | None] = mapped_column(String(4), index=True)
    exchange_name: Mapped[str | None] = mapped_column(String(120))

    name: Mapped[str] = mapped_column(String(300), nullable=False)
    instrument_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default=InstrumentType.EQUITY
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    country: Mapped[str | None] = mapped_column(String(2))

    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="America/New_York")
    calendar_code: Mapped[str] = mapped_column(String(12), nullable=False, default="XNYS")

    listed_on: Mapped[dt.date | None] = mapped_column(Date)
    delisted_on: Mapped[dt.date | None] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    issuer_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("issuers.id", ondelete="SET NULL"), index=True
    )
    # For ETFs/indexes: the instrument this one tracks or benchmarks against.
    benchmark_instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="SET NULL")
    )

    isin: Mapped[str | None] = mapped_column(String(12), index=True)
    figi: Mapped[str | None] = mapped_column(String(12), index=True)
    cusip: Mapped[str | None] = mapped_column(String(9))

    metadata_json: Mapped[dict] = mapped_column(JSONType, default=dict)

    issuer: Mapped[Issuer | None] = relationship(back_populates="instruments")
    identifiers: Mapped[list["InstrumentIdentifier"]] = relationship(
        back_populates="instrument", cascade="all, delete-orphan"
    )
    tickers: Mapped[list["InstrumentTicker"]] = relationship(
        back_populates="instrument", cascade="all, delete-orphan"
    )
    coverage: Mapped[list["InstrumentCoverage"]] = relationship(
        back_populates="instrument", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Instrument {self.symbol}@{self.exchange_code}>"


class InstrumentTicker(Base):
    """Ticker history with explicit validity ranges.

    ``valid_to`` NULL means "current". A point-in-time symbol lookup selects
    the row whose range contains the analysis timestamp, which is what lets the
    platform resolve ``FB`` for a 2019 analysis and ``META`` for a 2024 one.
    """

    __tablename__ = "instrument_tickers"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "symbol", "valid_from", name="uq_instrument_tickers_instrument_id_symbol_valid_from"
        ),
        Index("ix_instrument_tickers_symbol_range", "symbol", "valid_from", "valid_to"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange_code: Mapped[str | None] = mapped_column(String(12))
    valid_from: Mapped[dt.date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[dt.date | None] = mapped_column(Date)
    note: Mapped[str | None] = mapped_column(Text)

    instrument: Mapped[Instrument] = relationship(back_populates="tickers")


class InstrumentIdentifier(Base):
    """Provider-specific symbols, so no adapter owns the canonical identity."""

    __tablename__ = "instrument_identifiers"
    __table_args__ = (
        UniqueConstraint(
            "data_source_id", "identifier", name="uq_instrument_identifiers_data_source_id_identifier"
        ),
        Index("ix_instrument_identifiers_instrument", "instrument_id", "data_source_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    data_source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    identifier: Mapped[str] = mapped_column(String(64), nullable=False)
    identifier_type: Mapped[str] = mapped_column(String(32), nullable=False, default="symbol")

    instrument: Mapped[Instrument] = relationship(back_populates="identifiers")


class InstrumentCoverage(Base, TimestampMixin):
    """What data we actually hold for an instrument, per resolution and source.

    The UI reads this to avoid implying uniform coverage across instruments.
    """

    __tablename__ = "instrument_coverage"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "data_source_id",
            "data_kind",
            "resolution",
            # Shortened: PostgreSQL caps identifiers at 63 characters.
            name="uq_instrument_coverage_instr_source_kind_res",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    data_source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    data_kind: Mapped[str] = mapped_column(String(32), nullable=False)  # prices|fundamentals|news|...
    resolution: Mapped[str | None] = mapped_column(String(8))
    first_observation: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    last_observation: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    observation_count: Mapped[int] = mapped_column(Integer, default=0)
    last_refreshed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    known_gap_count: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str | None] = mapped_column(Text)

    instrument: Mapped[Instrument] = relationship(back_populates="coverage")


class PeerGroup(Base, TimestampMixin):
    """Sector/industry peer sets used for relative comparison."""

    __tablename__ = "peer_groups"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    basis: Mapped[str] = mapped_column(String(32), nullable=False, default="sector")
    members: Mapped[list["PeerGroupMember"]] = relationship(
        back_populates="group", cascade="all, delete-orphan"
    )


class PeerGroupMember(Base):
    __tablename__ = "peer_group_members"
    __table_args__ = (
        UniqueConstraint(
            "group_id", "instrument_id", "valid_from", name="uq_peer_group_members_group_id_instrument_id_valid_from"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("peer_groups.id", ondelete="CASCADE"), nullable=False
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Membership is time-ranged so a historical peer comparison uses the peers
    # of that era, not today's survivors.
    valid_from: Mapped[dt.date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[dt.date | None] = mapped_column(Date)
    weight: Mapped[float | None] = mapped_column()

    group: Mapped[PeerGroup] = relationship(back_populates="members")
