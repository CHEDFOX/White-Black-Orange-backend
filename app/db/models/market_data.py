"""Price, quote and data-quality tables."""

from __future__ import annotations

import datetime as dt
import decimal
import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.models.enums import AdjustmentBasis, DataClassification
from app.db.models.mixins import PointInTimeMixin
from app.db.types import JSONType, PriceNumeric, VolumeNumeric


class PriceBar(Base, PointInTimeMixin):
    """One OHLCV bar.

    The uniqueness key includes ``resolution`` and ``adjustment_basis`` because
    a raw daily bar and a total-return daily bar for the same instant are
    different observations, and silently overwriting one with the other is the
    single most common way a price database becomes unusable.

    ``event_time`` is the bar's *open* instant in UTC. ``bar_close_time`` is
    stored explicitly so an aggregation never has to guess session length.
    """

    __tablename__ = "price_bars"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "resolution",
            "adjustment_basis",
            "data_source_id",
            "event_time",
            "revision",
            name="uq_price_bars_instrument_resolution_basis_source_time_revision",
        ),
        Index(
            "ix_price_bars_series",
            "instrument_id",
            "resolution",
            "adjustment_basis",
            "event_time",
        ),
        Index(
            "ix_price_bars_pit",
            "instrument_id",
            "resolution",
            "adjustment_basis",
            "available_at",
        ),
        CheckConstraint("high >= low", name="high_ge_low"),
        CheckConstraint("open > 0 AND high > 0 AND low > 0 AND close > 0", name="prices_positive"),
        CheckConstraint("volume IS NULL OR volume >= 0", name="volume_non_negative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    resolution: Mapped[str] = mapped_column(String(8), nullable=False)
    adjustment_basis: Mapped[str] = mapped_column(
        String(16), nullable=False, default=AdjustmentBasis.RAW
    )
    bar_close_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    open: Mapped[decimal.Decimal] = mapped_column(PriceNumeric, nullable=False)
    high: Mapped[decimal.Decimal] = mapped_column(PriceNumeric, nullable=False)
    low: Mapped[decimal.Decimal] = mapped_column(PriceNumeric, nullable=False)
    close: Mapped[decimal.Decimal] = mapped_column(PriceNumeric, nullable=False)
    # Volume is nullable, never defaulted to zero: "no reported volume" and
    # "zero shares traded" are different facts.
    volume: Mapped[decimal.Decimal | None] = mapped_column(VolumeNumeric)
    trade_count: Mapped[int | None] = mapped_column(Integer)
    vwap: Mapped[decimal.Decimal | None] = mapped_column(PriceNumeric)

    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    # True when the provider explicitly flags the bar as built from partial or
    # consolidated-tape-incomplete data.
    is_partial: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    provider_timestamp: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    raw_payload: Mapped[dict | None] = mapped_column(JSONType)


class Quote(Base, PointInTimeMixin):
    """Bid/ask snapshot. Populated only where a provider licence permits."""

    __tablename__ = "quotes"
    __table_args__ = (
        Index("ix_quotes_series", "instrument_id", "event_time"),
        UniqueConstraint(
            "instrument_id", "data_source_id", "event_time", "revision",
            name="uq_quotes_instrument_id_data_source_id_event_time_revision",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    bid: Mapped[decimal.Decimal | None] = mapped_column(PriceNumeric)
    ask: Mapped[decimal.Decimal | None] = mapped_column(PriceNumeric)
    bid_size: Mapped[decimal.Decimal | None] = mapped_column(VolumeNumeric)
    ask_size: Mapped[decimal.Decimal | None] = mapped_column(VolumeNumeric)
    last: Mapped[decimal.Decimal | None] = mapped_column(PriceNumeric)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    is_delayed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    delay_minutes: Mapped[int | None] = mapped_column(Integer)


class DataQualityIssue(Base):
    """A validation finding, kept rather than discarded.

    Storing issues means a report can say "this period contains three bars
    whose high is below the open" instead of presenting a clean-looking chart
    built on broken input.
    """

    __tablename__ = "data_quality_issues"
    __table_args__ = (
        Index("ix_data_quality_issues_lookup", "instrument_id", "kind", "window_start"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE")
    )
    data_source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("data_sources.id", ondelete="SET NULL")
    )
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    resolution: Mapped[str | None] = mapped_column(String(8))
    window_start: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    detected_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict] = mapped_column(JSONType, default=dict)
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    classification: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DataClassification.PRODUCTION
    )


class DataGap(Base):
    """A trading session (or interval) for which no bar exists.

    Gaps are recorded explicitly so that analysis can decline to run, or run and
    disclose the holes -- never interpolate across them.
    """

    __tablename__ = "data_gaps"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "resolution", "gap_start",
            name="uq_data_gaps_instrument_id_resolution_gap_start",
        ),
        Index("ix_data_gaps_lookup", "instrument_id", "resolution", "gap_start"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    resolution: Mapped[str] = mapped_column(String(8), nullable=False)
    gap_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    gap_end: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expected_sessions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    detected_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    filled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
