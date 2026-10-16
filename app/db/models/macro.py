"""Economic series, releases and revision vintages."""

from __future__ import annotations

import datetime as dt
import decimal
import uuid

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.models.mixins import PointInTimeMixin
from app.db.types import JSONType, RatioNumeric


class EconomicSeries(Base, TimestampMixin):
    __tablename__ = "economic_series"
    __table_args__ = (
        UniqueConstraint(
            "data_source_id", "series_code",
            name="uq_economic_series_data_source_id_series_code",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    data_source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    series_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    frequency: Mapped[str] = mapped_column(String(16), nullable=False)
    unit: Mapped[str] = mapped_column(String(64), nullable=False)
    seasonal_adjustment: Mapped[str | None] = mapped_column(String(40))
    country: Mapped[str | None] = mapped_column(String(2))
    category: Mapped[str | None] = mapped_column(String(60), index=True)
    # True when this platform has vintage (as-first-released) data for the
    # series. When False, point-in-time macro analysis is approximate and the
    # report must say so.
    has_vintages: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    observations: Mapped[list["EconomicObservation"]] = relationship(
        back_populates="series", cascade="all, delete-orphan"
    )


class EconomicRelease(Base):
    """A scheduled publication event for one or more series."""

    __tablename__ = "economic_releases"
    __table_args__ = (
        UniqueConstraint(
            "data_source_id", "release_code", "release_date",
            name="uq_economic_releases_data_source_id_release_code_release_date",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    data_source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    release_code: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    release_date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    release_time_utc: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    is_scheduled: Mapped[bool] = mapped_column(Boolean, default=True)


class EconomicObservation(Base, PointInTimeMixin):
    """One value of one series, for one vintage.

    ``event_time`` is the period the number describes; ``available_at`` is when
    it was published. A revision is a new row (revision+1) with a later
    ``available_at``, so "US unemployment as known on 2020-04-15" returns the
    first print, not today's revised figure.
    """

    __tablename__ = "economic_observations"
    __table_args__ = (
        UniqueConstraint(
            "series_id", "event_time", "revision",
            name="uq_economic_observations_series_id_event_time_revision",
        ),
        Index("ix_economic_observations_pit", "series_id", "available_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    series_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("economic_series.id", ondelete="CASCADE"), nullable=False
    )
    release_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("economic_releases.id", ondelete="SET NULL")
    )
    value: Mapped[decimal.Decimal | None] = mapped_column(RatioNumeric)
    unit: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_payload: Mapped[dict | None] = mapped_column(JSONType)

    series: Mapped[EconomicSeries] = relationship(back_populates="observations")
