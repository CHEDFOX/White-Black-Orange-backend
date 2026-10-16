"""Point-in-time bookkeeping shared by every observation table.

The platform's central claim is that it can reconstruct what was *knowable* at
a past instant. That requires four separate clocks per observation, which are
routinely conflated in financial datasets:

``event_time``
    When the thing happened in the world (quarter end, trade print, rate decision).
``published_at``
    When the publisher first released it. A Q4 result with ``event_time`` of
    31 December may have ``published_at`` in February.
``available_at``
    When *this platform* could first have acted on it. Equals ``published_at``
    plus any provider delivery latency. Point-in-time queries filter on this
    column and no other.
``ingested_at``
    When our pipeline wrote the row. Operational only; never used for filtering,
    because a backfill run today must not make 2019 data unavailable in 2019.

Revisions are append-only: a corrected value is a new row with ``revision``
incremented and its own ``available_at``. The previous row keeps
``superseded_at`` set to the successor's ``available_at`` so that a query as of
a date before the correction still returns the original figure.
"""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, declared_attr, mapped_column

from app.db.models.enums import DataClassification


class PointInTimeMixin:
    event_time: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    published_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    available_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    superseded_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    @declared_attr
    @classmethod
    def data_source_id(cls) -> Mapped[uuid.UUID]:
        return mapped_column(
            ForeignKey("data_sources.id", ondelete="RESTRICT"),
            nullable=False,
            index=True,
        )

    classification: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DataClassification.PRODUCTION
    )


def pit_indexes(table: str, *entity_cols: str) -> tuple[Index, ...]:
    """Indexes that make ``available_at``/``superseded_at`` filtering cheap."""
    return (
        Index(
            f"ix_{table}_pit",
            *entity_cols,
            "available_at",
            "superseded_at",
        ),
    )
