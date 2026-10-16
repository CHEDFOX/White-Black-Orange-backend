"""Data sources, provider health and ingestion runs."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.models.enums import DataClassification, JobStatus
from app.db.types import JSONType


class DataSource(Base, TimestampMixin):
    """A distinct origin of data, with its licensing posture recorded.

    Every observation row points at one of these, so provenance is never lost
    and a licence restriction can be enforced or at least reported.
    """

    __tablename__ = "data_sources"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # market_data|macro|filings|news|synthetic
    homepage_url: Mapped[str | None] = mapped_column(String(500))
    classification: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DataClassification.PRODUCTION
    )

    # Licensing and redistribution posture, surfaced in the API so the UI can
    # attribute data correctly and refuse to re-publish what it may not.
    licence_summary: Mapped[str | None] = mapped_column(Text)
    redistribution_allowed: Mapped[bool] = mapped_column(Boolean, default=False)
    attribution_required: Mapped[bool] = mapped_column(Boolean, default=True)
    attribution_text: Mapped[str | None] = mapped_column(Text)
    official_api: Mapped[bool] = mapped_column(Boolean, default=True)
    caveats: Mapped[str | None] = mapped_column(Text)

    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    config: Mapped[dict] = mapped_column(JSONType, default=dict)

    health_checks: Mapped[list["ProviderHealthCheck"]] = relationship(
        back_populates="data_source", cascade="all, delete-orphan"
    )


class ProviderHealthCheck(Base):
    """Rolling record of provider reachability and latency."""

    __tablename__ = "provider_health_checks"
    __table_args__ = (Index("ix_provider_health_checks_src_time", "data_source_id", "checked_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    data_source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    checked_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    healthy: Mapped[bool] = mapped_column(Boolean, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    status_code: Mapped[int | None] = mapped_column(Integer)
    detail: Mapped[str | None] = mapped_column(Text)

    data_source: Mapped[DataSource] = relationship(back_populates="health_checks")


class IngestionRun(Base):
    """One execution of an ingestion job, for idempotency and audit."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_ingestion_runs_idempotency_key"),
        Index("ix_ingestion_runs_src_started", "data_source_id", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    data_source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_sources.id", ondelete="RESTRICT"), nullable=False
    )
    job_name: Mapped[str] = mapped_column(String(100), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(300), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=JobStatus.PENDING)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    rows_written: Mapped[int] = mapped_column(Integer, default=0)
    rows_rejected: Mapped[int] = mapped_column(Integer, default=0)
    request_params: Mapped[dict] = mapped_column(JSONType, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    stats: Mapped[dict] = mapped_column(JSONType, default=dict)
