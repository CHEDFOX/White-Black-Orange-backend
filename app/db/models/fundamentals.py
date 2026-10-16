"""Financial statements and facts.

Modelled as individual *facts* rather than wide statement rows because
providers disagree about line-item taxonomies, restatements are common, and
point-in-time correctness requires per-fact publication dates. A restated
FY2022 revenue figure filed in 2024 must not be visible to a 2023 analysis.
"""

from __future__ import annotations

import datetime as dt
import decimal
import uuid

from sqlalchemy import (
    Boolean,
    Date,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.models.mixins import PointInTimeMixin
from app.db.types import FactNumeric, JSONType


class FiscalPeriod(Base):
    """A reporting period, identified by what the data states rather than by label.

    The identity is ``(issuer, period_end, period_kind)`` where ``period_kind``
    is derived from the reported duration. Two earlier approaches both fail:

    * Keying on the filing's own ``fy``/``fp`` fields is wrong, because those
      describe the *filing*, not the period. FY2021 revenue restated in the
      FY2023 10-K arrives tagged ``fy=2023, fp=FY``, which would file a 2021
      figure under 2023 and make every historical comparison wrong.
    * Keying on ``(fiscal_year, fiscal_period)`` also collides, because a 10-Q
      reports both a discrete quarter and a year-to-date cumulative figure under
      the same ``fp``. A 90-day quarter and a 272-day nine-month total would
      share one period row, and a "quarterly revenue" could silently be a
      nine-month total.

    ``period_end`` and the duration are facts in the payload, so keying on them
    is unambiguous. ``fiscal_year`` is kept as a descriptive label only.
    """

    __tablename__ = "fiscal_periods"
    __table_args__ = (
        UniqueConstraint(
            "issuer_id", "period_end", "period_kind",
            name="uq_fiscal_periods_issuer_period_end_kind",
        ),
        Index("ix_fiscal_periods_issuer_end", "issuer_id", "period_end"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    issuer_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("issuers.id", ondelete="CASCADE"), nullable=False
    )
    # Derived from period_end, for display. Not an identity component, because
    # the issuer's own fiscal-year labelling cannot be reconstructed reliably.
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    # Legacy label from the filing, retained for traceability.
    fiscal_period: Mapped[str] = mapped_column(String(4), nullable=False)
    # FY | 9M | H | Q | PIT | OTHER -- derived from the reported duration.
    period_kind: Mapped[str] = mapped_column(String(8), nullable=False, default="OTHER")
    duration_days: Mapped[int | None] = mapped_column(Integer)
    period_start: Mapped[dt.date] = mapped_column(Date, nullable=False)
    period_end: Mapped[dt.date] = mapped_column(Date, nullable=False)
    # Calendar-quarter alignment, so a company with a September year-end can be
    # compared to one with a December year-end without pretending they match.
    calendar_quarter_end: Mapped[dt.date | None] = mapped_column(Date)

    facts: Mapped[list[FinancialFact]] = relationship(back_populates="period")


class Filing(Base, PointInTimeMixin):
    """A regulatory filing, the primary-source anchor for facts and citations."""

    __tablename__ = "filings"
    __table_args__ = (
        UniqueConstraint(
            "data_source_id", "external_id", "revision",
            name="uq_filings_data_source_id_external_id_revision",
        ),
        Index("ix_filings_issuer_filed", "issuer_id", "filed_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    issuer_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("issuers.id", ondelete="CASCADE"), nullable=False
    )
    external_id: Mapped[str] = mapped_column(String(100), nullable=False)
    form_type: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    filed_at: Mapped[dt.datetime] = mapped_column(nullable=False)
    period_of_report: Mapped[dt.date | None] = mapped_column(Date)
    accession_number: Mapped[str | None] = mapped_column(String(32), index=True)
    primary_document_url: Mapped[str | None] = mapped_column(String(600))
    title: Mapped[str | None] = mapped_column(Text)
    is_amendment: Mapped[bool] = mapped_column(Boolean, default=False)


class FinancialFact(Base, PointInTimeMixin):
    """A single reported number with its unit, taxonomy tag and source filing.

    ``available_at`` is the filing's acceptance timestamp, never the period end.
    """

    __tablename__ = "financial_facts"
    __table_args__ = (
        UniqueConstraint(
            "issuer_id",
            "concept",
            "fiscal_period_id",
            "data_source_id",
            "revision",
            name="uq_financial_facts_issuer_concept_period_source_revision",
        ),
        Index("ix_financial_facts_lookup", "issuer_id", "concept", "available_at"),
        Index("ix_financial_facts_period", "fiscal_period_id", "concept"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    issuer_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("issuers.id", ondelete="CASCADE"), nullable=False
    )
    fiscal_period_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("fiscal_periods.id", ondelete="SET NULL")
    )
    filing_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("filings.id", ondelete="SET NULL"), index=True
    )

    # Normalised concept name, e.g. "revenue", "net_income", "total_debt".
    concept: Mapped[str] = mapped_column(String(80), nullable=False)
    # The provider's own tag, kept so a mapping dispute can be traced.
    source_concept: Mapped[str | None] = mapped_column(String(160))
    taxonomy: Mapped[str | None] = mapped_column(String(32))

    # FactNumeric, not RatioNumeric: a revenue line is ~1e11 and would
    # overflow a Numeric(20, 10) column.
    value: Mapped[decimal.Decimal | None] = mapped_column(FactNumeric)
    unit: Mapped[str] = mapped_column(String(24), nullable=False)
    currency: Mapped[str | None] = mapped_column(String(3))
    # duration | instant -- a flow versus a balance; mixing them is nonsense.
    period_type: Mapped[str] = mapped_column(String(10), nullable=False, default="duration")
    is_restatement: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    segment: Mapped[str | None] = mapped_column(String(120))
    raw_payload: Mapped[dict | None] = mapped_column(JSONType)

    period: Mapped[FiscalPeriod | None] = relationship(back_populates="facts")
