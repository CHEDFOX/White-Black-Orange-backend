"""Corporate actions.

Adjustment factors are derived from these rows rather than trusted blindly from
a provider's pre-adjusted series, so that an adjustment can be recomputed,
audited and reversed.
"""

from __future__ import annotations

import datetime as dt
import decimal
import uuid

from sqlalchemy import Date, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.models.mixins import PointInTimeMixin
from app.db.types import JSONType, PriceNumeric, RatioNumeric


class CorporateAction(Base, PointInTimeMixin):
    __tablename__ = "corporate_actions"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "action_type",
            "ex_date",
            "data_source_id",
            "revision",
            name="uq_corporate_actions_instrument_type_exdate_source_revision",
        ),
        Index("ix_corporate_actions_lookup", "instrument_id", "ex_date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    action_type: Mapped[str] = mapped_column(String(24), nullable=False)

    # Announcement date matters for point-in-time: a split known on 1 June with
    # an ex-date of 1 July was actionable information in June.
    announced_on: Mapped[dt.date | None] = mapped_column(Date)
    ex_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    record_date: Mapped[dt.date | None] = mapped_column(Date)
    pay_date: Mapped[dt.date | None] = mapped_column(Date)

    # Split: 4-for-1 -> ratio_numerator=4, ratio_denominator=1.
    ratio_numerator: Mapped[decimal.Decimal | None] = mapped_column(RatioNumeric)
    ratio_denominator: Mapped[decimal.Decimal | None] = mapped_column(RatioNumeric)
    cash_amount: Mapped[decimal.Decimal | None] = mapped_column(PriceNumeric)
    currency: Mapped[str | None] = mapped_column(String(3))

    new_symbol: Mapped[str | None] = mapped_column(String(32))
    counterparty: Mapped[str | None] = mapped_column(String(300))
    description: Mapped[str | None] = mapped_column(Text)
    raw_payload: Mapped[dict | None] = mapped_column(JSONType)
