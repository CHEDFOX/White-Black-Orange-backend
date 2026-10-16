"""Corporate-action ingestion.

``available_at`` comes from the *announcement* date where the provider supplies
one, not the ex-date: a split announced on 1 June with an ex-date of 1 July was
actionable information in June, and treating it as unknown until July
understates what a market participant knew.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.timeutil import UTC
from app.db.models.corporate_action import CorporateAction
from app.db.models.enums import CorporateActionType, EventCategory, JobStatus
from app.db.models.instrument import Instrument
from app.db.models.news import CorporateEvent
from app.db.models.provider import DataSource
from app.providers.base import ProviderCorporateAction

log = get_logger(__name__)


@dataclass
class CorporateActionReport:
    written: int = 0
    unchanged: int = 0
    events_written: int = 0
    status: str = JobStatus.SUCCEEDED
    notes: list[str] = field(default_factory=list)


_CATEGORY = {
    CorporateActionType.SPLIT: EventCategory.CORPORATE_ACTION,
    CorporateActionType.REVERSE_SPLIT: EventCategory.CORPORATE_ACTION,
    CorporateActionType.CASH_DIVIDEND: EventCategory.DIVIDEND,
    CorporateActionType.SPECIAL_DIVIDEND: EventCategory.DIVIDEND,
    CorporateActionType.STOCK_DIVIDEND: EventCategory.DIVIDEND,
    CorporateActionType.MERGER: EventCategory.MERGER_ACQUISITION,
    CorporateActionType.ACQUISITION: EventCategory.MERGER_ACQUISITION,
    CorporateActionType.SPINOFF: EventCategory.MERGER_ACQUISITION,
    CorporateActionType.TICKER_CHANGE: EventCategory.MARKET_STRUCTURE,
    CorporateActionType.DELISTING: EventCategory.MARKET_STRUCTURE,
}


def _availability(action: ProviderCorporateAction) -> tuple[dt.datetime, dt.datetime | None]:
    """Return ``(available_at, published_at)`` for an action."""
    if action.announced_on is not None:
        announced = dt.datetime.combine(action.announced_on, dt.time(21, 0), tzinfo=UTC)
        return announced, announced
    # Without an announcement date, the ex-date is the only defensible
    # availability. Earlier would assert knowledge we cannot evidence.
    ex = dt.datetime.combine(action.ex_date, dt.time.min, tzinfo=UTC)
    return ex, None


def ingest_corporate_actions(
    session: Session,
    *,
    instrument: Instrument,
    data_source: DataSource,
    actions: list[ProviderCorporateAction],
) -> CorporateActionReport:
    report = CorporateActionReport()
    if not actions:
        report.notes.append(
            f"no corporate actions were returned for {instrument.symbol} in the "
            "requested window"
        )
        return report

    no_announcement = 0
    for action in actions:
        available_at, published_at = _availability(action)
        if action.announced_on is None:
            no_announcement += 1

        existing = session.execute(
            select(CorporateAction).where(
                CorporateAction.instrument_id == instrument.id,
                CorporateAction.action_type == action.action_type,
                CorporateAction.ex_date == action.ex_date,
                CorporateAction.data_source_id == data_source.id,
            )
        ).scalars().first()
        if existing is not None:
            report.unchanged += 1
            continue

        record = CorporateAction(
            instrument_id=instrument.id,
            data_source_id=data_source.id,
            action_type=action.action_type,
            announced_on=action.announced_on,
            ex_date=action.ex_date,
            record_date=action.record_date,
            pay_date=action.pay_date,
            ratio_numerator=action.ratio_numerator,
            ratio_denominator=action.ratio_denominator,
            cash_amount=action.cash_amount,
            currency=action.currency or instrument.currency,
            new_symbol=action.new_symbol,
            description=action.description,
            event_time=dt.datetime.combine(action.ex_date, dt.time.min, tzinfo=UTC),
            published_at=published_at,
            available_at=available_at,
            revision=0,
            raw_payload=action.raw,
        )
        session.add(record)
        session.flush()
        report.written += 1

        category = _CATEGORY.get(action.action_type, EventCategory.CORPORATE_ACTION)
        title = _title_for(action)
        event_exists = session.execute(
            select(CorporateEvent).where(
                CorporateEvent.instrument_id == instrument.id,
                CorporateEvent.category == category,
                CorporateEvent.event_time == record.event_time,
                CorporateEvent.data_source_id == data_source.id,
            )
        ).scalars().first()
        if event_exists is None:
            session.add(
                CorporateEvent(
                    instrument_id=instrument.id,
                    issuer_id=instrument.issuer_id,
                    data_source_id=data_source.id,
                    category=category,
                    title=title,
                    description=action.description,
                    is_scheduled=True,
                    is_confirmed=True,
                    importance=(
                        80
                        if action.action_type
                        in {CorporateActionType.SPLIT, CorporateActionType.REVERSE_SPLIT}
                        else 40
                    ),
                    corporate_action_id=record.id,
                    event_time=record.event_time,
                    published_at=published_at,
                    available_at=available_at,
                    revision=0,
                    payload={"action_type": action.action_type},
                )
            )
            report.events_written += 1

    session.flush()
    if no_announcement:
        report.notes.append(
            f"{no_announcement} action(s) had no announcement date, so their "
            "availability is set to the ex-date. A point-in-time analysis "
            "therefore treats them as unknown until the ex-date, which is "
            "conservative but understates what the market knew."
        )
    return report


def _title_for(action: ProviderCorporateAction) -> str:
    if action.action_type in {CorporateActionType.SPLIT, CorporateActionType.REVERSE_SPLIT}:
        if action.ratio_numerator and action.ratio_denominator:
            num = action.ratio_numerator.normalize()
            den = action.ratio_denominator.normalize()
            return f"{num}:{den} split"
        return "Share split"
    if action.action_type == CorporateActionType.CASH_DIVIDEND:
        return (
            f"Cash dividend {action.cash_amount} {action.currency or ''}".strip()
            if action.cash_amount
            else "Cash dividend"
        )
    return action.action_type.replace("_", " ").title()
