"""Instrument registration and search.

Registration is provider-driven but provider-independent: an adapter supplies
``ProviderInstrument`` metadata, and this module decides identity. A ticker
collision across venues produces two instruments, not one overwritten row.
"""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core.calendars import get_calendar
from app.core.logging import get_logger
from app.db.models.instrument import (
    Instrument,
    InstrumentCoverage,
    InstrumentIdentifier,
    InstrumentTicker,
    Issuer,
)
from app.db.models.provider import DataSource
from app.providers.base import ProviderInstrument

log = get_logger(__name__)


def find_instrument(
    session: Session, symbol: str, *, exchange_code: str | None = None
) -> Instrument | None:
    stmt = select(Instrument).where(func.upper(Instrument.symbol) == symbol.upper())
    if exchange_code:
        stmt = stmt.where(Instrument.exchange_code == exchange_code.upper())
    return session.execute(stmt.limit(1)).scalars().first()


def find_by_provider_identifier(
    session: Session, data_source_id: uuid.UUID, identifier: str
) -> Instrument | None:
    stmt = (
        select(Instrument)
        .join(InstrumentIdentifier, InstrumentIdentifier.instrument_id == Instrument.id)
        .where(
            InstrumentIdentifier.data_source_id == data_source_id,
            InstrumentIdentifier.identifier == identifier,
        )
    )
    return session.execute(stmt.limit(1)).scalars().first()


def upsert_issuer(
    session: Session,
    *,
    name: str,
    cik: str | None = None,
    sector: str | None = None,
    industry: str | None = None,
    country: str | None = None,
    website: str | None = None,
    description: str | None = None,
    sic_code: str | None = None,
    fiscal_year_end_month: int | None = None,
) -> Issuer:
    existing: Issuer | None = None
    if cik:
        existing = session.execute(
            select(Issuer).where(Issuer.cik == cik)
        ).scalars().first()
    if existing is None:
        existing = session.execute(
            select(Issuer).where(func.lower(Issuer.name) == name.lower())
        ).scalars().first()

    if existing is None:
        issuer = Issuer(
            name=name,
            cik=cik,
            sector=sector,
            industry=industry,
            country=country,
            website=website,
            description=description,
            sic_code=sic_code,
            fiscal_year_end_month=fiscal_year_end_month,
        )
        session.add(issuer)
        session.flush()
        return issuer

    # Fill blanks only. Overwriting curated metadata with a thinner provider's
    # version loses information.
    for field, value in (
        ("cik", cik), ("sector", sector), ("industry", industry),
        ("country", country), ("website", website), ("description", description),
        ("sic_code", sic_code), ("fiscal_year_end_month", fiscal_year_end_month),
    ):
        if value and not getattr(existing, field):
            setattr(existing, field, value)
    session.flush()
    return existing


def register_instrument(
    session: Session,
    provider_instrument: ProviderInstrument,
    data_source: DataSource,
    *,
    issuer: Issuer | None = None,
) -> Instrument:
    """Register or update an instrument from provider metadata."""
    existing = find_by_provider_identifier(
        session, data_source.id, provider_instrument.provider_symbol
    ) or find_instrument(
        session, provider_instrument.symbol, exchange_code=provider_instrument.exchange_code
    )

    calendar = get_calendar(provider_instrument.mic or provider_instrument.exchange_code)
    timezone = provider_instrument.timezone or calendar.timezone

    if existing is None:
        instrument = Instrument(
            symbol=provider_instrument.symbol.upper(),
            exchange_code=provider_instrument.exchange_code.upper(),
            mic=provider_instrument.mic,
            exchange_name=provider_instrument.exchange_name,
            name=provider_instrument.name,
            instrument_type=provider_instrument.instrument_type,
            currency=provider_instrument.currency.upper(),
            country=provider_instrument.country,
            timezone=timezone,
            calendar_code=(provider_instrument.mic or provider_instrument.exchange_code).upper(),
            listed_on=provider_instrument.first_trade_date,
            isin=provider_instrument.isin,
            figi=provider_instrument.figi,
            issuer_id=issuer.id if issuer else None,
            metadata_json=dict(provider_instrument.raw or {}),
        )
        session.add(instrument)
        session.flush()
    else:
        instrument = existing
        if issuer and not instrument.issuer_id:
            instrument.issuer_id = issuer.id
        if not instrument.isin and provider_instrument.isin:
            instrument.isin = provider_instrument.isin
        if provider_instrument.name and instrument.name != provider_instrument.name:
            instrument.name = provider_instrument.name
        if not instrument.listed_on and provider_instrument.first_trade_date:
            instrument.listed_on = provider_instrument.first_trade_date
        merged = dict(instrument.metadata_json or {})
        merged.update(provider_instrument.raw or {})
        instrument.metadata_json = merged
        session.flush()

    _ensure_identifier(
        session, instrument, data_source, provider_instrument.provider_symbol
    )
    _ensure_current_ticker(session, instrument, provider_instrument)
    return instrument


def _ensure_identifier(
    session: Session, instrument: Instrument, data_source: DataSource, identifier: str
) -> None:
    exists = session.execute(
        select(InstrumentIdentifier).where(
            InstrumentIdentifier.data_source_id == data_source.id,
            InstrumentIdentifier.identifier == identifier,
        )
    ).scalars().first()
    if exists is None:
        session.add(
            InstrumentIdentifier(
                instrument_id=instrument.id,
                data_source_id=data_source.id,
                identifier=identifier,
                identifier_type="symbol",
            )
        )
        session.flush()


def _ensure_current_ticker(
    session: Session, instrument: Instrument, provider_instrument: ProviderInstrument
) -> None:
    """Ensure an open-ended ticker row exists for the instrument's symbol."""
    symbol = provider_instrument.symbol.upper()
    exists = session.execute(
        select(InstrumentTicker).where(
            InstrumentTicker.instrument_id == instrument.id,
            func.upper(InstrumentTicker.symbol) == symbol,
            InstrumentTicker.valid_to.is_(None),
        )
    ).scalars().first()
    if exists is not None:
        return
    session.add(
        InstrumentTicker(
            instrument_id=instrument.id,
            symbol=symbol,
            exchange_code=instrument.exchange_code,
            # Without a documented assignment date, the listing date is the
            # earliest defensible start. Falling back to "today" would make a
            # historical lookup of this ticker fail.
            valid_from=provider_instrument.first_trade_date or dt.date(1970, 1, 1),
            valid_to=None,
            note="Current symbol recorded at registration.",
        )
    )
    session.flush()


def record_ticker_change(
    session: Session,
    instrument: Instrument,
    *,
    old_symbol: str,
    new_symbol: str,
    effective_date: dt.date,
    note: str | None = None,
) -> None:
    """Close the old ticker's validity range and open the new one.

    Together with ``resolve_symbol_at`` this is what makes a pre-rename
    historical analysis resolve correctly.
    """
    old = session.execute(
        select(InstrumentTicker).where(
            InstrumentTicker.instrument_id == instrument.id,
            func.upper(InstrumentTicker.symbol) == old_symbol.upper(),
            InstrumentTicker.valid_to.is_(None),
        )
    ).scalars().first()
    if old is not None:
        old.valid_to = effective_date - dt.timedelta(days=1)

    session.add(
        InstrumentTicker(
            instrument_id=instrument.id,
            symbol=new_symbol.upper(),
            exchange_code=instrument.exchange_code,
            valid_from=effective_date,
            valid_to=None,
            note=note or f"Renamed from {old_symbol.upper()}.",
        )
    )
    instrument.symbol = new_symbol.upper()
    session.flush()


def search_instruments(
    session: Session,
    query: str,
    *,
    limit: int = 20,
    instrument_types: tuple[str, ...] | None = None,
    only_with_prices: bool = False,
) -> list[Instrument]:
    """Rank instruments for a search box.

    Ranking is explicit rather than relying on database text search so the
    behaviour is identical on any backend: exact symbol first, then symbol
    prefix, then name prefix, then substring matches.
    """
    q = query.strip()
    if not q:
        return []
    upper = q.upper()
    pattern = f"%{q}%"

    stmt = select(Instrument).where(
        or_(
            func.upper(Instrument.symbol).like(f"{upper}%"),
            Instrument.name.ilike(pattern),
            Instrument.isin == upper,
        )
    )
    if instrument_types:
        stmt = stmt.where(Instrument.instrument_type.in_(instrument_types))
    if only_with_prices:
        stmt = stmt.join(
            InstrumentCoverage, InstrumentCoverage.instrument_id == Instrument.id
        ).where(
            InstrumentCoverage.data_kind == "prices",
            InstrumentCoverage.observation_count > 0,
        ).distinct()

    # Over-fetch so ranking has candidates to order; the cap keeps it bounded.
    candidates = list(session.execute(stmt.limit(max(limit * 10, 100))).scalars().all())

    def rank(inst: Instrument) -> tuple[int, int, str]:
        symbol = inst.symbol.upper()
        name_upper = inst.name.upper()
        if symbol == upper:
            tier = 0
        elif symbol.startswith(upper):
            tier = 1
        elif name_upper.startswith(upper):
            tier = 2
        elif upper in symbol:
            tier = 3
        else:
            tier = 4
        # Prefer active instruments, then shorter symbols (AAPL over AAPL34).
        return (tier, 0 if inst.is_active else 1, symbol)

    candidates.sort(key=rank)
    return candidates[:limit]


def update_coverage(
    session: Session,
    instrument_id: uuid.UUID,
    data_source_id: uuid.UUID,
    *,
    data_kind: str,
    resolution: str | None,
    first_observation: dt.datetime | None,
    last_observation: dt.datetime | None,
    observation_count: int,
    refreshed_at: dt.datetime,
    known_gap_count: int = 0,
    notes: str | None = None,
) -> InstrumentCoverage:
    """Record what we hold, so the UI never implies uniform coverage."""
    existing = session.execute(
        select(InstrumentCoverage).where(
            InstrumentCoverage.instrument_id == instrument_id,
            InstrumentCoverage.data_source_id == data_source_id,
            InstrumentCoverage.data_kind == data_kind,
            InstrumentCoverage.resolution == resolution,
        )
    ).scalars().first()

    if existing is None:
        coverage = InstrumentCoverage(
            instrument_id=instrument_id,
            data_source_id=data_source_id,
            data_kind=data_kind,
            resolution=resolution,
            first_observation=first_observation,
            last_observation=last_observation,
            observation_count=observation_count,
            last_refreshed_at=refreshed_at,
            known_gap_count=known_gap_count,
            notes=notes,
        )
        session.add(coverage)
        session.flush()
        return coverage

    if first_observation and (
        existing.first_observation is None or first_observation < existing.first_observation
    ):
        existing.first_observation = first_observation
    if last_observation and (
        existing.last_observation is None or last_observation > existing.last_observation
    ):
        existing.last_observation = last_observation
    existing.observation_count = observation_count
    existing.last_refreshed_at = refreshed_at
    existing.known_gap_count = known_gap_count
    if notes:
        existing.notes = notes
    session.flush()
    return existing
