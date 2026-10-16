"""Instrument search and metadata endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from pydantic import Field
from sqlalchemy.orm import Session

from app.api.deps import CommonGuards, get_db, parse_as_of
from app.api.schemas.common import ApiModel, CoverageEntry, DataFreshness
from app.core.calendars import get_calendar
from app.ingestion.instruments import search_instruments
from app.pit.availability import AsOf
from app.services.data_access import (
    coverage_summary,
    data_freshness,
    resolve_instrument,
)

router = APIRouter(prefix="/instruments", tags=["instruments"], dependencies=CommonGuards)


class InstrumentSummary(ApiModel):
    id: uuid.UUID
    symbol: str
    exchange_code: str
    mic: str | None = None
    exchange_name: str | None = None
    name: str
    instrument_type: str
    currency: str
    country: str | None = None
    is_active: bool
    listed_on: str | None = None
    delisted_on: str | None = None
    isin: str | None = None


class InstrumentDetail(InstrumentSummary):
    timezone: str
    calendar_code: str
    calendar_coverage_note: str
    issuer_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    cik: str | None = None
    website: str | None = None
    description: str | None = None
    coverage: list[CoverageEntry] = Field(default_factory=list)
    freshness: DataFreshness
    ticker_history: list[dict] = Field(default_factory=list)
    # What this instrument genuinely supports, so the UI never implies uniform
    # capability across instruments.
    supported_resolutions: list[str] = Field(default_factory=list)
    supported_adjustment_bases: list[str] = Field(default_factory=list)
    analysis_caveats: list[str] = Field(default_factory=list)


class SearchResponse(ApiModel):
    query: str
    results: list[InstrumentSummary]
    returned: int
    note: str | None = None


def _summary(instrument) -> InstrumentSummary:
    return InstrumentSummary(
        id=instrument.id,
        symbol=instrument.symbol,
        exchange_code=instrument.exchange_code,
        mic=instrument.mic,
        exchange_name=instrument.exchange_name,
        name=instrument.name,
        instrument_type=instrument.instrument_type,
        currency=instrument.currency,
        country=instrument.country,
        is_active=instrument.is_active,
        listed_on=instrument.listed_on.isoformat() if instrument.listed_on else None,
        delisted_on=instrument.delisted_on.isoformat() if instrument.delisted_on else None,
        isin=instrument.isin,
    )


@router.get("/search", response_model=SearchResponse, summary="Search instruments")
def search(
    q: str = Query(min_length=1, max_length=64, description="Ticker or name fragment"),
    instrument_type: str | None = Query(default=None),
    only_with_prices: bool = Query(
        default=False,
        description="Restrict to instruments with ingested price history.",
    ),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> SearchResponse:
    """Rank instruments by exact symbol, symbol prefix, then name."""
    types = (instrument_type,) if instrument_type else None
    results = search_instruments(
        db, q, limit=limit, instrument_types=types, only_with_prices=only_with_prices
    )
    note = None
    if not results:
        note = (
            f"No registered instrument matches {q!r}. The registry only contains "
            "instruments that have been ingested; use the ingestion scripts to add "
            "one, or check the spelling and venue."
        )
    return SearchResponse(
        query=q,
        results=[_summary(i) for i in results],
        returned=len(results),
        note=note,
    )


class ListResponse(ApiModel):
    instruments: list[InstrumentSummary]
    returned: int
    total: int
    note: str | None = None


@router.get("", response_model=ListResponse, summary="List ingested instruments")
def list_instruments(
    only_with_prices: bool = Query(
        default=True,
        description="Restrict to instruments that actually have price history.",
    ),
    instrument_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> ListResponse:
    """Every instrument in the registry.

    The registry holds what has been ingested, not a global universe, so this
    is the honest answer to "what can I analyse here?" -- which a search
    endpoint cannot give.
    """
    from sqlalchemy import func as sa_func
    from sqlalchemy import select

    from app.db.models.instrument import Instrument as InstrumentModel

    stmt = select(InstrumentModel)
    count_stmt = select(sa_func.count()).select_from(InstrumentModel)
    if instrument_type:
        stmt = stmt.where(InstrumentModel.instrument_type == instrument_type)
        count_stmt = count_stmt.where(
            InstrumentModel.instrument_type == instrument_type
        )
    if only_with_prices:
        from app.db.models.instrument import InstrumentCoverage as Coverage

        has_prices = (
            select(Coverage.instrument_id)
            .where(Coverage.data_kind == "prices", Coverage.observation_count > 0)
            .distinct()
            .scalar_subquery()
        )
        stmt = stmt.where(InstrumentModel.id.in_(has_prices))
        count_stmt = count_stmt.where(InstrumentModel.id.in_(has_prices))

    total = db.execute(count_stmt).scalar_one()
    rows = db.execute(
        stmt.order_by(InstrumentModel.symbol).limit(limit).offset(offset)
    ).scalars().all()

    note = None
    if total == 0:
        note = (
            "The registry is empty. Run the backend's seed script to ingest "
            "price history, fundamentals and macro series."
        )
    return ListResponse(
        instruments=[_summary(i) for i in rows],
        returned=len(rows),
        total=total,
        note=note,
    )


@router.get("/{identifier}", response_model=InstrumentDetail, summary="Instrument detail")
def detail(
    identifier: str,
    exchange_code: str | None = Query(default=None),
    as_of: AsOf = Depends(parse_as_of),
    db: Session = Depends(get_db),
) -> InstrumentDetail:
    """Instrument metadata plus what data is actually held for it."""
    instrument = resolve_instrument(db, identifier, as_of=as_of, exchange_code=exchange_code)
    calendar = get_calendar(instrument.calendar_code)
    coverage = coverage_summary(db, instrument.id)
    freshness = data_freshness(db, instrument.id)

    resolutions = sorted(
        {c["resolution"] for c in coverage if c["data_kind"] == "prices" and c["resolution"]}
    )
    bases: list[str] = []
    from sqlalchemy import select

    from app.db.models.market_data import PriceBar

    bases = sorted(
        set(
            db.execute(
                select(PriceBar.adjustment_basis)
                .where(PriceBar.instrument_id == instrument.id)
                .distinct()
            ).scalars().all()
        )
    )

    caveats: list[str] = [calendar.coverage_note] if calendar.coverage_note else []
    if instrument.issuer_id is None:
        caveats.append(
            "No issuer is linked, so fundamental analysis is unavailable for this "
            "instrument. ETFs, indexes and currencies have no issuer by nature."
        )
    if freshness["is_stale"]:
        caveats.append(freshness["note"])
    if not resolutions:
        caveats.append(
            "No price history is ingested for this instrument, so no analysis can "
            "be run until it is."
        )

    issuer = instrument.issuer
    return InstrumentDetail(
        **_summary(instrument).model_dump(),
        timezone=instrument.timezone,
        calendar_code=instrument.calendar_code,
        calendar_coverage_note=calendar.coverage_note,
        issuer_name=issuer.name if issuer else None,
        sector=issuer.sector if issuer else None,
        industry=issuer.industry if issuer else None,
        cik=issuer.cik if issuer else None,
        website=issuer.website if issuer else None,
        description=issuer.description if issuer else None,
        coverage=[CoverageEntry(**c) for c in coverage],
        freshness=DataFreshness(**freshness),
        ticker_history=[
            {
                "symbol": t.symbol,
                "exchange_code": t.exchange_code,
                "valid_from": t.valid_from.isoformat(),
                "valid_to": t.valid_to.isoformat() if t.valid_to else None,
                "note": t.note,
            }
            for t in sorted(instrument.tickers, key=lambda t: t.valid_from)
        ],
        supported_resolutions=resolutions,
        supported_adjustment_bases=bases,
        analysis_caveats=caveats,
    )
