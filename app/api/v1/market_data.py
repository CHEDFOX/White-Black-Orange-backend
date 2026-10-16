"""Price, chart and quote endpoints."""

from __future__ import annotations

import datetime as dt
import decimal
import uuid

from fastapi import APIRouter, Depends, Query
from pydantic import ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import CommonGuards, get_db, parse_as_of
from app.api.schemas.common import ApiModel, Attribution, DataFreshness
from app.core.errors import UnsupportedError
from app.core.timeutil import AGGREGATION_SOURCES, RESOLUTIONS, can_aggregate
from app.db.models.enums import AdjustmentBasis
from app.db.models.market_data import PriceBar
from app.db.models.provider import DataSource
from app.pit.availability import AsOf
from app.pit.queries import data_source_attribution, price_bars_as_of
from app.services.data_access import data_freshness, resolve_instrument

router = APIRouter(prefix="/market-data", tags=["market data"], dependencies=CommonGuards)


class Candle(ApiModel):
    """One bar. Prices are strings to preserve exact decimal values in JSON.

    The short field names are the conventional OHLCV chart keys and keep a
    multi-thousand-bar payload compact. ``low`` is aliased to ``l`` rather than
    named ``l``, which is too easily confused with ``1``.
    """

    t: str
    o: str
    h: str
    low: str = Field(serialization_alias="l", validation_alias="l")
    c: str
    v: str | None = None
    # Present only on a revised bar, so a client can show that the value changed.
    revision: int = 0


class CandleResponse(ApiModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    instrument_id: uuid.UUID
    symbol: str
    resolution: str
    adjustment_basis: str
    currency: str
    as_of: str
    candles: list[Candle]
    count: int
    first: str | None = None
    last: str | None = None
    gaps: list[dict] = Field(default_factory=list)
    quality_issues: list[dict] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)
    attribution: list[Attribution] = Field(default_factory=list)
    freshness: DataFreshness | None = None
    aggregated_from: str | None = None
    disclosures: list[str] = Field(default_factory=list)


class QuoteResponse(ApiModel):
    instrument_id: uuid.UUID
    symbol: str
    price: str | None = None
    currency: str
    at: str | None = None
    is_delayed: bool
    source: str | None = None
    note: str


@router.get("/{identifier}/candles", response_model=CandleResponse, summary="Chart data")
def candles(
    identifier: str,
    start: dt.datetime | None = Query(default=None),
    end: dt.datetime | None = Query(default=None),
    resolution: str = Query(default="1d"),
    adjustment_basis: str = Query(default=AdjustmentBasis.TOTAL_RETURN),
    as_of: AsOf = Depends(parse_as_of),
    limit: int = Query(default=5000, ge=1, le=20000),
    db: Session = Depends(get_db),
) -> CandleResponse:
    """Point-in-time OHLCV bars for charting.

    A resolution with no stored bars is served by aggregating a *finer* stored
    resolution, never by subdividing a coarser one. Daily bars cannot be turned
    into minute bars, and the endpoint says so rather than inventing them.
    """
    if resolution not in RESOLUTIONS:
        raise UnsupportedError(
            f"unknown resolution {resolution!r}; known: {sorted(RESOLUTIONS)}"
        )
    if adjustment_basis not in set(AdjustmentBasis):
        raise UnsupportedError(
            f"unknown adjustment basis {adjustment_basis!r}; "
            f"known: {sorted(AdjustmentBasis)}"
        )

    instrument = resolve_instrument(db, identifier, as_of=as_of)
    result = price_bars_as_of(
        db, instrument.id, as_of,
        start=start, end=end, resolution=resolution,
        adjustment_basis=adjustment_basis, limit=limit,
    )
    rows = result.rows
    aggregated_from: str | None = None
    disclosures: list[str] = []

    if not rows:
        # Try to build the requested resolution from a finer stored one.
        for source_resolution in AGGREGATION_SOURCES.get(resolution, ()):
            finer = price_bars_as_of(
                db, instrument.id, as_of,
                start=start, end=end, resolution=source_resolution,
                adjustment_basis=adjustment_basis,
            )
            if finer.rows:
                rows = _aggregate(finer.rows, resolution)
                aggregated_from = source_resolution
                disclosures.append(
                    f"No {resolution} bars are stored, so they were aggregated from "
                    f"{source_resolution} bars. Aggregation is one-directional: a "
                    "coarser series is never subdivided into a finer one."
                )
                break

    if not rows:
        stored = sorted(
            set(
                db.execute(
                    select(PriceBar.resolution)
                    .where(PriceBar.instrument_id == instrument.id)
                    .distinct()
                ).scalars().all()
            )
        )
        disclosures.append(
            f"No {resolution} bars are available for {instrument.symbol} on the "
            f"{adjustment_basis} basis at {as_of.cutoff.isoformat()}."
            + (f" Stored resolutions: {', '.join(stored)}." if stored else
               " No price history is ingested for this instrument.")
        )
        if stored and not any(can_aggregate(s, resolution) for s in stored):
            disclosures.append(
                f"None of the stored resolutions ({', '.join(stored)}) can be "
                f"aggregated up to {resolution}. Finer bars must be ingested from "
                "a provider that serves them."
            )

    disclosures.extend(result.coverage.notes)
    candles_out = [
        Candle(
            t=r.event_time.isoformat(),
            o=str(r.open),
            h=str(r.high),
            low=str(r.low),
            c=str(r.close),
            v=None if r.volume is None else str(r.volume),
            revision=r.revision,
        )
        for r in rows
    ]

    from app.services.data_access import _load_gaps, _load_quality_issues

    gaps: list[dict] = []
    issues: list[dict] = []
    if rows:
        gaps = _load_gaps(
            db, instrument.id, resolution, rows[0].event_time, rows[-1].event_time
        )
        issues = _load_quality_issues(
            db, instrument.id, resolution, rows[0].event_time, rows[-1].event_time
        )

    source_ids = {r.data_source_id for r in rows if getattr(r, "data_source_id", None)}
    attribution = data_source_attribution(db, source_ids)

    return CandleResponse(
        instrument_id=instrument.id,
        symbol=instrument.symbol,
        resolution=resolution,
        adjustment_basis=adjustment_basis,
        currency=rows[0].currency if rows else instrument.currency,
        as_of=as_of.cutoff.isoformat(),
        candles=candles_out,
        count=len(candles_out),
        first=candles_out[0].t if candles_out else None,
        last=candles_out[-1].t if candles_out else None,
        gaps=gaps,
        quality_issues=issues,
        coverage_notes=list(result.coverage.notes),
        attribution=[Attribution(**a) for a in attribution],
        freshness=DataFreshness(**data_freshness(db, instrument.id)),
        aggregated_from=aggregated_from,
        disclosures=disclosures,
    )


def _aggregate(rows: list[PriceBar], target: str) -> list[PriceBar]:
    """Aggregate finer bars into coarser ones.

    Open is the first bar's open, close the last bar's close, high and low the
    extremes, volume the sum. Volume is summed only when every contributing bar
    reports one: summing over a missing value would understate it silently.
    """
    buckets: dict[dt.datetime, list[PriceBar]] = {}
    for row in rows:
        buckets.setdefault(_bucket_key(row.event_time, target), []).append(row)

    out: list[PriceBar] = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda r: r.event_time)
        volumes = [r.volume for r in group]
        total_volume = (
            sum(volumes, start=decimal.Decimal(0))
            if all(v is not None for v in volumes)
            else None
        )
        first = group[0]
        synthetic = PriceBar(
            instrument_id=first.instrument_id,
            resolution=target,
            adjustment_basis=first.adjustment_basis,
            data_source_id=first.data_source_id,
            event_time=key,
            bar_close_time=group[-1].bar_close_time,
            published_at=group[-1].published_at,
            available_at=max(r.available_at for r in group),
            revision=max(r.revision for r in group),
            open=first.open,
            high=max(r.high for r in group),
            low=min(r.low for r in group),
            close=group[-1].close,
            volume=total_volume,
            currency=first.currency,
            classification=first.classification,
        )
        out.append(synthetic)
    return out


def _bucket_key(ts: dt.datetime, target: str) -> dt.datetime:
    if target == "1wk":
        return (ts - dt.timedelta(days=ts.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if target == "1mo":
        return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if target == "1d":
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds = RESOLUTIONS[target]
    epoch = int(ts.timestamp())
    return dt.datetime.fromtimestamp(epoch - (epoch % seconds), tz=dt.UTC)


@router.get("/{identifier}/latest", response_model=QuoteResponse, summary="Latest price")
def latest(identifier: str, db: Session = Depends(get_db)) -> QuoteResponse:
    """The most recent stored close.

    Deliberately not called a "quote": this is the last completed bar from a
    stored series, not a live market price, and the response says so.
    """
    instrument = resolve_instrument(db, identifier)
    row = db.execute(
        select(PriceBar)
        .where(
            PriceBar.instrument_id == instrument.id,
            PriceBar.resolution == "1d",
            PriceBar.superseded_at.is_(None),
        )
        .order_by(PriceBar.event_time.desc())
        .limit(1)
    ).scalars().first()

    if row is None:
        return QuoteResponse(
            instrument_id=instrument.id,
            symbol=instrument.symbol,
            price=None,
            currency=instrument.currency,
            at=None,
            is_delayed=True,
            source=None,
            note="No price history is stored for this instrument.",
        )

    source_key = db.execute(
        select(DataSource.key).where(DataSource.id == row.data_source_id)
    ).scalars().first()
    freshness = data_freshness(db, instrument.id)
    return QuoteResponse(
        instrument_id=instrument.id,
        symbol=instrument.symbol,
        price=str(row.close),
        currency=row.currency,
        at=row.event_time.isoformat(),
        is_delayed=True,
        source=source_key,
        note=(
            "This is the close of the most recent stored daily bar, not a live "
            f"market price. {freshness['note']}."
        ),
    )
