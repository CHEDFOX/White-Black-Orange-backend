"""Watchlist endpoints.

Authentication is pluggable; in this build a caller identifies itself with an
``X-Subject`` header and a user row is created on first use. That is adequate
for a single-tenant deployment and is explicitly not an authorisation system.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import CommonGuards, get_db
from app.api.schemas.common import ApiModel
from app.core.errors import NotFoundError
from app.core.timeutil import now_utc
from app.db.models.users import AppUser, Watchlist, WatchlistItem
from app.services.data_access import resolve_instrument

router = APIRouter(prefix="/watchlists", tags=["watchlists"], dependencies=CommonGuards)


def current_user(
    x_subject: str = Header(default="local", description="Caller identity"),
    db: Session = Depends(get_db),
) -> AppUser:
    user = db.execute(
        select(AppUser).where(AppUser.external_subject == x_subject)
    ).scalars().first()
    if user is None:
        user = AppUser(external_subject=x_subject, display_name=x_subject)
        db.add(user)
        db.commit()
    return user


class WatchlistItemOut(ApiModel):
    instrument_id: uuid.UUID
    symbol: str
    name: str
    position: int
    note: str | None = None
    added_at: str


class WatchlistOut(ApiModel):
    id: uuid.UUID
    name: str
    description: str | None = None
    items: list[WatchlistItemOut]


class CreateWatchlist(ApiModel):
    name: str
    description: str | None = None


class AddItem(ApiModel):
    identifier: str
    note: str | None = None


def _out(db: Session, watchlist: Watchlist) -> WatchlistOut:
    from app.db.models.instrument import Instrument

    items: list[WatchlistItemOut] = []
    for item in sorted(watchlist.items, key=lambda i: i.position):
        instrument = db.get(Instrument, item.instrument_id)
        if instrument is None:
            continue
        items.append(
            WatchlistItemOut(
                instrument_id=instrument.id,
                symbol=instrument.symbol,
                name=instrument.name,
                position=item.position,
                note=item.note,
                added_at=item.added_at.isoformat(),
            )
        )
    return WatchlistOut(
        id=watchlist.id,
        name=watchlist.name,
        description=watchlist.description,
        items=items,
    )


@router.get("", response_model=list[WatchlistOut], summary="List watchlists")
def list_watchlists(
    user: AppUser = Depends(current_user), db: Session = Depends(get_db)
) -> list[WatchlistOut]:
    rows = db.execute(
        select(Watchlist).where(Watchlist.user_id == user.id).order_by(Watchlist.name)
    ).scalars().all()
    return [_out(db, w) for w in rows]


@router.post("", response_model=WatchlistOut, status_code=201, summary="Create a watchlist")
def create_watchlist(
    payload: CreateWatchlist = Body(),
    user: AppUser = Depends(current_user),
    db: Session = Depends(get_db),
) -> WatchlistOut:
    existing = db.execute(
        select(Watchlist).where(
            Watchlist.user_id == user.id, Watchlist.name == payload.name
        )
    ).scalars().first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"a watchlist named {payload.name!r} already exists",
        )
    watchlist = Watchlist(
        user_id=user.id, name=payload.name, description=payload.description
    )
    db.add(watchlist)
    db.commit()
    return _out(db, watchlist)


def _get_owned(db: Session, user: AppUser, watchlist_id: uuid.UUID) -> Watchlist:
    watchlist = db.get(Watchlist, watchlist_id)
    if watchlist is None or watchlist.user_id != user.id:
        raise NotFoundError(f"watchlist {watchlist_id} not found")
    return watchlist


@router.post("/{watchlist_id}/items", response_model=WatchlistOut, summary="Add an instrument")
def add_item(
    watchlist_id: uuid.UUID,
    payload: AddItem = Body(),
    user: AppUser = Depends(current_user),
    db: Session = Depends(get_db),
) -> WatchlistOut:
    watchlist = _get_owned(db, user, watchlist_id)
    instrument = resolve_instrument(db, payload.identifier)

    existing = db.execute(
        select(WatchlistItem).where(
            WatchlistItem.watchlist_id == watchlist.id,
            WatchlistItem.instrument_id == instrument.id,
        )
    ).scalars().first()
    if existing is None:
        position = max((i.position for i in watchlist.items), default=-1) + 1
        db.add(
            WatchlistItem(
                watchlist_id=watchlist.id,
                instrument_id=instrument.id,
                position=position,
                note=payload.note,
                added_at=now_utc(),
            )
        )
        db.commit()
        db.refresh(watchlist)
    return _out(db, watchlist)


@router.delete(
    "/{watchlist_id}/items/{instrument_id}",
    response_model=WatchlistOut,
    summary="Remove an instrument",
)
def remove_item(
    watchlist_id: uuid.UUID,
    instrument_id: uuid.UUID,
    user: AppUser = Depends(current_user),
    db: Session = Depends(get_db),
) -> WatchlistOut:
    watchlist = _get_owned(db, user, watchlist_id)
    item = db.execute(
        select(WatchlistItem).where(
            WatchlistItem.watchlist_id == watchlist.id,
            WatchlistItem.instrument_id == instrument_id,
        )
    ).scalars().first()
    if item is not None:
        db.delete(item)
        db.commit()
        db.refresh(watchlist)
    return _out(db, watchlist)


@router.delete("/{watchlist_id}", status_code=204, summary="Delete a watchlist")
def delete_watchlist(
    watchlist_id: uuid.UUID,
    user: AppUser = Depends(current_user),
    db: Session = Depends(get_db),
) -> None:
    watchlist = _get_owned(db, user, watchlist_id)
    db.delete(watchlist)
    db.commit()
