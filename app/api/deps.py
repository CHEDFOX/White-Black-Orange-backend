"""Shared API dependencies: sessions, auth, rate limiting, common parameters."""

from __future__ import annotations

import datetime as dt
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.timeutil import UTC, now_utc
from app.db.session import get_session_factory
from app.pit.availability import AsOf


def get_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce an API key when configured.

    Disabled by default so a local checkout works without ceremony; required in
    any deployment that sets ``REQUIRE_API_KEY``.
    """
    if not settings.require_api_key:
        return
    if not x_api_key or x_api_key not in settings.api_key_set:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="a valid X-API-Key header is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )


class SlidingWindowLimiter:
    """In-process rate limiter.

    Adequate for a single instance. A multi-instance deployment needs the
    Redis-backed equivalent; this one is honest about being per-process rather
    than pretending to be global.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.max_requests:
                retry_after = int(self.window - (now - hits[0])) + 1
                return False, retry_after
            hits.append(now)
            return True, 0


_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)


def rate_limit(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    client = x_api_key or (request.client.host if request.client else "unknown")
    allowed, retry_after = _limiter.check(client)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"rate limit of {settings.rate_limit_per_minute}/minute exceeded",
            headers={"Retry-After": str(retry_after)},
        )


CommonGuards = [Depends(require_api_key), Depends(rate_limit)]


def parse_as_of(
    as_of: dt.datetime | None = Query(
        default=None,
        description=(
            "Information cutoff for a point-in-time reconstruction. Nothing "
            "published after this instant is used. Omit for a current analysis."
        ),
    ),
) -> AsOf:
    """Build an ``AsOf`` from the query parameter.

    A naive timestamp is interpreted as UTC rather than rejected, because a
    caller passing ``2024-06-01`` means that date and a 422 here would be
    unhelpful. A future cutoff is clamped to now, with the clamp reported by
    the endpoint.
    """
    if as_of is None:
        return AsOf.now(purpose="current_analysis")
    cutoff = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    current = now_utc()
    if cutoff > current:
        return AsOf(cutoff=current, purpose="current_analysis").with_disclosure(
            f"The requested cutoff {cutoff.isoformat()} is in the future and was "
            f"clamped to {current.isoformat()}."
        )
    return AsOf(cutoff=cutoff, purpose="historical_reconstruction")


def pagination(
    limit: int = Query(
        default=settings.default_page_size, ge=1, le=settings.max_page_size
    ),
    offset: int = Query(default=0, ge=0),
) -> tuple[int, int]:
    return limit, offset
