"""Timezone, resolution and trading-calendar helpers.

Every timestamp that crosses a module boundary is timezone-aware UTC. Naive
datetimes are rejected rather than silently localised, because a naive
timestamp in market data is almost always a bug that moves a bar across a
session boundary.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

UTC = dt.timezone.utc

# Resolution identifiers used across the platform. Minute-level identifiers are
# accepted by the schema, but a resolution is only "supported" for an instrument
# when a provider actually delivers it -- never by resampling coarser bars.
RESOLUTIONS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "1d": 86_400,
    "1wk": 604_800,
    "1mo": 2_592_000,  # nominal; month boundaries are computed calendrically
}

# Resolutions that may be produced by aggregating a finer stored resolution.
# The rule is strictly one-directional: fine -> coarse only.
AGGREGATION_SOURCES: dict[str, tuple[str, ...]] = {
    "5m": ("1m",),
    "15m": ("5m", "1m"),
    "30m": ("15m", "5m", "1m"),
    "1h": ("30m", "15m", "5m", "1m"),
    "1d": ("1h", "30m", "15m", "5m", "1m"),
    "1wk": ("1d",),
    "1mo": ("1d",),
}


def ensure_utc(value: dt.datetime, *, field: str = "timestamp") -> dt.datetime:
    """Return ``value`` as timezone-aware UTC, rejecting naive datetimes."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"{field} must be timezone-aware; got naive datetime {value!r}. "
            "Naive timestamps silently misplace bars across session boundaries."
        )
    return value.astimezone(UTC)


def now_utc() -> dt.datetime:
    return dt.datetime.now(tz=UTC)


def to_exchange_time(value: dt.datetime, tz_name: str) -> dt.datetime:
    return ensure_utc(value).astimezone(ZoneInfo(tz_name))


def resolution_seconds(resolution: str) -> int:
    try:
        return RESOLUTIONS[resolution]
    except KeyError as exc:
        raise ValueError(
            f"unknown resolution {resolution!r}; known: {sorted(RESOLUTIONS)}"
        ) from exc


def can_aggregate(source: str, target: str) -> bool:
    """True when ``target`` bars may be built from ``source`` bars."""
    return source in AGGREGATION_SOURCES.get(target, ())


@dataclass(frozen=True, slots=True)
class TradingSession:
    """A single continuous trading session in UTC."""

    date: dt.date
    open_utc: dt.datetime
    close_utc: dt.datetime

    def contains(self, ts: dt.datetime) -> bool:
        return self.open_utc <= ensure_utc(ts) < self.close_utc


class TradingCalendar:
    """A minimal, explicit trading calendar.

    This implementation covers weekday sessions plus an explicit holiday set. It
    is deliberately simple and auditable; ``pandas_market_calendars`` is used
    when available to supply the holiday set for a known exchange. Half-days are
    represented by an explicit early-close map.

    Limitation: a calendar without a curated holiday set will treat holidays as
    trading days. ``coverage_note`` states which case applies so callers can
    report it instead of assuming completeness.
    """

    def __init__(
        self,
        *,
        name: str,
        timezone: str,
        open_time: dt.time = dt.time(9, 30),
        close_time: dt.time = dt.time(16, 0),
        holidays: frozenset[dt.date] = frozenset(),
        early_closes: dict[dt.date, dt.time] | None = None,
        coverage_note: str = "",
    ) -> None:
        self.name = name
        self.timezone = timezone
        self.tz = ZoneInfo(timezone)
        self.open_time = open_time
        self.close_time = close_time
        self.holidays = holidays
        self.early_closes = early_closes or {}
        self.coverage_note = coverage_note

    def is_trading_day(self, day: dt.date) -> bool:
        return day.weekday() < 5 and day not in self.holidays

    def session(self, day: dt.date) -> TradingSession | None:
        if not self.is_trading_day(day):
            return None
        close = self.early_closes.get(day, self.close_time)
        open_local = dt.datetime.combine(day, self.open_time, tzinfo=self.tz)
        close_local = dt.datetime.combine(day, close, tzinfo=self.tz)
        return TradingSession(
            date=day,
            open_utc=open_local.astimezone(UTC),
            close_utc=close_local.astimezone(UTC),
        )

    def sessions_between(self, start: dt.date, end: dt.date) -> list[TradingSession]:
        out: list[TradingSession] = []
        day = start
        while day <= end:
            s = self.session(day)
            if s is not None:
                out.append(s)
            day += dt.timedelta(days=1)
        return out

    def trading_days_between(self, start: dt.date, end: dt.date) -> list[dt.date]:
        return [s.date for s in self.sessions_between(start, end)]
