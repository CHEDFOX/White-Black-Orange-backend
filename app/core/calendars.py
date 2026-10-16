"""Trading-calendar registry.

Calendars are keyed by exchange code. When ``pandas_market_calendars`` is
installed we take its curated holiday schedule; otherwise we fall back to a
weekday-only calendar and say so in ``coverage_note`` so that downstream
reports can state the limitation rather than imply completeness.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

from app.core.logging import get_logger
from app.core.timeutil import TradingCalendar

log = get_logger(__name__)

# Exchange code -> (pandas_market_calendars name, timezone, open, close)
_EXCHANGE_SPECS: dict[str, tuple[str | None, str, dt.time, dt.time]] = {
    "XNAS": ("NASDAQ", "America/New_York", dt.time(9, 30), dt.time(16, 0)),
    "XNYS": ("NYSE", "America/New_York", dt.time(9, 30), dt.time(16, 0)),
    "ARCX": ("NYSE", "America/New_York", dt.time(9, 30), dt.time(16, 0)),
    "BATS": ("NYSE", "America/New_York", dt.time(9, 30), dt.time(16, 0)),
    "XLON": ("LSE", "Europe/London", dt.time(8, 0), dt.time(16, 30)),
    "XETR": ("XETR", "Europe/Berlin", dt.time(9, 0), dt.time(17, 30)),
    "XPAR": ("EUREX", "Europe/Paris", dt.time(9, 0), dt.time(17, 30)),
    "XTKS": ("JPX", "Asia/Tokyo", dt.time(9, 0), dt.time(15, 0)),
    "XHKG": ("HKEX", "Asia/Hong_Kong", dt.time(9, 30), dt.time(16, 0)),
    "XSWX": ("SIX", "Europe/Zurich", dt.time(9, 0), dt.time(17, 30)),
    "XTSE": ("TSX", "America/Toronto", dt.time(9, 30), dt.time(16, 0)),
    "XASX": ("ASX", "Australia/Sydney", dt.time(10, 0), dt.time(16, 0)),
}

DEFAULT_EXCHANGE = "XNYS"

_HOLIDAY_RANGE = (dt.date(1990, 1, 1), dt.date(2035, 12, 31))


def _load_curated_holidays(
    pmc_name: str,
) -> tuple[frozenset[dt.date], dict[dt.date, dt.time]] | None:
    try:
        import pandas_market_calendars as mcal
    except Exception:  # pragma: no cover - optional dependency
        return None
    try:
        cal = mcal.get_calendar(pmc_name)
        sched = cal.schedule(
            start_date=_HOLIDAY_RANGE[0].isoformat(),
            end_date=_HOLIDAY_RANGE[1].isoformat(),
        )
    except Exception as exc:  # pragma: no cover - upstream data issues
        log.warning("calendar_load_failed", calendar=pmc_name, error=str(exc))
        return None

    trading_days = {d.date() for d in sched.index}
    holidays: set[dt.date] = set()
    day = _HOLIDAY_RANGE[0]
    while day <= _HOLIDAY_RANGE[1]:
        if day.weekday() < 5 and day not in trading_days:
            holidays.add(day)
        day += dt.timedelta(days=1)

    early: dict[dt.date, dt.time] = {}
    tz = cal.tz
    for ts, row in sched.iterrows():
        close_local = row["market_close"].tz_convert(tz)
        early[ts.date()] = close_local.time()
    return frozenset(holidays), early


@lru_cache(maxsize=64)
def get_calendar(exchange: str | None) -> TradingCalendar:
    code = (exchange or DEFAULT_EXCHANGE).upper()
    spec = _EXCHANGE_SPECS.get(code)
    if spec is None:
        log.info("calendar_unknown_exchange", exchange=code)
        return TradingCalendar(
            name=code,
            timezone="UTC",
            coverage_note=(
                f"No calendar is configured for exchange {code}. Sessions are "
                "approximated as weekdays in UTC; holidays and local hours are "
                "not modelled."
            ),
        )

    pmc_name, tz_name, open_t, close_t = spec
    holidays: frozenset[dt.date] = frozenset()
    early: dict[dt.date, dt.time] = {}
    note = (
        f"Weekday-only approximation for {code}: no curated holiday schedule "
        "is loaded, so exchange holidays are treated as trading days."
    )
    if pmc_name:
        loaded = _load_curated_holidays(pmc_name)
        if loaded is not None:
            holidays, early = loaded
            note = (
                f"Curated schedule for {code} from pandas_market_calendars "
                f"({pmc_name}) covering {_HOLIDAY_RANGE[0]}..{_HOLIDAY_RANGE[1]}."
            )

    return TradingCalendar(
        name=code,
        timezone=tz_name,
        open_time=open_t,
        close_time=close_t,
        holidays=holidays,
        early_closes=early,
        coverage_note=note,
    )


def known_exchanges() -> list[str]:
    return sorted(_EXCHANGE_SPECS)
