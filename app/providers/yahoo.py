"""Yahoo Finance chart adapter.

LICENSING CAVEAT -- read before enabling in production
-------------------------------------------------------
``query1.finance.yahoo.com/v8/finance/chart`` is an undocumented endpoint used
by Yahoo's own web client. It requires no credential, which makes it usable for
development and evaluation, but it is **not a licensed redistribution feed**.
The adapter therefore declares ``official_api=False`` and
``redistribution_allowed=False``, and the data-source row it registers carries
the same flags so the API can attribute data correctly and refuse to re-serve
it where that matters.

For production use, configure a licensed provider (Tiingo, Polygon, Nasdaq Data
Link, a direct exchange feed) and list it ahead of ``yahoo_chart`` in
``MARKET_DATA_PROVIDERS``. The analysis engines do not change.

Normalisation decisions, stated explicitly because each one can corrupt a
series if done wrongly:

* **Floats to Decimal.** Yahoo serialises prices as float64, so a close of
  332.53 arrives as 332.5299987792969. The payload's ``priceHint`` gives the
  instrument's quoted decimal places; we quantise to it. Without this, every
  price in the database carries binary-float noise.
* **Adjusted vs unadjusted.** ``quote.close`` is the raw print;
  ``adjclose`` is Yahoo's total-return series. They are ingested as two
  separate series with distinct ``adjustment_basis`` values, never merged.
* **Null bars.** Yahoo pads its arrays with ``null`` for sessions it has no
  data for. Those entries are dropped, not forward-filled or zeroed, and
  counted in ``notes`` so the gap detector can see them.
* **Daily timestamps.** For ``1d`` the timestamp is the session open in UTC.
  The bar close is derived from the instrument's trading calendar rather than
  by adding 24 hours, so a half-day is not reported as a full session.
"""

from __future__ import annotations

import datetime as dt
import decimal
from typing import Any

from app.core.calendars import get_calendar
from app.core.config import settings
from app.core.errors import ProviderError, UnsupportedError
from app.core.logging import get_logger
from app.core.timeutil import UTC, ensure_utc, now_utc, resolution_seconds
from app.db.models.enums import AdjustmentBasis, CorporateActionType, InstrumentType
from app.providers.base import (
    BarBatch,
    HealthStatus,
    MarketDataProvider,
    ProviderBar,
    ProviderCapabilities,
    ProviderCorporateAction,
    ProviderFetchRequest,
    ProviderInstrument,
    ProviderQuote,
)
from app.providers.http import ProviderHttpClient

log = get_logger(__name__)

PROVIDER_KEY = "yahoo_chart"

# Yahoo's interval tokens, keyed by our resolution identifiers.
_INTERVALS: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}

# Yahoo caps intraday history. Requesting more silently returns less, so the
# adapter refuses rather than returning a short series that looks complete.
_INTRADAY_MAX_HISTORY: dict[str, dt.timedelta] = {
    "1m": dt.timedelta(days=7),
    "5m": dt.timedelta(days=58),
    "15m": dt.timedelta(days=58),
    "30m": dt.timedelta(days=58),
    "1h": dt.timedelta(days=720),
}

_EXCHANGE_TO_MIC: dict[str, str] = {
    "NMS": "XNAS",
    "NGM": "XNAS",
    "NCM": "XNAS",
    "NYQ": "XNYS",
    "PCX": "ARCX",
    "ASE": "XASE",
    "BTS": "BATS",
    "LSE": "XLON",
    "GER": "XETR",
    "PAR": "XPAR",
    "JPX": "XTKS",
    "HKG": "XHKG",
    "EBS": "XSWX",
    "TOR": "XTSE",
    "ASX": "XASX",
}

_INSTRUMENT_TYPES: dict[str, str] = {
    "EQUITY": InstrumentType.EQUITY,
    "ETF": InstrumentType.ETF,
    "INDEX": InstrumentType.INDEX,
    "MUTUALFUND": InstrumentType.FUND,
    "CURRENCY": InstrumentType.CURRENCY,
    "CRYPTOCURRENCY": InstrumentType.CRYPTO,
    "FUTURE": InstrumentType.FUTURE,
    "OPTION": InstrumentType.OPTION,
}

CAPABILITIES = ProviderCapabilities(
    provider_key=PROVIDER_KEY,
    display_name="Yahoo Finance chart endpoint (unofficial)",
    kinds=frozenset({"prices", "corporate_actions", "quotes"}),
    resolutions=frozenset(_INTERVALS),
    adjustment_bases=frozenset({AdjustmentBasis.RAW, AdjustmentBasis.TOTAL_RETURN}),
    instrument_types=frozenset(_INSTRUMENT_TYPES.values()),
    exchanges=None,
    max_history=None,
    supports_streaming=False,
    # Yahoo serves only the current revision of a series. We can therefore
    # record when *we* first saw a value, but we cannot reconstruct what Yahoo
    # published on a past date. Historical price analysis is still PIT-safe
    # because bars are immutable once a session closes; fundamentals are not,
    # which is why those come from SEC EDGAR instead.
    supports_point_in_time=False,
    supports_search=False,
    requires_credential=False,
    rate_limit_per_minute=settings.yahoo_rate_limit_per_minute,
    official_api=False,
    redistribution_allowed=False,
    attribution_required=True,
    attribution_text="Price data via Yahoo Finance.",
    licence_summary=(
        "Undocumented public endpoint. No redistribution licence. Suitable for "
        "development and personal evaluation only; replace with a licensed feed "
        "before serving third parties."
    ),
    caveats=(
        "Unofficial endpoint: shape and availability can change without notice.",
        "No point-in-time vintages: only the current revision of a series is served.",
        "Adjusted closes use Yahoo's own methodology, which is not published.",
        "Intraday history is capped (1m: ~7 days, 5m-30m: ~60 days, 1h: ~2 years).",
    ),
)


def _as_decimal(value: Any, places: int) -> decimal.Decimal | None:
    """Quantise a provider float to the instrument's quoted precision."""
    if value is None:
        return None
    try:
        d = decimal.Decimal(repr(float(value)))
    except (TypeError, ValueError, decimal.InvalidOperation):
        return None
    if not d.is_finite():
        return None
    quant = decimal.Decimal(1).scaleb(-places)
    return d.quantize(quant, rounding=decimal.ROUND_HALF_EVEN)


def _volume_decimal(value: Any) -> decimal.Decimal | None:
    if value is None:
        return None
    try:
        return decimal.Decimal(int(value))
    except (TypeError, ValueError, decimal.InvalidOperation):
        return None


class YahooChartProvider(MarketDataProvider):
    capabilities = CAPABILITIES

    def __init__(self, client: ProviderHttpClient | None = None) -> None:
        self._client = client or ProviderHttpClient(
            base_url=settings.yahoo_chart_base_url,
            provider_key=PROVIDER_KEY,
            headers={
                # The endpoint rejects requests without a browser-like agent.
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
                ),
                "Accept": "application/json",
            },
            rate_limit_per_minute=settings.yahoo_rate_limit_per_minute,
        )

    # --- internals -----------------------------------------------------------

    def _chart(
        self,
        symbol: str,
        *,
        params: dict,
    ) -> dict:
        response = self._client.get(f"/v8/finance/chart/{symbol}", params=params)
        if response.status_code == 404:
            raise ProviderError(
                f"Yahoo has no chart for symbol {symbol!r}",
                details={"provider": PROVIDER_KEY, "symbol": symbol, "status": 404},
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"Yahoo chart request failed with status {response.status_code}",
                details={"provider": PROVIDER_KEY, "symbol": symbol,
                         "status": response.status_code},
            )
        payload = response.json()
        chart = payload.get("chart") or {}
        if chart.get("error"):
            err = chart["error"]
            raise ProviderError(
                f"Yahoo chart error for {symbol!r}: {err.get('description') or err}",
                details={"provider": PROVIDER_KEY, "symbol": symbol, "error": err},
            )
        results = chart.get("result") or []
        if not results:
            raise ProviderError(
                f"Yahoo returned no chart result for {symbol!r}",
                details={"provider": PROVIDER_KEY, "symbol": symbol},
            )
        return results[0]

    @staticmethod
    def _instrument_from_meta(meta: dict, provider_symbol: str) -> ProviderInstrument:
        exchange = (meta.get("exchangeName") or "").upper()
        first_trade = meta.get("firstTradeDate")
        return ProviderInstrument(
            provider_symbol=provider_symbol,
            symbol=(meta.get("symbol") or provider_symbol).upper(),
            name=meta.get("longName") or meta.get("shortName") or provider_symbol,
            exchange_code=_EXCHANGE_TO_MIC.get(exchange, exchange or "UNKNOWN"),
            currency=(meta.get("currency") or "USD").upper(),
            instrument_type=_INSTRUMENT_TYPES.get(
                (meta.get("instrumentType") or "").upper(), InstrumentType.OTHER
            ),
            mic=_EXCHANGE_TO_MIC.get(exchange),
            exchange_name=meta.get("fullExchangeName"),
            timezone=meta.get("exchangeTimezoneName"),
            first_trade_date=(
                dt.datetime.fromtimestamp(first_trade, tz=UTC).date()
                if isinstance(first_trade, (int, float))
                else None
            ),
            raw={k: meta.get(k) for k in ("priceHint", "validRanges", "dataGranularity")},
        )

    # --- MarketDataProvider --------------------------------------------------

    def resolve_instrument(
        self, symbol: str, *, exchange_hint: str | None = None
    ) -> ProviderInstrument | None:
        try:
            result = self._chart(symbol, params={"range": "5d", "interval": "1d"})
        except ProviderError as exc:
            log.info("yahoo_resolve_failed", symbol=symbol, error=str(exc))
            return None
        return self._instrument_from_meta(result.get("meta") or {}, symbol)

    def fetch_bars(self, request: ProviderFetchRequest) -> BarBatch:
        resolution = request.resolution
        if resolution not in _INTERVALS:
            raise UnsupportedError(
                f"{PROVIDER_KEY} does not serve resolution {resolution!r}; "
                f"supported: {sorted(_INTERVALS)}"
            )
        if request.adjustment_basis not in self.capabilities.adjustment_bases:
            raise UnsupportedError(
                f"{PROVIDER_KEY} cannot serve adjustment basis "
                f"{request.adjustment_basis!r}; supported: "
                f"{sorted(self.capabilities.adjustment_bases)}"
            )

        start = ensure_utc(request.start, field="start")
        end = ensure_utc(request.end, field="end")
        if end <= start:
            raise UnsupportedError("end must be after start")

        notes: list[str] = []
        cap = _INTRADAY_MAX_HISTORY.get(resolution)
        if cap is not None:
            earliest = now_utc() - cap
            if start < earliest:
                raise UnsupportedError(
                    f"{PROVIDER_KEY} serves at most {cap.days} days of {resolution} "
                    f"history; requested start {start.date()} is older than "
                    f"{earliest.date()}. Requesting it would return a shorter "
                    "series than asked for without saying so."
                )

        params = {
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
            "interval": _INTERVALS[resolution],
            "events": "div,split",
            "includeAdjustedClose": "true",
        }
        result = self._chart(request.symbol, params=params)
        meta = result.get("meta") or {}
        instrument = self._instrument_from_meta(meta, request.symbol)
        places = int(meta.get("priceHint") or 2)

        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quote_blocks = indicators.get("quote") or [{}]
        quote = quote_blocks[0] if quote_blocks else {}

        want_adjusted = request.adjustment_basis == AdjustmentBasis.TOTAL_RETURN
        adj_blocks = indicators.get("adjclose") or []
        adj_closes = (adj_blocks[0].get("adjclose") if adj_blocks else None) or []
        if want_adjusted and not adj_closes:
            raise UnsupportedError(
                f"{PROVIDER_KEY} returned no adjusted closes for {request.symbol!r}; "
                "a total-return series cannot be constructed from raw closes alone."
            )

        calendar = get_calendar(instrument.mic or instrument.exchange_code)
        bars: list[ProviderBar] = []
        dropped_null = 0
        dropped_invalid = 0

        for i, ts in enumerate(timestamps):
            o = _as_decimal(_at(quote.get("open"), i), places)
            h = _as_decimal(_at(quote.get("high"), i), places)
            low = _as_decimal(_at(quote.get("low"), i), places)
            c = _as_decimal(_at(quote.get("close"), i), places)
            if None in (o, h, low, c):
                # Yahoo pads missing sessions with nulls. Dropping is the only
                # honest option; filling would invent a price.
                dropped_null += 1
                continue

            if want_adjusted:
                adj = _as_decimal(_at(adj_closes, i), places)
                if adj is None:
                    dropped_null += 1
                    continue
                # Scale the whole bar by the close-to-adjusted-close ratio so
                # OHLC stays internally consistent on the adjusted basis.
                factor = adj / c
                o, h, low, c = (
                    (o * factor).quantize(adj),
                    (h * factor).quantize(adj),
                    (low * factor).quantize(adj),
                    adj,
                )

            if h < low or min(o, h, low, c) <= 0:
                dropped_invalid += 1
                continue

            event_time = dt.datetime.fromtimestamp(int(ts), tz=UTC)
            bars.append(
                ProviderBar(
                    event_time=event_time,
                    bar_close_time=_bar_close(event_time, resolution, calendar),
                    open=o,
                    high=h,
                    low=low,
                    close=c,
                    volume=_volume_decimal(_at(quote.get("volume"), i)),
                    currency=instrument.currency,
                    resolution=resolution,
                    adjustment_basis=request.adjustment_basis,
                    provider_timestamp=None,
                )
            )

        if dropped_null:
            notes.append(f"{dropped_null} bar(s) dropped: provider returned null prices")
        if dropped_invalid:
            notes.append(f"{dropped_invalid} bar(s) dropped: failed OHLC sanity checks")

        bars.sort(key=lambda b: b.event_time)
        return BarBatch(
            instrument=instrument,
            resolution=resolution,
            adjustment_basis=request.adjustment_basis,
            bars=tuple(bars),
            # Daily bars become available after the session closes and the
            # consolidated tape settles. One hour is a conservative, documented
            # assumption, not a measured figure.
            availability_lag=dt.timedelta(hours=1) if resolution in {"1d", "1wk", "1mo"} else dt.timedelta(minutes=1),
            requested_start=start,
            requested_end=end,
            truncated=False,
            notes=tuple(notes),
        )

    def fetch_corporate_actions(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> list[ProviderCorporateAction]:
        params = {
            "period1": int(dt.datetime.combine(start, dt.time.min, tzinfo=UTC).timestamp()),
            "period2": int(dt.datetime.combine(end, dt.time.max, tzinfo=UTC).timestamp()),
            "interval": "1d",
            "events": "div,split",
        }
        result = self._chart(symbol, params=params)
        events = result.get("events") or {}
        out: list[ProviderCorporateAction] = []

        for entry in (events.get("dividends") or {}).values():
            amount = entry.get("amount")
            date = entry.get("date")
            if amount is None or date is None:
                continue
            out.append(
                ProviderCorporateAction(
                    action_type=CorporateActionType.CASH_DIVIDEND,
                    ex_date=dt.datetime.fromtimestamp(int(date), tz=UTC).date(),
                    cash_amount=decimal.Decimal(repr(float(amount))),
                    currency=(result.get("meta") or {}).get("currency", "USD").upper(),
                    raw=dict(entry),
                )
            )

        for entry in (events.get("splits") or {}).values():
            num = entry.get("numerator")
            den = entry.get("denominator")
            date = entry.get("date")
            if not num or not den or date is None:
                continue
            num_d = decimal.Decimal(repr(float(num)))
            den_d = decimal.Decimal(repr(float(den)))
            out.append(
                ProviderCorporateAction(
                    action_type=(
                        CorporateActionType.SPLIT
                        if num_d >= den_d
                        else CorporateActionType.REVERSE_SPLIT
                    ),
                    ex_date=dt.datetime.fromtimestamp(int(date), tz=UTC).date(),
                    ratio_numerator=num_d,
                    ratio_denominator=den_d,
                    description=entry.get("splitRatio"),
                    raw=dict(entry),
                )
            )

        out.sort(key=lambda a: a.ex_date)
        return out

    def fetch_quote(self, symbol: str) -> ProviderQuote | None:
        try:
            result = self._chart(symbol, params={"range": "1d", "interval": "1m"})
        except ProviderError:
            return None
        meta = result.get("meta") or {}
        price = meta.get("regularMarketPrice")
        ts = meta.get("regularMarketTime")
        if price is None or ts is None:
            return None
        places = int(meta.get("priceHint") or 2)
        return ProviderQuote(
            event_time=dt.datetime.fromtimestamp(int(ts), tz=UTC),
            currency=(meta.get("currency") or "USD").upper(),
            last=_as_decimal(price, places),
            # The endpoint does not state its delay. Treating it as delayed is
            # the safe default; claiming real-time would be unverified.
            is_delayed=True,
            delay_minutes=None,
            raw={"regularMarketVolume": meta.get("regularMarketVolume")},
        )

    def health_check(self) -> HealthStatus:
        started = dt.datetime.now(tz=UTC)
        try:
            response = self._client.get(
                "/v8/finance/chart/AAPL", params={"range": "1d", "interval": "1d"}
            )
        except Exception as exc:
            return HealthStatus(
                provider_key=PROVIDER_KEY,
                healthy=False,
                checked_at=started,
                detail=str(exc)[:500],
            )
        latency = (dt.datetime.now(tz=UTC) - started).total_seconds() * 1000
        return HealthStatus(
            provider_key=PROVIDER_KEY,
            healthy=response.status_code == 200,
            checked_at=started,
            latency_ms=latency,
            status_code=response.status_code,
            detail="" if response.status_code == 200 else response.text[:300],
        )

    def close(self) -> None:
        self._client.close()


def _at(seq: Any, index: int) -> Any:
    if not isinstance(seq, list) or index >= len(seq):
        return None
    return seq[index]


def _bar_close(event_time: dt.datetime, resolution: str, calendar) -> dt.datetime:
    """Derive the bar's close instant.

    For intraday bars this is open + interval. For daily and coarser bars the
    session close comes from the trading calendar, so an early close is not
    reported as a full session.
    """
    if resolution in {"1m", "5m", "15m", "30m", "1h"}:
        return event_time + dt.timedelta(seconds=resolution_seconds(resolution))

    local_date = event_time.astimezone(calendar.tz).date()
    session = calendar.session(local_date)
    if session is not None and resolution == "1d":
        return session.close_utc
    if resolution == "1d":
        return event_time + dt.timedelta(hours=6, minutes=30)
    if resolution == "1wk":
        return event_time + dt.timedelta(days=5)
    # 1mo: step to the first of the next month, then back one day.
    year, month = event_time.year, event_time.month
    nxt = dt.datetime(year + (month // 12), (month % 12) + 1, 1, tzinfo=UTC)
    return nxt - dt.timedelta(days=1)
