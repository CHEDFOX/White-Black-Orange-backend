"""EODHD (EOD Historical Data) adapter -- the platform's primary market-data source.

Chosen as the default because it is an officially documented REST API with a
published free tier, which makes its licensing posture far clearer than a
scraped browser endpoint. The literal token ``demo`` serves a small fixed
symbol set, so a fresh checkout has real market data without a signup step;
``EODHD_API_KEY`` unlocks full coverage without any code change.

Normalisation decisions
-----------------------
* **Decimal from the string.** The JSON is parsed with
  ``parse_float=Decimal`` so ``187.15`` becomes ``Decimal("187.15")`` and never
  passes through a float. This is the only way to keep an exact tape price.
* **Raw and adjusted are separate series.** ``close`` is the exchange print;
  ``adjusted_close`` is EODHD's split-and-dividend adjusted figure. They are
  ingested under different ``adjustment_basis`` values. For the adjusted basis
  the whole bar is scaled by ``adjusted_close / close`` so OHLC stays coherent.
* **Daily bars get calendar-derived timestamps.** The API returns a date, not an
  instant. ``event_time`` is that session's open and ``bar_close_time`` its
  close, both from the instrument's trading calendar -- not midnight UTC, which
  would place a US session on the wrong day for an Asian viewer.
* **Dividend announcement dates are preserved.** EODHD supplies
  ``declarationDate``, which is when a dividend became *knowable*. That is the
  timestamp point-in-time analysis needs; the ex-date is when it took effect.
* **Intraday is only served where the API serves it.** No daily bar is ever
  subdivided into minute bars.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
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

PROVIDER_KEY = "eodhd"

# Symbols the public ``demo`` token serves. Used to produce an actionable error
# instead of a bare 403 when running without a key.
DEMO_SYMBOLS = frozenset(
    {"AAPL.US", "MSFT.US", "TSLA.US", "AMZN.US", "VTI.US", "BTC-USD.CC", "EURUSD.FOREX"}
)

_EOD_RESOLUTIONS = frozenset({"1d", "1wk", "1mo"})
_INTRADAY_RESOLUTIONS = {"1m": "1m", "5m": "5m", "1h": "1h"}

_PERIOD_TOKENS = {"1d": "d", "1wk": "w", "1mo": "m"}

# EODHD exchange suffix -> MIC. The suffix is the venue, so ``BMW.XETRA`` and
# ``BMW.US`` resolve to two different instruments, as they must.
_SUFFIX_TO_MIC: dict[str, str] = {
    "US": "XNAS",
    "NASDAQ": "XNAS",
    "NYSE": "XNYS",
    "LSE": "XLON",
    "XETRA": "XETR",
    "F": "XETR",
    "PA": "XPAR",
    "SW": "XSWX",
    "TO": "XTSE",
    "AU": "XASX",
    "HK": "XHKG",
    "TSE": "XTKS",
}

_SUFFIX_TO_TYPE: dict[str, str] = {
    "CC": InstrumentType.CRYPTO,
    "FOREX": InstrumentType.CURRENCY,
    "INDX": InstrumentType.INDEX,
    "COMM": InstrumentType.COMMODITY,
}

_SUFFIX_CURRENCY: dict[str, str] = {
    "US": "USD",
    "NASDAQ": "USD",
    "NYSE": "USD",
    "LSE": "GBP",
    "XETRA": "EUR",
    "PA": "EUR",
    "SW": "CHF",
    "TO": "CAD",
    "AU": "AUD",
    "HK": "HKD",
    "TSE": "JPY",
    "CC": "USD",
    "FOREX": "USD",
}

CAPABILITIES = ProviderCapabilities(
    provider_key=PROVIDER_KEY,
    display_name="EOD Historical Data (EODHD)",
    kinds=frozenset({"prices", "corporate_actions", "fundamentals", "quotes"}),
    resolutions=_EOD_RESOLUTIONS | frozenset(_INTRADAY_RESOLUTIONS),
    adjustment_bases=frozenset({AdjustmentBasis.RAW, AdjustmentBasis.TOTAL_RETURN}),
    instrument_types=frozenset(
        {
            InstrumentType.EQUITY,
            InstrumentType.ETF,
            InstrumentType.INDEX,
            InstrumentType.FUND,
            InstrumentType.CURRENCY,
            InstrumentType.CRYPTO,
        }
    ),
    exchanges=None,
    max_history=None,
    supports_streaming=False,
    # EODHD serves the current revision of a price series. Bars are immutable
    # once a session closes, so historical price analysis is PIT-safe; the
    # fundamentals endpoint returns latest-restated figures, which is why
    # ``available_at`` for facts is taken from the filing report date and the
    # limitation is disclosed rather than hidden.
    supports_point_in_time=False,
    supports_search=False,
    requires_credential=False,
    rate_limit_per_minute=settings.eodhd_rate_limit_per_minute,
    official_api=True,
    redistribution_allowed=False,
    attribution_required=True,
    attribution_text="Historical market data provided by EODHD (eodhd.com).",
    licence_summary=(
        "Official documented REST API. Free and paid tiers; redistribution "
        "requires a commercial agreement. The public 'demo' token serves a "
        "fixed handful of symbols for evaluation."
    ),
    caveats=(
        "The 'demo' token covers only AAPL.US, MSFT.US, TSLA.US, AMZN.US, "
        "VTI.US, BTC-USD.CC and EURUSD.FOREX. Set EODHD_API_KEY for more.",
        "Adjusted closes use EODHD's own methodology; the raw series is stored "
        "alongside so any adjustment can be recomputed.",
        "The fundamentals endpoint returns the latest restated figures, not "
        "original-release vintages.",
        "Daily bar timestamps are derived from the exchange calendar; the API "
        "itself returns only a date.",
    ),
)


def _decimal(value: Any) -> decimal.Decimal | None:
    if value is None:
        return None
    if isinstance(value, decimal.Decimal):
        return value if value.is_finite() else None
    try:
        d = decimal.Decimal(str(value))
    except (TypeError, ValueError, decimal.InvalidOperation):
        return None
    return d if d.is_finite() else None


def split_symbol(symbol: str) -> tuple[str, str]:
    """Split ``AAPL.US`` into ``("AAPL", "US")``; default the suffix to ``US``."""
    if "." in symbol:
        base, suffix = symbol.rsplit(".", 1)
        return base.upper(), suffix.upper()
    return symbol.upper(), "US"


class EodhdProvider(MarketDataProvider):
    capabilities = CAPABILITIES

    def __init__(self, client: ProviderHttpClient | None = None, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.eodhd_api_key
        self._client = client or ProviderHttpClient(
            base_url=settings.eodhd_base_url,
            provider_key=PROVIDER_KEY,
            headers={"User-Agent": f"StockIntelligencePlatform/0.1 (+{PROVIDER_KEY})"},
            rate_limit_per_minute=settings.eodhd_rate_limit_per_minute,
        )

    @property
    def is_demo(self) -> bool:
        return self.api_key == "demo"

    # --- internals -----------------------------------------------------------

    def _get_json(self, path: str, params: dict) -> Any:
        merged = {"api_token": self.api_key, "fmt": "json", **params}
        response = self._client.get(path, params=merged)
        if response.status_code == 403:
            hint = ""
            if self.is_demo:
                hint = (
                    " The 'demo' token only serves "
                    f"{', '.join(sorted(DEMO_SYMBOLS))}. Set EODHD_API_KEY to "
                    "query other symbols."
                )
            raise ProviderError(
                f"EODHD denied access to {path}.{hint}",
                details={"provider": PROVIDER_KEY, "path": path, "status": 403,
                         "demo_token": self.is_demo},
            )
        if response.status_code == 404:
            raise ProviderError(
                f"EODHD has no data at {path}",
                details={"provider": PROVIDER_KEY, "path": path, "status": 404},
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"EODHD request to {path} failed with status {response.status_code}",
                details={"provider": PROVIDER_KEY, "path": path,
                         "status": response.status_code},
            )
        try:
            # parse_float=Decimal keeps tape prices exact end to end.
            return json.loads(response.text, parse_float=decimal.Decimal)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"EODHD returned non-JSON for {path}: {response.text[:200]}",
                details={"provider": PROVIDER_KEY, "path": path},
            ) from exc

    # --- MarketDataProvider --------------------------------------------------

    def resolve_instrument(
        self, symbol: str, *, exchange_hint: str | None = None
    ) -> ProviderInstrument | None:
        base, suffix = split_symbol(symbol)
        provider_symbol = f"{base}.{suffix}"

        general: dict = {}
        if suffix in {"US", "NASDAQ", "NYSE", "LSE", "XETRA", "PA", "TO", "AU", "HK", "TSE"}:
            try:
                payload = self._get_json(f"/api/fundamentals/{provider_symbol}", {})
                if isinstance(payload, dict):
                    general = payload.get("General") or {}
            except ProviderError as exc:
                log.info("eodhd_fundamentals_unavailable", symbol=provider_symbol, error=str(exc))

        if not general:
            # Fall back to proving the price series exists. Without this an
            # unknown symbol would be registered as a real instrument.
            try:
                probe = self._get_json(
                    f"/api/eod/{provider_symbol}",
                    {"period": "d", "from": (now_utc() - dt.timedelta(days=30)).date().isoformat()},
                )
            except ProviderError:
                return None
            if not isinstance(probe, list) or not probe:
                return None

        exchange_code = _SUFFIX_TO_MIC.get(suffix, suffix)
        inferred_type = _SUFFIX_TO_TYPE.get(suffix, InstrumentType.EQUITY)
        if general.get("Type") == "ETF":
            inferred_type = InstrumentType.ETF
        elif general.get("Type") == "FUND":
            inferred_type = InstrumentType.FUND
        elif general.get("Type") == "INDEX":
            inferred_type = InstrumentType.INDEX

        ipo = general.get("IPODate")
        first_trade: dt.date | None = None
        if isinstance(ipo, str):
            try:
                first_trade = dt.date.fromisoformat(ipo)
            except ValueError:
                first_trade = None

        calendar = get_calendar(exchange_code)
        return ProviderInstrument(
            provider_symbol=provider_symbol,
            symbol=base,
            name=general.get("Name") or base,
            exchange_code=exchange_code,
            currency=(general.get("CurrencyCode") or _SUFFIX_CURRENCY.get(suffix, "USD")).upper(),
            instrument_type=inferred_type,
            mic=_SUFFIX_TO_MIC.get(suffix),
            exchange_name=general.get("Exchange") or suffix,
            country=general.get("CountryISO"),
            timezone=calendar.timezone,
            isin=general.get("ISIN"),
            first_trade_date=first_trade,
            raw={
                "cik": general.get("CIK"),
                "sector": general.get("Sector"),
                "industry": general.get("Industry"),
                "fiscal_year_end": general.get("FiscalYearEnd"),
                "eodhd_type": general.get("Type"),
                "description": general.get("Description"),
                "website": general.get("WebURL"),
            },
        )

    def fetch_bars(self, request: ProviderFetchRequest) -> BarBatch:
        resolution = request.resolution
        if resolution not in self.capabilities.resolutions:
            raise UnsupportedError(
                f"{PROVIDER_KEY} does not serve resolution {resolution!r}; "
                f"supported: {sorted(self.capabilities.resolutions)}"
            )
        if request.adjustment_basis not in self.capabilities.adjustment_bases:
            raise UnsupportedError(
                f"{PROVIDER_KEY} cannot serve adjustment basis "
                f"{request.adjustment_basis!r}"
            )

        start = ensure_utc(request.start, field="start")
        end = ensure_utc(request.end, field="end")
        if end <= start:
            raise UnsupportedError("end must be after start")

        base, suffix = split_symbol(request.symbol)
        provider_symbol = f"{base}.{suffix}"
        instrument = ProviderInstrument(
            provider_symbol=provider_symbol,
            symbol=base,
            name=base,
            exchange_code=_SUFFIX_TO_MIC.get(suffix, suffix),
            currency=_SUFFIX_CURRENCY.get(suffix, "USD"),
            instrument_type=_SUFFIX_TO_TYPE.get(suffix, InstrumentType.EQUITY),
            mic=_SUFFIX_TO_MIC.get(suffix),
            timezone=get_calendar(_SUFFIX_TO_MIC.get(suffix, suffix)).timezone,
        )

        if resolution in _INTRADAY_RESOLUTIONS:
            return self._fetch_intraday(request, instrument, start, end)
        return self._fetch_eod(request, instrument, start, end)

    def _fetch_eod(
        self,
        request: ProviderFetchRequest,
        instrument: ProviderInstrument,
        start: dt.datetime,
        end: dt.datetime,
    ) -> BarBatch:
        rows = self._get_json(
            f"/api/eod/{instrument.provider_symbol}",
            {
                "period": _PERIOD_TOKENS[request.resolution],
                "from": start.date().isoformat(),
                "to": end.date().isoformat(),
                "order": "a",
            },
        )
        if not isinstance(rows, list):
            raise ProviderError(
                f"EODHD returned {type(rows).__name__} for EOD bars, expected a list",
                details={"provider": PROVIDER_KEY, "symbol": instrument.provider_symbol},
            )

        calendar = get_calendar(instrument.mic or instrument.exchange_code)
        want_adjusted = request.adjustment_basis == AdjustmentBasis.TOTAL_RETURN
        bars: list[ProviderBar] = []
        notes: list[str] = []
        dropped_missing = 0
        dropped_invalid = 0
        no_session = 0

        for row in rows:
            if not isinstance(row, dict):
                dropped_invalid += 1
                continue
            raw_date = row.get("date")
            if not isinstance(raw_date, str):
                dropped_invalid += 1
                continue
            try:
                day = dt.date.fromisoformat(raw_date[:10])
            except ValueError:
                dropped_invalid += 1
                continue

            o, h, low, c = (
                _decimal(row.get("open")),
                _decimal(row.get("high")),
                _decimal(row.get("low")),
                _decimal(row.get("close")),
            )
            if None in (o, h, low, c):
                dropped_missing += 1
                continue

            if want_adjusted:
                adj = _decimal(row.get("adjusted_close"))
                if adj is None or c == 0:
                    dropped_missing += 1
                    continue
                factor = adj / c
                exp = -adj.as_tuple().exponent if adj.as_tuple().exponent < 0 else 2
                quant = decimal.Decimal(1).scaleb(-max(exp, 4))
                o = (o * factor).quantize(quant, rounding=decimal.ROUND_HALF_EVEN)
                h = (h * factor).quantize(quant, rounding=decimal.ROUND_HALF_EVEN)
                low = (low * factor).quantize(quant, rounding=decimal.ROUND_HALF_EVEN)
                c = adj

            if h < low or min(o, h, low, c) <= 0:
                dropped_invalid += 1
                continue

            session = calendar.session(day)
            if session is None:
                # The provider reports a bar on a day our calendar calls closed.
                # Keep the bar -- it is the provider's observation, not ours to
                # discard -- but timestamp it from the calendar's nominal hours
                # and record the disagreement.
                no_session += 1
                open_utc = dt.datetime.combine(
                    day, calendar.open_time, tzinfo=calendar.tz
                ).astimezone(UTC)
                close_utc = dt.datetime.combine(
                    day, calendar.close_time, tzinfo=calendar.tz
                ).astimezone(UTC)
            else:
                open_utc, close_utc = session.open_utc, session.close_utc

            if request.resolution == "1wk":
                close_utc = open_utc + dt.timedelta(days=5)
            elif request.resolution == "1mo":
                nxt_month = (day.month % 12) + 1
                nxt_year = day.year + (day.month // 12)
                close_utc = dt.datetime(nxt_year, nxt_month, 1, tzinfo=UTC) - dt.timedelta(days=1)

            bars.append(
                ProviderBar(
                    event_time=open_utc,
                    bar_close_time=close_utc,
                    open=o,
                    high=h,
                    low=low,
                    close=c,
                    volume=_decimal(row.get("volume")),
                    currency=instrument.currency,
                    resolution=request.resolution,
                    adjustment_basis=request.adjustment_basis,
                )
            )

        if dropped_missing:
            notes.append(f"{dropped_missing} row(s) dropped: required price field missing")
        if dropped_invalid:
            notes.append(f"{dropped_invalid} row(s) dropped: unparseable or failed OHLC checks")
        if no_session:
            notes.append(
                f"{no_session} bar(s) fall on days the {calendar.name} calendar treats as "
                "closed; timestamps use nominal session hours"
            )

        bars.sort(key=lambda b: b.event_time)
        return BarBatch(
            instrument=instrument,
            resolution=request.resolution,
            adjustment_basis=request.adjustment_basis,
            bars=tuple(bars),
            availability_lag=dt.timedelta(hours=8),
            requested_start=start,
            requested_end=end,
            notes=tuple(notes),
        )

    def _fetch_intraday(
        self,
        request: ProviderFetchRequest,
        instrument: ProviderInstrument,
        start: dt.datetime,
        end: dt.datetime,
    ) -> BarBatch:
        if request.adjustment_basis != AdjustmentBasis.RAW:
            raise UnsupportedError(
                f"{PROVIDER_KEY} serves intraday bars on the raw basis only; "
                "an adjusted intraday series would have to be synthesised."
            )
        rows = self._get_json(
            f"/api/intraday/{instrument.provider_symbol}",
            {
                "interval": _INTRADAY_RESOLUTIONS[request.resolution],
                "from": int(start.timestamp()),
                "to": int(end.timestamp()),
            },
        )
        if not isinstance(rows, list):
            raise ProviderError(
                "EODHD returned a non-list intraday payload",
                details={"provider": PROVIDER_KEY, "symbol": instrument.provider_symbol},
            )

        step = dt.timedelta(seconds=resolution_seconds(request.resolution))
        bars: list[ProviderBar] = []
        dropped = 0
        for row in rows:
            if not isinstance(row, dict):
                dropped += 1
                continue
            ts = row.get("timestamp")
            o, h, low, c = (
                _decimal(row.get("open")),
                _decimal(row.get("high")),
                _decimal(row.get("low")),
                _decimal(row.get("close")),
            )
            if ts is None or None in (o, h, low, c) or h < low or min(o, h, low, c) <= 0:
                dropped += 1
                continue
            event_time = dt.datetime.fromtimestamp(int(ts), tz=UTC)
            bars.append(
                ProviderBar(
                    event_time=event_time,
                    bar_close_time=event_time + step,
                    open=o,
                    high=h,
                    low=low,
                    close=c,
                    volume=_decimal(row.get("volume")),
                    currency=instrument.currency,
                    resolution=request.resolution,
                    adjustment_basis=AdjustmentBasis.RAW,
                )
            )

        bars.sort(key=lambda b: b.event_time)
        notes = (f"{dropped} intraday row(s) dropped as incomplete",) if dropped else ()
        return BarBatch(
            instrument=instrument,
            resolution=request.resolution,
            adjustment_basis=AdjustmentBasis.RAW,
            bars=tuple(bars),
            availability_lag=dt.timedelta(minutes=15),
            requested_start=start,
            requested_end=end,
            notes=notes,
        )

    def fetch_corporate_actions(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> list[ProviderCorporateAction]:
        base, suffix = split_symbol(symbol)
        provider_symbol = f"{base}.{suffix}"
        window = {"from": start.isoformat(), "to": end.isoformat()}
        out: list[ProviderCorporateAction] = []

        try:
            dividends = self._get_json(f"/api/div/{provider_symbol}", window)
        except ProviderError as exc:
            log.info("eodhd_dividends_unavailable", symbol=provider_symbol, error=str(exc))
            dividends = []
        for row in dividends if isinstance(dividends, list) else []:
            if not isinstance(row, dict):
                continue
            ex_date = _parse_date(row.get("date"))
            amount = _decimal(row.get("value"))
            if ex_date is None or amount is None:
                continue
            out.append(
                ProviderCorporateAction(
                    action_type=CorporateActionType.CASH_DIVIDEND,
                    ex_date=ex_date,
                    # The declaration date is when the dividend became knowable.
                    announced_on=_parse_date(row.get("declarationDate")),
                    record_date=_parse_date(row.get("recordDate")),
                    pay_date=_parse_date(row.get("paymentDate")),
                    cash_amount=amount,
                    currency=(row.get("currency") or instrument_currency(suffix)).upper(),
                    description=row.get("period"),
                    raw={k: str(v) for k, v in row.items()},
                )
            )

        try:
            splits = self._get_json(f"/api/splits/{provider_symbol}", window)
        except ProviderError as exc:
            log.info("eodhd_splits_unavailable", symbol=provider_symbol, error=str(exc))
            splits = []
        for row in splits if isinstance(splits, list) else []:
            if not isinstance(row, dict):
                continue
            ex_date = _parse_date(row.get("date"))
            ratio = row.get("split")
            if ex_date is None or not isinstance(ratio, str) or "/" not in ratio:
                continue
            num_s, den_s = ratio.split("/", 1)
            num, den = _decimal(num_s.strip()), _decimal(den_s.strip())
            if num is None or den is None or den == 0:
                continue
            out.append(
                ProviderCorporateAction(
                    action_type=(
                        CorporateActionType.SPLIT if num >= den else CorporateActionType.REVERSE_SPLIT
                    ),
                    ex_date=ex_date,
                    ratio_numerator=num,
                    ratio_denominator=den,
                    description=ratio,
                    raw={k: str(v) for k, v in row.items()},
                )
            )

        out.sort(key=lambda a: (a.ex_date, a.action_type))
        return out

    def fetch_quote(self, symbol: str) -> ProviderQuote | None:
        base, suffix = split_symbol(symbol)
        try:
            row = self._get_json(f"/api/real-time/{base}.{suffix}", {})
        except ProviderError:
            return None
        if not isinstance(row, dict):
            return None
        ts = row.get("timestamp")
        last = _decimal(row.get("close"))
        if ts is None or last is None:
            return None
        return ProviderQuote(
            event_time=dt.datetime.fromtimestamp(int(ts), tz=UTC),
            currency=instrument_currency(suffix),
            last=last,
            # EODHD documents its real-time feed as delayed on non-live plans.
            # Claiming real-time without verifying the plan would be a false
            # freshness claim in the UI.
            is_delayed=True,
            delay_minutes=None,
            raw={k: str(v) for k, v in row.items()},
        )

    def fetch_fundamentals(self, symbol: str) -> dict | None:
        base, suffix = split_symbol(symbol)
        try:
            payload = self._get_json(f"/api/fundamentals/{base}.{suffix}", {})
        except ProviderError as exc:
            log.info("eodhd_fundamentals_failed", symbol=symbol, error=str(exc))
            return None
        return payload if isinstance(payload, dict) else None

    def health_check(self) -> HealthStatus:
        started = now_utc()
        probe = "AAPL.US"
        try:
            response = self._client.get(
                f"/api/eod/{probe}",
                params={
                    "api_token": self.api_key,
                    "fmt": "json",
                    "period": "d",
                    "from": (started - dt.timedelta(days=7)).date().isoformat(),
                },
            )
        except Exception as exc:
            return HealthStatus(
                provider_key=PROVIDER_KEY, healthy=False, checked_at=started, detail=str(exc)[:500]
            )
        latency = (now_utc() - started).total_seconds() * 1000
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


def instrument_currency(suffix: str) -> str:
    return _SUFFIX_CURRENCY.get(suffix.upper(), "USD")


def _parse_date(value: Any) -> dt.date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None
