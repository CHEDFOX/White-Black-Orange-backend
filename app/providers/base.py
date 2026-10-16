"""Market-data provider abstraction.

Analysis code never imports a provider. It consumes the normalised dataclasses
below, which is what allows a licensed vendor to replace the default adapter
without touching a single feature module.

Every adapter declares its own capabilities and licensing posture, so the API
can report honestly what is and is not available for a given instrument instead
of failing opaquely or -- worse -- fabricating a value.
"""

from __future__ import annotations

import abc
import datetime as dt
import decimal
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.db.models.enums import AdjustmentBasis, CorporateActionType, InstrumentType


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """What an adapter can actually deliver.

    ``supports_point_in_time`` is the honest flag that decides whether a
    historical analysis built on this source can claim PIT correctness or must
    disclose that it uses the latest-revision view.
    """

    provider_key: str
    display_name: str
    # Any of: prices, corporate_actions, fundamentals, macro, news, quotes
    kinds: frozenset[str]
    resolutions: frozenset[str]
    adjustment_bases: frozenset[str]
    instrument_types: frozenset[str]
    exchanges: frozenset[str] | None  # None means "unknown / not enumerable"
    max_history: dt.timedelta | None
    supports_streaming: bool = False
    supports_point_in_time: bool = False
    supports_search: bool = False
    requires_credential: bool = False
    rate_limit_per_minute: int | None = None
    official_api: bool = True
    redistribution_allowed: bool = False
    attribution_required: bool = True
    attribution_text: str | None = None
    licence_summary: str = ""
    caveats: tuple[str, ...] = ()

    def supports(self, kind: str) -> bool:
        return kind in self.kinds

    def supports_resolution(self, resolution: str) -> bool:
        return resolution in self.resolutions


@dataclass(frozen=True, slots=True)
class ProviderBar:
    """A normalised OHLCV bar as delivered by a provider.

    Prices are ``Decimal`` from the moment they are parsed. Converting a
    provider's decimal string to float and back is how a close of 182.63
    becomes 182.62999999999998 in a chart axis.
    """

    event_time: dt.datetime  # bar open, UTC
    bar_close_time: dt.datetime  # UTC
    open: decimal.Decimal
    high: decimal.Decimal
    low: decimal.Decimal
    close: decimal.Decimal
    volume: decimal.Decimal | None
    currency: str
    resolution: str
    adjustment_basis: str = AdjustmentBasis.RAW
    vwap: decimal.Decimal | None = None
    trade_count: int | None = None
    is_partial: bool = False
    provider_timestamp: dt.datetime | None = None
    raw: dict | None = None


@dataclass(frozen=True, slots=True)
class ProviderInstrument:
    """Instrument metadata as reported by a provider."""

    provider_symbol: str
    symbol: str
    name: str
    exchange_code: str
    currency: str
    instrument_type: str = InstrumentType.EQUITY
    mic: str | None = None
    exchange_name: str | None = None
    country: str | None = None
    timezone: str | None = None
    isin: str | None = None
    figi: str | None = None
    first_trade_date: dt.date | None = None
    raw: dict | None = None


@dataclass(frozen=True, slots=True)
class ProviderCorporateAction:
    action_type: str
    ex_date: dt.date
    announced_on: dt.date | None = None
    pay_date: dt.date | None = None
    record_date: dt.date | None = None
    ratio_numerator: decimal.Decimal | None = None
    ratio_denominator: decimal.Decimal | None = None
    cash_amount: decimal.Decimal | None = None
    currency: str | None = None
    new_symbol: str | None = None
    description: str | None = None
    raw: dict | None = None

    def __post_init__(self) -> None:
        if self.action_type not in set(CorporateActionType):
            raise ValueError(f"unknown corporate action type {self.action_type!r}")


@dataclass(frozen=True, slots=True)
class ProviderQuote:
    event_time: dt.datetime
    currency: str
    bid: decimal.Decimal | None = None
    ask: decimal.Decimal | None = None
    bid_size: decimal.Decimal | None = None
    ask_size: decimal.Decimal | None = None
    last: decimal.Decimal | None = None
    is_delayed: bool = True
    delay_minutes: int | None = None
    raw: dict | None = None


@dataclass(frozen=True, slots=True)
class BarBatch:
    """Bars plus the metadata a caller needs to store them honestly."""

    instrument: ProviderInstrument
    resolution: str
    adjustment_basis: str
    bars: tuple[ProviderBar, ...]
    # Provider delivery latency, added to publication time to get availability.
    availability_lag: dt.timedelta = dt.timedelta(0)
    requested_start: dt.datetime | None = None
    requested_end: dt.datetime | None = None
    truncated: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HealthStatus:
    provider_key: str
    healthy: bool
    checked_at: dt.datetime
    latency_ms: float | None = None
    status_code: int | None = None
    detail: str = ""


@dataclass
class ProviderFetchRequest:
    """Normalised request. Adapters translate this; callers never build URLs."""

    symbol: str
    start: dt.datetime
    end: dt.datetime
    resolution: str = "1d"
    adjustment_basis: str = AdjustmentBasis.RAW
    exchange_hint: str | None = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class SupportsSearch(Protocol):
    def search_instruments(self, query: str, *, limit: int = 20) -> list[ProviderInstrument]: ...


@runtime_checkable
class SupportsCorporateActions(Protocol):
    def fetch_corporate_actions(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> list[ProviderCorporateAction]: ...


@runtime_checkable
class SupportsQuotes(Protocol):
    def fetch_quote(self, symbol: str) -> ProviderQuote | None: ...


class MarketDataProvider(abc.ABC):
    """Base class every market-data adapter implements."""

    capabilities: ProviderCapabilities

    @property
    def key(self) -> str:
        return self.capabilities.provider_key

    @abc.abstractmethod
    def fetch_bars(self, request: ProviderFetchRequest) -> BarBatch:
        """Return normalised bars for the request.

        Must raise ``UnsupportedError`` for a resolution or adjustment basis the
        provider does not serve, rather than substituting a different one.
        """

    @abc.abstractmethod
    def resolve_instrument(
        self, symbol: str, *, exchange_hint: str | None = None
    ) -> ProviderInstrument | None:
        """Return provider metadata for a symbol, or None when unknown."""

    @abc.abstractmethod
    def health_check(self) -> HealthStatus:
        """Cheap reachability probe used by the data-source health endpoint."""

    def close(self) -> None:  # pragma: no cover - adapters override as needed
        return None
