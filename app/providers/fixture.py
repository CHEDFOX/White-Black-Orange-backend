"""Deterministic synthetic provider for automated tests.

Everything this adapter produces is labelled ``DataClassification.SYNTHETIC``
by the ingestion layer, and its data-source row is created with
``classification='synthetic'``. Tests can therefore assert that synthetic bars
never appear in a production query, and a report built on this source is
visibly marked as such.

The generator is a seeded random walk with configurable drift, volatility,
injectable gaps and injectable defects, which is what makes the validation
tests meaningful: a validator that is only ever fed clean data is untested.
"""

from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import math
import random
from dataclasses import dataclass, field

from app.core.calendars import get_calendar
from app.core.errors import UnsupportedError
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

PROVIDER_KEY = "fixture"

CAPABILITIES = ProviderCapabilities(
    provider_key=PROVIDER_KEY,
    display_name="Deterministic synthetic fixture provider (TEST ONLY)",
    kinds=frozenset({"prices", "corporate_actions", "quotes"}),
    resolutions=frozenset({"1m", "5m", "15m", "30m", "1h", "1d", "1wk"}),
    adjustment_bases=frozenset({AdjustmentBasis.RAW, AdjustmentBasis.TOTAL_RETURN}),
    instrument_types=frozenset({InstrumentType.EQUITY, InstrumentType.ETF, InstrumentType.INDEX}),
    exchanges=frozenset({"XNYS", "XNAS"}),
    max_history=None,
    supports_streaming=False,
    # The fixture provider *can* model vintages, which is what makes
    # point-in-time tests possible without a paid ALFRED subscription.
    supports_point_in_time=True,
    supports_search=True,
    requires_credential=False,
    official_api=False,
    redistribution_allowed=True,
    attribution_required=False,
    licence_summary="Synthetic data generated locally. Not market data.",
    caveats=(
        "SYNTHETIC DATA. Every value is generated from a seeded PRNG and has no "
        "relationship to any real instrument. Never serve this as market data.",
    ),
)


@dataclass
class SyntheticSpec:
    """Controls one synthetic instrument's generated series."""

    symbol: str
    name: str = ""
    exchange_code: str = "XNYS"
    currency: str = "USD"
    instrument_type: str = InstrumentType.EQUITY
    start_price: decimal.Decimal = decimal.Decimal("100.00")
    annual_drift: float = 0.08
    annual_volatility: float = 0.25
    seed: int | None = None
    # Sessions to omit entirely, to exercise gap detection.
    skip_dates: frozenset[dt.date] = frozenset()
    # Dates on which to emit a deliberately broken bar (high < low), to
    # exercise the OHLC validator.
    defect_dates: frozenset[dt.date] = frozenset()
    # Dates to emit twice, to exercise duplicate detection.
    duplicate_dates: frozenset[dt.date] = frozenset()
    corporate_actions: tuple[ProviderCorporateAction, ...] = field(default_factory=tuple)

    def resolved_seed(self) -> int:
        if self.seed is not None:
            return self.seed
        digest = hashlib.sha256(self.symbol.encode()).digest()
        return int.from_bytes(digest[:8], "big")


DEFAULT_SPECS: dict[str, SyntheticSpec] = {
    "SYNTH": SyntheticSpec(symbol="SYNTH", name="Synthetic Test Corp", seed=42),
    "SYNTHB": SyntheticSpec(
        symbol="SYNTHB",
        name="Synthetic Benchmark Index",
        instrument_type=InstrumentType.INDEX,
        start_price=decimal.Decimal("4000.00"),
        annual_drift=0.06,
        annual_volatility=0.15,
        seed=7,
    ),
    "SYNTHGAP": SyntheticSpec(
        symbol="SYNTHGAP",
        name="Synthetic Corp With Gaps",
        seed=99,
        skip_dates=frozenset({dt.date(2023, 3, 15), dt.date(2023, 3, 16), dt.date(2023, 3, 17)}),
    ),
    "SYNTHBAD": SyntheticSpec(
        symbol="SYNTHBAD",
        name="Synthetic Corp With Defects",
        seed=123,
        defect_dates=frozenset({dt.date(2023, 6, 14)}),
        duplicate_dates=frozenset({dt.date(2023, 6, 15)}),
    ),
    "SYNTHSPLIT": SyntheticSpec(
        symbol="SYNTHSPLIT",
        name="Synthetic Corp With Split",
        seed=555,
        corporate_actions=(
            ProviderCorporateAction(
                action_type=CorporateActionType.SPLIT,
                ex_date=dt.date(2023, 7, 3),
                announced_on=dt.date(2023, 6, 1),
                ratio_numerator=decimal.Decimal(2),
                ratio_denominator=decimal.Decimal(1),
                description="2:1",
            ),
            ProviderCorporateAction(
                action_type=CorporateActionType.CASH_DIVIDEND,
                ex_date=dt.date(2023, 2, 10),
                announced_on=dt.date(2023, 1, 25),
                pay_date=dt.date(2023, 2, 24),
                cash_amount=decimal.Decimal("0.50"),
                currency="USD",
            ),
        ),
    ),
}

_CENT = decimal.Decimal("0.01")


class FixtureProvider(MarketDataProvider):
    capabilities = CAPABILITIES

    def __init__(self, specs: dict[str, SyntheticSpec] | None = None) -> None:
        self.specs = dict(DEFAULT_SPECS)
        if specs:
            self.specs.update(specs)

    def register(self, spec: SyntheticSpec) -> None:
        self.specs[spec.symbol.upper()] = spec

    def _spec(self, symbol: str) -> SyntheticSpec | None:
        return self.specs.get(symbol.upper())

    def resolve_instrument(
        self, symbol: str, *, exchange_hint: str | None = None
    ) -> ProviderInstrument | None:
        spec = self._spec(symbol)
        if spec is None:
            return None
        calendar = get_calendar(spec.exchange_code)
        return ProviderInstrument(
            provider_symbol=spec.symbol,
            symbol=spec.symbol,
            name=spec.name or spec.symbol,
            exchange_code=spec.exchange_code,
            currency=spec.currency,
            instrument_type=spec.instrument_type,
            mic=spec.exchange_code,
            exchange_name=f"Synthetic {spec.exchange_code}",
            country="US",
            timezone=calendar.timezone,
            first_trade_date=dt.date(2015, 1, 2),
            raw={"synthetic": True, "seed": spec.resolved_seed()},
        )

    def search_instruments(self, query: str, *, limit: int = 20) -> list[ProviderInstrument]:
        q = query.strip().upper()
        out: list[ProviderInstrument] = []
        for symbol, spec in sorted(self.specs.items()):
            if q and q not in symbol and q not in spec.name.upper():
                continue
            inst = self.resolve_instrument(symbol)
            if inst is not None:
                out.append(inst)
            if len(out) >= limit:
                break
        return out

    def fetch_bars(self, request: ProviderFetchRequest) -> BarBatch:
        spec = self._spec(request.symbol)
        if spec is None:
            raise UnsupportedError(
                f"fixture provider has no spec for {request.symbol!r}; "
                f"known: {sorted(self.specs)}"
            )
        if request.resolution not in self.capabilities.resolutions:
            raise UnsupportedError(
                f"fixture provider does not serve resolution {request.resolution!r}"
            )

        start = ensure_utc(request.start, field="start")
        end = ensure_utc(request.end, field="end")
        if end <= start:
            raise UnsupportedError("end must be after start")

        instrument = self.resolve_instrument(spec.symbol)
        assert instrument is not None
        calendar = get_calendar(spec.exchange_code)

        if request.resolution in {"1d", "1wk"}:
            bars = self._daily_bars(spec, calendar, start, end, request)
        else:
            bars = self._intraday_bars(spec, calendar, start, end, request)

        return BarBatch(
            instrument=instrument,
            resolution=request.resolution,
            adjustment_basis=request.adjustment_basis,
            bars=tuple(bars),
            availability_lag=dt.timedelta(minutes=5),
            requested_start=start,
            requested_end=end,
            notes=("SYNTHETIC DATA: generated locally, not market data.",),
        )

    # --- generation ----------------------------------------------------------

    def _walk(self, spec: SyntheticSpec, n: int, steps_per_year: float) -> list[decimal.Decimal]:
        """Seeded geometric random walk, quantised to cents.

        Uses ``random.Random`` rather than NumPy so the sequence is reproducible
        across NumPy versions -- a fixture that changes when a dependency
        upgrades is not a fixture.
        """
        rng = random.Random(spec.resolved_seed())
        dt_step = 1.0 / steps_per_year
        mu = spec.annual_drift
        sigma = spec.annual_volatility
        price = float(spec.start_price)
        out: list[decimal.Decimal] = []
        for _ in range(n):
            shock = rng.gauss(0.0, 1.0)
            price *= math.exp((mu - 0.5 * sigma**2) * dt_step + sigma * math.sqrt(dt_step) * shock)
            price = max(price, 0.01)
            out.append(decimal.Decimal(f"{price:.4f}").quantize(_CENT))
        return out

    def _bar_from_close(
        self,
        rng: random.Random,
        prev_close: decimal.Decimal,
        close: decimal.Decimal,
        spec: SyntheticSpec,
        *,
        defective: bool,
    ) -> tuple[decimal.Decimal, decimal.Decimal, decimal.Decimal, decimal.Decimal]:
        gap = decimal.Decimal(f"{rng.gauss(0, 0.002):.6f}")
        open_px = (prev_close * (decimal.Decimal(1) + gap)).quantize(_CENT)
        open_px = max(open_px, _CENT)
        hi = max(open_px, close)
        lo = min(open_px, close)
        spread = (hi - lo) + (hi * decimal.Decimal(f"{abs(rng.gauss(0, 0.004)):.6f}"))
        high = (hi + spread / 2).quantize(_CENT)
        low = max((lo - spread / 2).quantize(_CENT), _CENT)
        if defective:
            # Deliberately invert high/low so the OHLC validator has something
            # real to catch.
            high, low = low, high
        return open_px, high, low, close

    def _daily_bars(
        self,
        spec: SyntheticSpec,
        calendar,
        start: dt.datetime,
        end: dt.datetime,
        request: ProviderFetchRequest,
    ) -> list[ProviderBar]:
        sessions = calendar.sessions_between(
            start.astimezone(calendar.tz).date(), end.astimezone(calendar.tz).date()
        )
        sessions = [s for s in sessions if s.date not in spec.skip_dates]
        if request.resolution == "1wk":
            weekly: dict[tuple[int, int], object] = {}
            for s in sessions:
                weekly.setdefault(s.date.isocalendar()[:2], s)
            sessions = list(weekly.values())
        if not sessions:
            return []

        steps_per_year = 252.0 if request.resolution == "1d" else 52.0
        closes = self._walk(spec, len(sessions), steps_per_year)
        rng = random.Random(spec.resolved_seed() ^ 0xBEEF)
        bars: list[ProviderBar] = []
        prev_close = spec.start_price

        split_factor = decimal.Decimal(1)
        if request.adjustment_basis == AdjustmentBasis.TOTAL_RETURN:
            for action in spec.corporate_actions:
                if action.action_type == CorporateActionType.SPLIT and action.ratio_numerator:
                    split_factor *= action.ratio_numerator / (action.ratio_denominator or 1)

        for session, close in zip(sessions, closes, strict=True):
            o, h, low, c = self._bar_from_close(
                rng, prev_close, close, spec, defective=session.date in spec.defect_dates
            )
            prev_close = close
            if request.adjustment_basis == AdjustmentBasis.TOTAL_RETURN and split_factor != 1:
                o, h, low, c = (
                    (o / split_factor).quantize(decimal.Decimal("0.0001")),
                    (h / split_factor).quantize(decimal.Decimal("0.0001")),
                    (low / split_factor).quantize(decimal.Decimal("0.0001")),
                    (c / split_factor).quantize(decimal.Decimal("0.0001")),
                )
            volume = decimal.Decimal(int(abs(rng.gauss(5_000_000, 1_200_000))) + 1)
            bar = ProviderBar(
                event_time=session.open_utc,
                bar_close_time=session.close_utc,
                open=o,
                high=h,
                low=low,
                close=c,
                volume=volume,
                currency=spec.currency,
                resolution=request.resolution,
                adjustment_basis=request.adjustment_basis,
            )
            bars.append(bar)
            if session.date in spec.duplicate_dates:
                bars.append(bar)
        return bars

    def _intraday_bars(
        self,
        spec: SyntheticSpec,
        calendar,
        start: dt.datetime,
        end: dt.datetime,
        request: ProviderFetchRequest,
    ) -> list[ProviderBar]:
        step = dt.timedelta(seconds=resolution_seconds(request.resolution))
        slots: list[dt.datetime] = []
        for session in calendar.sessions_between(
            start.astimezone(calendar.tz).date(), end.astimezone(calendar.tz).date()
        ):
            if session.date in spec.skip_dates:
                continue
            ts = session.open_utc
            while ts < session.close_utc:
                if start <= ts < end:
                    slots.append(ts)
                ts += step
        if not slots:
            return []

        bars_per_year = (6.5 * 3600 / step.total_seconds()) * 252
        closes = self._walk(spec, len(slots), bars_per_year)
        rng = random.Random(spec.resolved_seed() ^ 0xCAFE)
        bars: list[ProviderBar] = []
        prev_close = spec.start_price
        for ts, close in zip(slots, closes, strict=True):
            o, h, low, c = self._bar_from_close(rng, prev_close, close, spec, defective=False)
            prev_close = close
            bars.append(
                ProviderBar(
                    event_time=ts,
                    bar_close_time=ts + step,
                    open=o,
                    high=h,
                    low=low,
                    close=c,
                    volume=decimal.Decimal(int(abs(rng.gauss(50_000, 15_000))) + 1),
                    currency=spec.currency,
                    resolution=request.resolution,
                    adjustment_basis=request.adjustment_basis,
                )
            )
        return bars

    def fetch_corporate_actions(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> list[ProviderCorporateAction]:
        spec = self._spec(symbol)
        if spec is None:
            return []
        return [a for a in spec.corporate_actions if start <= a.ex_date <= end]

    def fetch_quote(self, symbol: str) -> ProviderQuote | None:
        spec = self._spec(symbol)
        if spec is None:
            return None
        rng = random.Random(spec.resolved_seed() ^ 0xFEED)
        mid = spec.start_price * decimal.Decimal(f"{1 + rng.gauss(0, 0.01):.6f}")
        mid = mid.quantize(_CENT)
        half = (mid * decimal.Decimal("0.0005")).quantize(_CENT)
        return ProviderQuote(
            event_time=now_utc(),
            currency=spec.currency,
            bid=mid - half,
            ask=mid + half,
            last=mid,
            is_delayed=False,
            delay_minutes=0,
        )

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            provider_key=PROVIDER_KEY,
            healthy=True,
            checked_at=now_utc(),
            latency_ms=0.0,
            status_code=200,
            detail="synthetic provider is always available",
        )
