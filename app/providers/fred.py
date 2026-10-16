"""FRED / ALFRED economic-data adapter.

Point-in-time honesty, which is the whole reason this adapter has two modes:

* **Without a key** it reads ``fredgraph.csv``, which serves only the *latest
  revision* of a series. The endpoint accepts a ``vintage_date`` parameter and
  silently ignores it -- verified by requesting GDPC1 as of 2020-05-01 and
  getting today's rebased figures back. The adapter therefore reports
  ``supports_point_in_time=False`` in this mode, and the macro ingestion layer
  records the limitation on every row it writes.
* **With ``FRED_API_KEY``** it uses the official ALFRED API with
  ``realtime_start``/``realtime_end``, which returns genuine vintages: the
  number as it was first published, before revision. Only in this mode may a
  historical macro analysis claim point-in-time correctness.

Getting this wrong is the classic macro backtest bug: using today's revised
GDP to "predict" a 2008 recession that the then-published figures did not yet
show.
"""

from __future__ import annotations

import csv
import datetime as dt
import decimal
import io
from dataclasses import dataclass

from app.core.config import settings
from app.core.errors import ProviderError, UnsupportedError
from app.core.logging import get_logger
from app.core.timeutil import UTC, now_utc
from app.providers.base import HealthStatus, ProviderCapabilities
from app.providers.http import ProviderHttpClient

log = get_logger(__name__)

PROVIDER_KEY = "fred"


@dataclass(frozen=True, slots=True)
class MacroSeriesSpec:
    """A curated macro series with its publication cadence."""

    code: str
    title: str
    frequency: str
    unit: str
    category: str
    # Typical lag between the period the figure describes and its publication.
    # Used as the ``available_at`` offset when no key supplies real vintages.
    typical_publication_lag_days: int
    seasonal_adjustment: str | None = None
    country: str = "US"
    notes: str = ""


# Curated starting set covering the macro factors the analysis engines read.
CURATED_SERIES: dict[str, MacroSeriesSpec] = {
    "DGS10": MacroSeriesSpec(
        "DGS10", "10-Year Treasury Constant Maturity Rate", "daily", "percent",
        "interest_rates", 1, notes="Published next business day."
    ),
    "DGS2": MacroSeriesSpec(
        "DGS2", "2-Year Treasury Constant Maturity Rate", "daily", "percent",
        "interest_rates", 1
    ),
    "DFF": MacroSeriesSpec(
        "DFF", "Federal Funds Effective Rate", "daily", "percent", "interest_rates", 1
    ),
    "T10Y2Y": MacroSeriesSpec(
        "T10Y2Y", "10-Year minus 2-Year Treasury Spread", "daily", "percent",
        "interest_rates", 1, notes="Yield-curve slope; a recession proxy, not a prediction."
    ),
    "CPIAUCSL": MacroSeriesSpec(
        "CPIAUCSL", "Consumer Price Index for All Urban Consumers", "monthly", "index_1982_84",
        "inflation", 14, seasonal_adjustment="seasonally_adjusted",
        notes="Released mid-month for the prior month; revised."
    ),
    "UNRATE": MacroSeriesSpec(
        "UNRATE", "Unemployment Rate", "monthly", "percent", "employment", 7,
        seasonal_adjustment="seasonally_adjusted",
        notes="Released first Friday for the prior month; revised."
    ),
    "PAYEMS": MacroSeriesSpec(
        "PAYEMS", "All Employees, Total Nonfarm", "monthly", "thousands_of_persons",
        "employment", 7, seasonal_adjustment="seasonally_adjusted"
    ),
    "GDPC1": MacroSeriesSpec(
        "GDPC1", "Real Gross Domestic Product", "quarterly", "billions_chained_dollars",
        "growth", 30, seasonal_adjustment="seasonally_adjusted_annual_rate",
        notes="Advance estimate ~30 days after quarter end; revised twice more."
    ),
    "VIXCLS": MacroSeriesSpec(
        "VIXCLS", "CBOE Volatility Index: VIX", "daily", "index", "risk", 1
    ),
    "BAMLH0A0HYM2": MacroSeriesSpec(
        "BAMLH0A0HYM2", "ICE BofA US High Yield Index Option-Adjusted Spread", "daily",
        "percent", "credit", 1, notes="Credit-stress proxy."
    ),
    "DTWEXBGS": MacroSeriesSpec(
        "DTWEXBGS", "Nominal Broad US Dollar Index", "daily", "index", "currency", 2
    ),
    "DCOILWTICO": MacroSeriesSpec(
        "DCOILWTICO", "Crude Oil Prices: West Texas Intermediate", "daily",
        "dollars_per_barrel", "commodity", 1
    ),
}


def _capabilities() -> ProviderCapabilities:
    keyed = bool(settings.fred_api_key)
    return ProviderCapabilities(
        provider_key=PROVIDER_KEY,
        display_name="FRED / ALFRED (Federal Reserve Bank of St. Louis)",
        kinds=frozenset({"macro"}),
        resolutions=frozenset({"1d"}),
        adjustment_bases=frozenset(),
        instrument_types=frozenset(),
        exchanges=None,
        max_history=None,
        supports_point_in_time=keyed,
        requires_credential=False,
        rate_limit_per_minute=60,
        official_api=keyed,
        redistribution_allowed=True,
        attribution_required=True,
        attribution_text=(
            "Economic data from the Federal Reserve Bank of St. Louis (FRED)."
        ),
        licence_summary=(
            "FRED series are mostly US federal government works in the public "
            "domain; some third-party series carry their own terms. Attribution "
            "is required."
        ),
        caveats=(
            ()
            if keyed
            else (
                "No FRED_API_KEY configured: only the latest revision of each "
                "series is available. Historical macro analysis therefore uses "
                "revised figures, not the numbers published at the time.",
                "The fredgraph CSV endpoint accepts vintage_date but ignores it; "
                "real vintages require the keyed ALFRED API.",
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class MacroObservation:
    """One macro value with its period and the instant it became available."""

    series_code: str
    event_time: dt.datetime
    value: decimal.Decimal | None
    unit: str
    published_at: dt.datetime | None
    available_at: dt.datetime
    # True when ``published_at`` came from a real vintage rather than being
    # estimated from the series' typical publication lag.
    publication_known: bool
    revision: int = 0


class FredProvider:
    """FRED adapter. Not a ``MarketDataProvider``: it serves macro series only."""

    def __init__(
        self, client: ProviderHttpClient | None = None, api_key: str | None = None
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.fred_api_key
        self._csv_client = client or ProviderHttpClient(
            base_url=settings.fred_base_url,
            provider_key=PROVIDER_KEY,
            headers={"User-Agent": "StockIntelligencePlatform/0.1"},
            rate_limit_per_minute=60,
            timeout=60.0,
        )
        self._api_client: ProviderHttpClient | None = None
        if self.api_key:
            self._api_client = ProviderHttpClient(
                base_url="https://api.stlouisfed.org",
                provider_key=f"{PROVIDER_KEY}_api",
                headers={"User-Agent": "StockIntelligencePlatform/0.1"},
                rate_limit_per_minute=120,
                timeout=60.0,
            )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return _capabilities()

    @property
    def key(self) -> str:
        return PROVIDER_KEY

    @property
    def has_vintages(self) -> bool:
        return self._api_client is not None

    def series_spec(self, code: str) -> MacroSeriesSpec:
        spec = CURATED_SERIES.get(code.upper())
        if spec is None:
            raise UnsupportedError(
                f"series {code!r} is not in the curated set; add a MacroSeriesSpec "
                f"so its publication cadence is documented. Known: {sorted(CURATED_SERIES)}"
            )
        return spec

    def fetch_observations(
        self,
        code: str,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
        as_of: dt.date | None = None,
    ) -> list[MacroObservation]:
        """Fetch observations.

        ``as_of`` requests the vintage known at that date. Without a key it
        cannot be honoured, and rather than returning revised figures labelled
        as a vintage, the adapter raises.
        """
        spec = self.series_spec(code)
        if as_of is not None and not self.has_vintages:
            raise UnsupportedError(
                f"a point-in-time vintage of {code} as of {as_of} was requested, but no "
                "FRED_API_KEY is configured. The keyless CSV endpoint returns only the "
                "latest revision; serving it as a vintage would be a look-ahead bug."
            )
        if self.has_vintages:
            return self._fetch_via_api(spec, start, end, as_of)
        return self._fetch_via_csv(spec, start, end)

    def _fetch_via_csv(
        self, spec: MacroSeriesSpec, start: dt.date | None, end: dt.date | None
    ) -> list[MacroObservation]:
        params: dict[str, str] = {"id": spec.code}
        if start:
            params["cosd"] = start.isoformat()
        if end:
            params["coed"] = end.isoformat()
        response = self._csv_client.get("/graph/fredgraph.csv", params=params)
        if response.status_code != 200:
            raise ProviderError(
                f"FRED CSV request for {spec.code} failed with {response.status_code}",
                details={"provider": PROVIDER_KEY, "series": spec.code},
            )

        reader = csv.DictReader(io.StringIO(response.text))
        date_field = (reader.fieldnames or ["observation_date"])[0]
        lag = dt.timedelta(days=spec.typical_publication_lag_days)
        out: list[MacroObservation] = []
        for row in reader:
            raw_date = (row.get(date_field) or "").strip()
            raw_value = (row.get(spec.code) or "").strip()
            if not raw_date:
                continue
            try:
                period = dt.date.fromisoformat(raw_date)
            except ValueError:
                continue
            # FRED writes "." for a missing observation. It is missing, not zero.
            value = None if raw_value in {"", "."} else _to_decimal(raw_value)
            event_time = dt.datetime.combine(period, dt.time.min, tzinfo=UTC)
            # Availability is *estimated* from the series' documented cadence.
            # publication_known=False records that this is an estimate.
            out.append(
                MacroObservation(
                    series_code=spec.code,
                    event_time=event_time,
                    value=value,
                    unit=spec.unit,
                    published_at=None,
                    available_at=event_time + lag,
                    publication_known=False,
                )
            )
        return out

    def _fetch_via_api(
        self,
        spec: MacroSeriesSpec,
        start: dt.date | None,
        end: dt.date | None,
        as_of: dt.date | None,
    ) -> list[MacroObservation]:
        assert self._api_client is not None
        params: dict[str, str] = {
            "series_id": spec.code,
            "api_key": self.api_key or "",
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start.isoformat()
        if end:
            params["observation_end"] = end.isoformat()
        if as_of:
            # realtime_start == realtime_end == as_of asks ALFRED for exactly
            # the vintage in force on that date.
            params["realtime_start"] = as_of.isoformat()
            params["realtime_end"] = as_of.isoformat()
        else:
            # Ask for the whole revision history so each vintage is a row.
            params["realtime_start"] = "1776-07-04"
            params["realtime_end"] = "9999-12-31"

        response = self._api_client.get("/fred/series/observations", params=params)
        if response.status_code != 200:
            raise ProviderError(
                f"ALFRED request for {spec.code} failed with {response.status_code}: "
                f"{response.text[:200]}",
                details={"provider": PROVIDER_KEY, "series": spec.code},
            )
        payload = response.json()
        rows = payload.get("observations") or []

        by_period: dict[dt.date, int] = {}
        out: list[MacroObservation] = []
        for row in rows:
            try:
                period = dt.date.fromisoformat(row["date"])
                realtime_start = dt.date.fromisoformat(row["realtime_start"])
            except (KeyError, ValueError):
                continue
            raw_value = (row.get("value") or "").strip()
            value = None if raw_value in {"", "."} else _to_decimal(raw_value)
            revision = by_period.get(period, 0)
            by_period[period] = revision + 1
            published = dt.datetime.combine(realtime_start, dt.time.min, tzinfo=UTC)
            out.append(
                MacroObservation(
                    series_code=spec.code,
                    event_time=dt.datetime.combine(period, dt.time.min, tzinfo=UTC),
                    value=value,
                    unit=spec.unit,
                    published_at=published,
                    available_at=published,
                    publication_known=True,
                    revision=revision,
                )
            )
        out.sort(key=lambda o: (o.event_time, o.revision))
        return out

    def health_check(self) -> HealthStatus:
        started = now_utc()
        try:
            response = self._csv_client.get(
                "/graph/fredgraph.csv",
                params={
                    "id": "DGS10",
                    "cosd": (started - dt.timedelta(days=10)).date().isoformat(),
                },
            )
        except Exception as exc:
            return HealthStatus(
                provider_key=PROVIDER_KEY, healthy=False, checked_at=started, detail=str(exc)[:500]
            )
        return HealthStatus(
            provider_key=PROVIDER_KEY,
            healthy=response.status_code == 200,
            checked_at=started,
            latency_ms=(now_utc() - started).total_seconds() * 1000,
            status_code=response.status_code,
        )

    def close(self) -> None:
        self._csv_client.close()
        if self._api_client is not None:
            self._api_client.close()


def _to_decimal(raw: str) -> decimal.Decimal | None:
    try:
        d = decimal.Decimal(raw)
    except decimal.InvalidOperation:
        return None
    return d if d.is_finite() else None
