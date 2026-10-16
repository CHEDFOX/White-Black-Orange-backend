"""Environment-based configuration.

All provider credentials live here and are read on the backend only. Nothing in
this module may be shipped to a browser bundle.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- application ---------------------------------------------------------
    app_name: str = "Stock Intelligence Platform"
    api_v1_prefix: str = "/api/v1"
    environment: Literal["development", "test", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool = True

    # --- database ------------------------------------------------------------
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/sip"
    database_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # --- redis / jobs --------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # --- market data providers ----------------------------------------------
    # Comma-separated, highest priority first. Only providers whose credential
    # requirements are satisfied are activated at runtime.
    market_data_providers: str = "eodhd,yahoo_chart"
    provider_request_timeout_seconds: float = 20.0
    provider_max_retries: int = 3
    provider_backoff_base_seconds: float = 0.5

    # EODHD: official documented REST API. The literal token "demo" works for
    # a small fixed symbol set (AAPL.US, MSFT.US, TSLA.US, AMZN.US, VTI.US,
    # BTC-USD.CC, EURUSD.FOREX) and is the default so a fresh checkout has real
    # data without signup. Set EODHD_API_KEY for full coverage.
    eodhd_base_url: str = "https://eodhd.com"
    eodhd_api_key: str = "demo"
    eodhd_enabled: bool = True
    eodhd_rate_limit_per_minute: int = 50

    # Yahoo chart endpoint: no credential, unofficial, evaluation use only.
    # Kept as a fallback. Yahoo rate-limits aggressively per IP, so the default
    # limit here is deliberately low -- see docs/DATA_SOURCES.md.
    yahoo_chart_base_url: str = "https://query1.finance.yahoo.com"
    yahoo_enabled: bool = True
    yahoo_rate_limit_per_minute: int = 5

    # Licensed providers: adapters activate only when a key is present.
    tiingo_api_key: str | None = None
    alpha_vantage_api_key: str | None = None
    polygon_api_key: str | None = None

    # FRED macro series. The fredgraph CSV endpoint needs no key; the versioned
    # ALFRED endpoint (true point-in-time vintages) does.
    fred_base_url: str = "https://fred.stlouisfed.org"
    fred_api_key: str | None = None
    fred_enabled: bool = True

    # SEC EDGAR requires a descriptive User-Agent with contact details.
    sec_base_url: str = "https://data.sec.gov"
    sec_user_agent: str = "StockIntelligencePlatform/0.1 (contact: set SEC_USER_AGENT)"
    sec_enabled: bool = True
    sec_rate_limit_per_second: float = 8.0

    # --- LLM (optional; explanation only, never a source of figures) ---------
    llm_provider: Literal["null", "anthropic"] = "null"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 2000

    # --- analysis defaults ---------------------------------------------------
    # VTI rather than SPY: the EODHD demo token serves VTI.US but not SPY.US.
    default_benchmark_ticker: str = "VTI"
    min_bars_for_analysis: int = 30
    similarity_window_bars: int = 60
    trading_days_per_year: int = 252

    # --- api behaviour -------------------------------------------------------
    cors_origins: str = "http://localhost:3000"
    max_page_size: int = 1000
    default_page_size: int = 200
    rate_limit_per_minute: int = 240
    require_api_key: bool = False
    api_keys: str = ""

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def enabled_provider_ids(self) -> list[str]:
        return [p.strip() for p in self.market_data_providers.split(",") if p.strip()]

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
