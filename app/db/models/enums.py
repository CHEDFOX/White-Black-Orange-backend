"""Domain enumerations.

These are stored as short strings rather than native PostgreSQL enums so new
values can be added without a migration that locks the type.
"""

from __future__ import annotations

from enum import StrEnum


class InstrumentType(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    FUND = "fund"
    ADR = "adr"
    REIT = "reit"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    FUTURE = "future"
    OPTION = "option"
    CRYPTO = "crypto"
    OTHER = "other"


class AdjustmentBasis(StrEnum):
    """How a price series has been adjusted.

    RAW is the price printed by the exchange. SPLIT_ONLY adjusts for splits.
    TOTAL_RETURN additionally adjusts for cash distributions. Mixing bases in
    one series silently corrupts every return calculation, so the basis is part
    of the primary key of a bar.
    """

    RAW = "raw"
    SPLIT_ONLY = "split_only"
    TOTAL_RETURN = "total_return"


class CorporateActionType(StrEnum):
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    CASH_DIVIDEND = "cash_dividend"
    SPECIAL_DIVIDEND = "special_dividend"
    STOCK_DIVIDEND = "stock_dividend"
    SPINOFF = "spinoff"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    TICKER_CHANGE = "ticker_change"
    DELISTING = "delisting"
    RIGHTS_ISSUE = "rights_issue"
    OTHER = "other"


class DataQualitySeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DataQualityKind(StrEnum):
    OHLC_INCONSISTENT = "ohlc_inconsistent"
    NEGATIVE_PRICE = "negative_price"
    ZERO_VOLUME = "zero_volume"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    MISSING_SESSION = "missing_session"
    OUTLIER_RETURN = "outlier_return"
    TIMESTAMP_MISALIGNED = "timestamp_misaligned"
    STALE_DATA = "stale_data"
    PROVIDER_DISAGREEMENT = "provider_disagreement"
    UNEXPECTED_GAP = "unexpected_gap"


class ClaimKind(StrEnum):
    """Epistemic status of a statement the platform makes.

    The product promise is that every sentence in a report carries one of these
    labels, so a model estimate is never presented as an observed fact.
    """

    OBSERVED_FACT = "observed_fact"
    DETERMINISTIC_CALCULATION = "deterministic_calculation"
    STATISTICAL_RELATIONSHIP = "statistical_relationship"
    MODEL_ESTIMATE = "model_estimate"
    CAUSAL_HYPOTHESIS = "causal_hypothesis"
    INTERPRETATION = "interpretation"
    SCENARIO = "scenario"
    UNKNOWN = "unknown"


class EvidenceStance(StrEnum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    CONTEXTUAL = "contextual"


class EventCategory(StrEnum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    DIVIDEND = "dividend"
    MERGER_ACQUISITION = "merger_acquisition"
    LEADERSHIP = "leadership"
    PRODUCT = "product"
    REGULATORY = "regulatory"
    LEGAL = "legal"
    ANALYST_ACTION = "analyst_action"
    MACRO_RELEASE = "macro_release"
    CORPORATE_ACTION = "corporate_action"
    FILING = "filing"
    MARKET_STRUCTURE = "market_structure"
    OTHER = "other"


class RegimeLabel(StrEnum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    STRESS = "stress"
    TRANSITION = "transition"
    INDETERMINATE = "indeterminate"


class ForecastTarget(StrEnum):
    LOG_RETURN = "log_return"
    SIMPLE_RETURN = "simple_return"
    REALIZED_VOLATILITY = "realized_volatility"
    DIRECTION = "direction"
    MAX_DRAWDOWN = "max_drawdown"


class ModelLifecycle(StrEnum):
    EXPERIMENTAL = "experimental"
    VALIDATED = "validated"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"
    RETIRED = "retired"


class ValidationStatus(StrEnum):
    UNTESTED = "untested"
    UNIT_TESTED = "unit_tested"
    BACKTESTED = "backtested"
    VALIDATED = "validated"
    FAILED = "failed"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class DataClassification(StrEnum):
    """Keeps synthetic test data from ever being mistaken for market data."""

    PRODUCTION = "production"
    SYNTHETIC = "synthetic"
