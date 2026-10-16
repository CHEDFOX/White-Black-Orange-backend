"""Domain exceptions mapped to HTTP responses by the API layer."""

from __future__ import annotations


class PlatformError(Exception):
    """Base class for all domain errors."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(PlatformError):
    status_code = 404
    code = "not_found"


class ValidationError(PlatformError):
    status_code = 422
    code = "validation_error"


class InsufficientDataError(PlatformError):
    """Raised when a method's minimum data requirement is not met.

    This is a first-class outcome, not a bug: the platform must say "not enough
    data" instead of producing a number from nothing.
    """

    status_code = 409
    code = "insufficient_data"


class ProviderError(PlatformError):
    status_code = 502
    code = "provider_error"


class ProviderRateLimitError(ProviderError):
    status_code = 429
    code = "provider_rate_limited"


class ProviderUnavailableError(ProviderError):
    status_code = 503
    code = "provider_unavailable"


class LookAheadError(PlatformError):
    """Raised when an analysis attempts to read data published after its cutoff."""

    status_code = 500
    code = "look_ahead_violation"


class ImmutabilityError(PlatformError):
    """Raised on an attempt to modify an immutable record (e.g. a forecast)."""

    status_code = 409
    code = "immutable_record"


class UnsupportedError(PlatformError):
    status_code = 400
    code = "unsupported"
