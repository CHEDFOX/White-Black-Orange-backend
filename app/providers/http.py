"""Shared HTTP plumbing for provider adapters: rate limiting, retries, backoff."""

from __future__ import annotations

import random
import threading
import time
from collections import deque

import httpx

from app.core.config import settings
from app.core.errors import ProviderRateLimitError, ProviderUnavailableError
from app.core.logging import get_logger

log = get_logger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class RateLimiter:
    """Thread-safe sliding-window limiter.

    A provider's published limit is a contract, not a suggestion: exceeding it
    is how an API key gets revoked, so the limiter blocks rather than letting
    callers burst and fail.
    """

    def __init__(self, max_calls: int, period_seconds: float = 60.0) -> None:
        self.max_calls = max(1, max_calls)
        self.period = period_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0])
            time.sleep(max(wait, 0.01))


class ProviderHttpClient:
    """httpx wrapper adding rate limiting, bounded retries and jittered backoff."""

    def __init__(
        self,
        *,
        base_url: str,
        provider_key: str,
        headers: dict[str, str] | None = None,
        rate_limit_per_minute: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.provider_key = provider_key
        self.base_url = base_url.rstrip("/")
        self.max_retries = settings.provider_max_retries if max_retries is None else max_retries
        self._limiter = (
            RateLimiter(rate_limit_per_minute) if rate_limit_per_minute else None
        )
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers or {},
            timeout=timeout or settings.provider_request_timeout_seconds,
            follow_redirects=True,
        )

    def get(self, path: str, *, params: dict | None = None) -> httpx.Response:
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= self.max_retries:
            if self._limiter is not None:
                self._limiter.acquire()
            try:
                response = self._client.get(path, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                log.warning(
                    "provider_http_error",
                    provider=self.provider_key,
                    path=path,
                    attempt=attempt,
                    error=str(exc),
                )
            else:
                if response.status_code not in RETRY_STATUS:
                    return response
                last_exc = None
                log.warning(
                    "provider_http_retryable_status",
                    provider=self.provider_key,
                    path=path,
                    attempt=attempt,
                    status=response.status_code,
                )
                if response.status_code == 429 and attempt == self.max_retries:
                    raise ProviderRateLimitError(
                        f"{self.provider_key} rate limit exceeded after {attempt + 1} attempts",
                        details={"provider": self.provider_key, "path": path},
                    )
            attempt += 1
            if attempt <= self.max_retries:
                self._sleep_backoff(attempt)

        detail = str(last_exc) if last_exc else "retryable status codes exhausted"
        raise ProviderUnavailableError(
            f"{self.provider_key} unavailable after {self.max_retries + 1} attempts: {detail}",
            details={"provider": self.provider_key, "path": path},
        )

    def _sleep_backoff(self, attempt: int) -> None:
        base = settings.provider_backoff_base_seconds * (2 ** (attempt - 1))
        # Full jitter: deterministic backoff makes concurrent workers retry in
        # lockstep and re-trip the same rate limit.
        time.sleep(base * (0.5 + random.random() * 0.5))

    def close(self) -> None:
        self._client.close()
