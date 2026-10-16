"""Provider registry and resolution.

Adapters are activated from configuration. One whose credential requirement is
unmet is skipped with a logged reason rather than failing at first use, so the
``/data-sources`` endpoint can report exactly why a provider is absent.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterable

from app.core.config import settings
from app.core.errors import NotFoundError, UnsupportedError
from app.core.logging import get_logger
from app.providers.base import MarketDataProvider, ProviderCapabilities

log = get_logger(__name__)

ProviderFactory = Callable[[], MarketDataProvider]

_FACTORIES: dict[str, ProviderFactory] = {}
_CAPABILITIES: dict[str, ProviderCapabilities] = {}
_CREDENTIAL_CHECKS: dict[str, Callable[[], str | None]] = {}
_instances: dict[str, MarketDataProvider] = {}
_lock = threading.Lock()


def register_provider(
    key: str,
    factory: ProviderFactory,
    capabilities: ProviderCapabilities,
    *,
    credential_check: Callable[[], str | None] | None = None,
) -> None:
    """Register an adapter.

    ``credential_check`` returns None when usable, or a human-readable reason
    why it is not.
    """
    _FACTORIES[key] = factory
    _CAPABILITIES[key] = capabilities
    if credential_check is not None:
        _CREDENTIAL_CHECKS[key] = credential_check


def _bootstrap() -> None:
    if _FACTORIES:
        return

    from app.providers.eodhd import CAPABILITIES as EODHD_CAPS
    from app.providers.eodhd import EodhdProvider
    from app.providers.fixture import CAPABILITIES as FIXTURE_CAPS
    from app.providers.fixture import FixtureProvider
    from app.providers.yahoo import CAPABILITIES as YAHOO_CAPS
    from app.providers.yahoo import YahooChartProvider

    register_provider(
        "eodhd",
        EodhdProvider,
        EODHD_CAPS,
        credential_check=lambda: (
            None
            if settings.eodhd_enabled and settings.eodhd_api_key
            else "EODHD_ENABLED is false or EODHD_API_KEY is empty"
        ),
    )
    register_provider(
        "yahoo_chart",
        YahooChartProvider,
        YAHOO_CAPS,
        credential_check=lambda: (
            None if settings.yahoo_enabled else "YAHOO_ENABLED is false"
        ),
    )
    register_provider("fixture", FixtureProvider, FIXTURE_CAPS)


def known_provider_keys() -> list[str]:
    _bootstrap()
    return sorted(_FACTORIES)


def capabilities_for(key: str) -> ProviderCapabilities:
    _bootstrap()
    try:
        return _CAPABILITIES[key]
    except KeyError as exc:
        raise NotFoundError(f"unknown provider {key!r}") from exc


def unavailable_reason(key: str) -> str | None:
    _bootstrap()
    if key not in _FACTORIES:
        return f"provider {key!r} is not registered"
    check = _CREDENTIAL_CHECKS.get(key)
    return check() if check else None


def get_provider(key: str) -> MarketDataProvider:
    _bootstrap()
    reason = unavailable_reason(key)
    if reason:
        raise UnsupportedError(f"provider {key!r} is not available: {reason}")
    with _lock:
        instance = _instances.get(key)
        if instance is None:
            instance = _FACTORIES[key]()
            _instances[key] = instance
        return instance


def active_provider_keys() -> list[str]:
    """Configured providers, in priority order, that are actually usable."""
    _bootstrap()
    out: list[str] = []
    for key in settings.enabled_provider_ids:
        if key not in _FACTORIES:
            log.warning("provider_not_registered", provider=key)
            continue
        reason = unavailable_reason(key)
        if reason:
            log.info("provider_skipped", provider=key, reason=reason)
            continue
        out.append(key)
    return out


def active_providers() -> list[MarketDataProvider]:
    return [get_provider(k) for k in active_provider_keys()]


def providers_supporting(
    kind: str, *, resolution: str | None = None, keys: Iterable[str] | None = None
) -> list[MarketDataProvider]:
    """Usable providers that serve ``kind`` (and optionally ``resolution``)."""
    candidates = list(keys) if keys is not None else active_provider_keys()
    out: list[MarketDataProvider] = []
    for key in candidates:
        caps = capabilities_for(key)
        if not caps.supports(kind):
            continue
        if resolution is not None and not caps.supports_resolution(resolution):
            continue
        if unavailable_reason(key):
            continue
        out.append(get_provider(key))
    return out


def reset_instances() -> None:
    """Dispose cached adapters. Used between tests."""
    with _lock:
        for instance in _instances.values():
            # Best effort: a failure to close one adapter must not leave the
            # rest cached for the next test.
            with contextlib.suppress(Exception):
                instance.close()
        _instances.clear()
