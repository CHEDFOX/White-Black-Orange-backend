"""Data-source registration.

Each provider's licensing posture is written to the database once, so every
observation row can point at it and the API can attribute data correctly
without the adapter being loaded.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.enums import DataClassification
from app.db.models.provider import DataSource
from app.providers.base import ProviderCapabilities


def upsert_data_source(
    session: Session,
    capabilities: ProviderCapabilities,
    *,
    kind: str,
    classification: str | None = None,
    priority: int = 100,
) -> DataSource:
    """Create or refresh the ``data_sources`` row for a provider."""
    existing = session.execute(
        select(DataSource).where(DataSource.key == capabilities.provider_key)
    ).scalars().first()

    inferred_classification = classification or (
        DataClassification.SYNTHETIC
        if capabilities.provider_key == "fixture"
        else DataClassification.PRODUCTION
    )

    fields = {
        "name": capabilities.display_name,
        "kind": kind,
        "classification": inferred_classification,
        "licence_summary": capabilities.licence_summary,
        "redistribution_allowed": capabilities.redistribution_allowed,
        "attribution_required": capabilities.attribution_required,
        "attribution_text": capabilities.attribution_text,
        "official_api": capabilities.official_api,
        "caveats": "\n".join(capabilities.caveats) or None,
        "priority": priority,
        "config": {
            "kinds": sorted(capabilities.kinds),
            "resolutions": sorted(capabilities.resolutions),
            "adjustment_bases": sorted(capabilities.adjustment_bases),
            "supports_point_in_time": capabilities.supports_point_in_time,
            "supports_streaming": capabilities.supports_streaming,
            "rate_limit_per_minute": capabilities.rate_limit_per_minute,
        },
    }

    if existing is None:
        source = DataSource(key=capabilities.provider_key, **fields)
        session.add(source)
        session.flush()
        return source

    for key, value in fields.items():
        setattr(existing, key, value)
    session.flush()
    return existing


def get_data_source(session: Session, key: str) -> DataSource | None:
    return session.execute(
        select(DataSource).where(DataSource.key == key)
    ).scalars().first()


def register_all_known_sources(session: Session) -> dict[str, DataSource]:
    """Register every adapter the build knows about, whether or not enabled."""
    from app.providers.eodhd import CAPABILITIES as EODHD
    from app.providers.fixture import CAPABILITIES as FIXTURE
    from app.providers.fred import _capabilities as fred_caps
    from app.providers.sec_edgar import CAPABILITIES as SEC
    from app.providers.yahoo import CAPABILITIES as YAHOO

    out: dict[str, DataSource] = {}
    out["eodhd"] = upsert_data_source(session, EODHD, kind="market_data", priority=10)
    out["yahoo_chart"] = upsert_data_source(session, YAHOO, kind="market_data", priority=50)
    out["sec_edgar"] = upsert_data_source(session, SEC, kind="filings", priority=10)
    out["fred"] = upsert_data_source(session, fred_caps(), kind="macro", priority=10)
    out["fixture"] = upsert_data_source(
        session, FIXTURE, kind="synthetic", classification=DataClassification.SYNTHETIC,
        priority=900,
    )
    return out
