"""Health, method registry and data-source endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.analysis.registry import registry_snapshot
from app.api.deps import get_db
from app.api.schemas.common import ApiModel
from app.core.config import settings
from app.core.timeutil import now_utc
from app.db.models.instrument import Instrument
from app.db.models.market_data import PriceBar
from app.db.models.provider import DataSource
from app.providers.registry import (
    active_provider_keys,
    capabilities_for,
    known_provider_keys,
    unavailable_reason,
)

router = APIRouter(tags=["system"])


class HealthResponse(ApiModel):
    status: str
    environment: str
    database: str
    instruments: int
    price_bars: int
    checked_at: str


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(db: Session = Depends(get_db)) -> HealthResponse:
    try:
        instruments = db.execute(select(func.count()).select_from(Instrument)).scalar_one()
        bars = db.execute(select(func.count()).select_from(PriceBar)).scalar_one()
        database = "ok"
    except Exception as exc:
        return HealthResponse(
            status="degraded",
            environment=settings.environment,
            database=f"error: {type(exc).__name__}",
            instruments=0,
            price_bars=0,
            checked_at=now_utc().isoformat(),
        )
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        database=database,
        instruments=instruments,
        price_bars=bars,
        checked_at=now_utc().isoformat(),
    )


@router.get("/methods", summary="Registered analysis methods")
def methods(family: str | None = Query(default=None)) -> dict:
    """The analytical method registry.

    Each entry declares its inputs, minimum data, output schema, missing-data
    policy and limitations, so a client can explain any figure it displays.
    """
    all_methods = registry_snapshot()
    if family:
        all_methods = [m for m in all_methods if m["family"] == family]
    families: dict[str, int] = {}
    for method in registry_snapshot():
        families[method["family"]] = families.get(method["family"], 0) + 1
    return {"methods": all_methods, "count": len(all_methods), "families": families}


@router.get("/data-sources", summary="Data source health and licensing")
def data_sources(
    probe: bool = Query(
        default=False,
        description=(
            "Perform a live reachability check against each provider. Off by "
            "default because it consumes provider rate limit."
        ),
    ),
    db: Session = Depends(get_db),
) -> dict:
    """Configured providers, their licensing posture and why any are unavailable."""
    registered = db.execute(select(DataSource)).scalars().all()
    by_key = {s.key: s for s in registered}
    active = set(active_provider_keys())

    entries = []
    for key in known_provider_keys():
        caps = capabilities_for(key)
        reason = unavailable_reason(key)
        row = by_key.get(key)
        entry = {
            "key": key,
            "display_name": caps.display_name,
            "configured": key in settings.enabled_provider_ids,
            "active": key in active,
            "unavailable_reason": reason,
            "kinds": sorted(caps.kinds),
            "resolutions": sorted(caps.resolutions),
            "adjustment_bases": sorted(caps.adjustment_bases),
            "supports_point_in_time": caps.supports_point_in_time,
            "supports_streaming": caps.supports_streaming,
            "official_api": caps.official_api,
            "redistribution_allowed": caps.redistribution_allowed,
            "attribution_required": caps.attribution_required,
            "attribution_text": caps.attribution_text,
            "licence_summary": caps.licence_summary,
            "caveats": list(caps.caveats),
            "rate_limit_per_minute": caps.rate_limit_per_minute,
            "classification": row.classification if row else None,
            "registered": row is not None,
        }
        if probe and key in active:
            try:
                from app.providers.registry import get_provider

                status = get_provider(key).health_check()
                entry["probe"] = {
                    "healthy": status.healthy,
                    "latency_ms": status.latency_ms,
                    "status_code": status.status_code,
                    "detail": status.detail[:200] if status.detail else None,
                    "checked_at": status.checked_at.isoformat(),
                }
            except Exception as exc:
                entry["probe"] = {"healthy": False, "detail": str(exc)[:200]}
        entries.append(entry)

    # Non-market-data sources that are not in the MarketDataProvider registry.
    for key in ("sec_edgar", "fred"):
        row = by_key.get(key)
        if row is None:
            continue
        entries.append({
            "key": key,
            "display_name": row.name,
            "configured": True,
            "active": row.enabled,
            "unavailable_reason": None,
            "kinds": (row.config or {}).get("kinds", []),
            "supports_point_in_time": (row.config or {}).get(
                "supports_point_in_time", False
            ),
            "official_api": row.official_api,
            "redistribution_allowed": row.redistribution_allowed,
            "attribution_required": row.attribution_required,
            "attribution_text": row.attribution_text,
            "licence_summary": row.licence_summary,
            "caveats": (row.caveats or "").split("\n") if row.caveats else [],
            "classification": row.classification,
            "registered": True,
        })

    return {
        "sources": entries,
        "configured_order": settings.enabled_provider_ids,
        "note": (
            "A provider listed as configured but not active has an unmet "
            "credential or is disabled; the reason is given. Analysis never "
            "substitutes one provider's data for another's without recording it."
        ),
    }


@router.get("/limitations", summary="Platform limitations")
def limitations() -> dict:
    """What this platform does not know or does not support.

    Served as an endpoint so a client cannot display analysis without being
    able to show the caveats.
    """
    return {
        "data": [
            "Coverage is limited to instruments that have been ingested; this is "
            "not a global universe.",
            "Macro series are stored without revision vintages unless FRED_API_KEY "
            "is configured, so historical macro analysis uses revised figures.",
            "No options, short-interest, positioning, flow or index-constituent "
            "data is ingested, so the signals that depend on them report as "
            "unavailable rather than as benign.",
            "Intraday history depends entirely on the configured provider's depth.",
            "No news or filings text analysis is active in this build, so event "
            "timelines are limited to corporate actions and filing metadata.",
        ],
        "analysis": [
            "Regime labels are estimates from an explicit rule, not observations.",
            "Statistical relationships are not causal claims. An event preceding "
            "a price move is not evidence that it caused it.",
            "Reported p-values are corrected for the tests the platform performed "
            "in that family, not for tests performed and discarded elsewhere.",
            "Similarity between historical windows is measured on a specific "
            "feature set; it has no established predictive value.",
        ],
        "forecasting": [
            "No model in this build has demonstrated profitable predictive "
            "performance. Several are reference baselines whose purpose is to be "
            "beaten.",
            "A forecast is a distribution under stated assumptions, not a "
            "prediction of what will happen.",
            "Walk-forward results use overlapping windows, so the effective "
            "sample is smaller than the forecast count.",
            "Backtest results are gross of execution costs unless the recorded "
            "assumptions say otherwise.",
        ],
        "bubble": [
            "The assessment detects conditions associated with past speculative "
            "episodes. It cannot distinguish a bubble from a justified re-rating.",
            "No crash date or crash probability is produced.",
        ],
        "regulatory": [
            "This is research and educational tooling, not personalised "
            "investment advice.",
            "No trade execution subsystem exists, and none is planned without a "
            "separate review, explicit authorisation and independent risk controls.",
        ],
    }
