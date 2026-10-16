"""Evidence-linked report endpoints."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import CommonGuards, get_db, parse_as_of
from app.api.schemas.common import ApiModel
from app.db.models.enums import AdjustmentBasis
from app.pit.availability import AsOf
from app.reporting.generator import generate_report
from app.reporting.llm.base import NarrativeRequest, get_narrative_provider
from app.services.analysis_service import PAST_METHODS, PRESENT_METHODS, analyse
from app.services.data_access import resolve_instrument

router = APIRouter(prefix="/reports", tags=["reports"], dependencies=CommonGuards)


class ReportOut(ApiModel):
    instrument_id: str
    symbol: str
    name: str
    report_kind: str
    title: str
    summary: str
    created_at: str
    information_cutoff: str
    period_start: str | None = None
    period_end: str | None = None
    findings: list[dict]
    limitations: list[str]
    data_coverage: dict
    method_versions: dict[str, str]
    inputs: dict
    attribution: list[dict]
    narrative: str | None = None
    narrative_model: str | None = None
    narrative_status: str
    generator_version: str
    reproduction_key: str
    claim_kind_counts: dict[str, int]
    claim_kind_legend: dict[str, str]


CLAIM_KIND_LEGEND = {
    "observed_fact": "Directly observed in a source record.",
    "deterministic_calculation": (
        "Arithmetic on stored data. Reproducible exactly from the recorded inputs."
    ),
    "statistical_relationship": (
        "An estimated relationship with sampling uncertainty. Not a causal claim."
    ),
    "model_estimate": (
        "The output of a model under stated assumptions. Not an observation."
    ),
    "causal_hypothesis": (
        "A proposed explanation. Temporal order alone is not evidence of cause."
    ),
    "interpretation": "A reading of the evidence that another analyst might dispute.",
    "scenario": "A conditional statement, not a forecast or a probability.",
    "unknown": "A question the available data cannot answer.",
}


@router.get("/{identifier}/historical", response_model=ReportOut, summary="Historical report")
def historical(
    identifier: str,
    start: dt.datetime = Query(),
    end: dt.datetime = Query(),
    adjustment_basis: str = Query(default=AdjustmentBasis.TOTAL_RETURN),
    benchmark: str | None = Query(default=None),
    include_narrative: bool = Query(
        default=False,
        description=(
            "Request optional LLM prose. The structured findings are complete "
            "without it, and any narrative containing a figure not present in "
            "the findings is discarded."
        ),
    ),
    db: Session = Depends(get_db),
) -> ReportOut:
    """An evidence-linked report on a past window.

    The information cutoff is the window's end, so nothing published later is
    used in the explanation.
    """
    cutoff = end if end.tzinfo else end.replace(tzinfo=dt.UTC)
    as_of = AsOf(cutoff=cutoff, purpose="historical_reconstruction")
    instrument = resolve_instrument(db, identifier, as_of=as_of)
    bundle = analyse(
        db, instrument, as_of=as_of, start=start, end=end,
        adjustment_basis=adjustment_basis, method_keys=PAST_METHODS,
        benchmark_ticker=benchmark, purpose="historical_reconstruction",
    )
    return _build(db, bundle, "historical", include_narrative)


@router.get("/{identifier}/current", response_model=ReportOut, summary="Current report")
def current(
    identifier: str,
    lookback_days: int = Query(default=365, ge=30, le=7300),
    adjustment_basis: str = Query(default=AdjustmentBasis.TOTAL_RETURN),
    benchmark: str | None = Query(default=None),
    include_narrative: bool = Query(default=False),
    as_of: AsOf = Depends(parse_as_of),
    db: Session = Depends(get_db),
) -> ReportOut:
    """An evidence-linked report on the current situation."""
    instrument = resolve_instrument(db, identifier, as_of=as_of)
    start = as_of.cutoff - dt.timedelta(days=lookback_days)
    bundle = analyse(
        db, instrument, as_of=as_of, start=start, end=as_of.cutoff,
        adjustment_basis=adjustment_basis, method_keys=PRESENT_METHODS,
        benchmark_ticker=benchmark, purpose="current_analysis",
    )
    return _build(db, bundle, "current", include_narrative)


def _build(db: Session, bundle, kind: str, include_narrative: bool) -> ReportOut:
    from sqlalchemy import select

    from app.db.models.provider import DataSource

    attribution = [
        {
            "key": s.key,
            "name": s.name,
            "attribution_text": s.attribution_text,
            "attribution_required": s.attribution_required,
            "official_api": s.official_api,
            "redistribution_allowed": s.redistribution_allowed,
            "caveats": s.caveats,
        }
        for s in db.execute(
            select(DataSource).where(DataSource.key.in_(bundle.data_sources))
        ).scalars().all()
    ]

    report = generate_report(bundle, report_kind=kind, attribution=attribution)

    narrative_status = "not_requested"
    if include_narrative:
        provider = get_narrative_provider()
        result = provider.generate(
            NarrativeRequest(
                title=report.title,
                summary=report.summary,
                findings=[f.as_dict() for f in report.findings],
                limitations=report.limitations,
            )
        )
        if result.ok:
            report.narrative = result.text
            report.narrative_model = result.model
            narrative_status = "generated"
        else:
            narrative_status = result.rejected_reason or "unavailable"

    payload = report.as_dict()
    payload["narrative_status"] = narrative_status
    payload["claim_kind_legend"] = CLAIM_KIND_LEGEND
    return ReportOut(**payload)
