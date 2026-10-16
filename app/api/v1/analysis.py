"""Analysis endpoints: past, present, statistical, fundamental, risk, bubble."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import CommonGuards, get_db, parse_as_of
from app.api.schemas.common import AnalysisOut, ApiModel
from app.db.models.enums import AdjustmentBasis
from app.pit.availability import AsOf
from app.services.analysis_service import (
    FUNDAMENTAL_METHODS,
    PAST_METHODS,
    PRESENT_METHODS,
    RISK_METHODS,
    STATISTICAL_METHODS,
    analyse,
)
from app.services.data_access import resolve_instrument

router = APIRouter(prefix="/analysis", tags=["analysis"], dependencies=CommonGuards)


def _run(
    db: Session,
    identifier: str,
    as_of: AsOf,
    *,
    start: dt.datetime | None,
    end: dt.datetime | None,
    resolution: str,
    adjustment_basis: str,
    methods: tuple[str, ...],
    purpose: str,
    include_fundamentals: bool = True,
    include_benchmark: bool = True,
    benchmark: str | None = None,
    params_by_method: dict[str, dict] | None = None,
) -> AnalysisOut:
    instrument = resolve_instrument(db, identifier, as_of=as_of)
    bundle = analyse(
        db,
        instrument,
        as_of=as_of,
        start=start,
        end=end,
        resolution=resolution,
        adjustment_basis=adjustment_basis,
        method_keys=methods,
        include_fundamentals=include_fundamentals,
        include_benchmark=include_benchmark,
        benchmark_ticker=benchmark,
        params_by_method=params_by_method,
        purpose=purpose,
    )
    return AnalysisOut(**bundle.as_dict())


@router.get("/{identifier}/past", response_model=AnalysisOut, summary="Historical analysis")
def past(
    identifier: str,
    start: dt.datetime = Query(description="Start of the historical window"),
    end: dt.datetime = Query(description="End of the historical window"),
    resolution: str = Query(default="1d"),
    adjustment_basis: str = Query(default=AdjustmentBasis.TOTAL_RETURN),
    benchmark: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> AnalysisOut:
    """Reconstruct a past window using only information available at its end.

    The information cutoff is the window's end, so the analysis of a 2019
    period cannot see 2020 data. What happened *afterwards* is a separate
    request, which is what keeps the explanation free of hindsight.
    """
    cutoff = end if end.tzinfo else end.replace(tzinfo=dt.UTC)
    as_of = AsOf(cutoff=cutoff, purpose="historical_reconstruction")
    return _run(
        db, identifier, as_of,
        start=start, end=end, resolution=resolution,
        adjustment_basis=adjustment_basis, methods=PAST_METHODS,
        purpose="historical_reconstruction", benchmark=benchmark,
    )


@router.get("/{identifier}/present", response_model=AnalysisOut, summary="Current analysis")
def present(
    identifier: str,
    lookback_days: int = Query(default=365, ge=30, le=7300),
    resolution: str = Query(default="1d"),
    adjustment_basis: str = Query(default=AdjustmentBasis.TOTAL_RETURN),
    benchmark: str | None = Query(default=None),
    as_of: AsOf = Depends(parse_as_of),
    db: Session = Depends(get_db),
) -> AnalysisOut:
    """Current technical and trend analysis over a trailing window."""
    start = as_of.cutoff - dt.timedelta(days=lookback_days)
    return _run(
        db, identifier, as_of,
        start=start, end=as_of.cutoff, resolution=resolution,
        adjustment_basis=adjustment_basis, methods=PRESENT_METHODS,
        purpose="current_analysis", benchmark=benchmark,
    )


@router.get(
    "/{identifier}/statistical",
    response_model=AnalysisOut,
    summary="Statistical and econometric analysis",
)
def statistical(
    identifier: str,
    lookback_days: int = Query(default=1095, ge=180, le=7300),
    resolution: str = Query(default="1d"),
    benchmark: str | None = Query(default=None),
    as_of: AsOf = Depends(parse_as_of),
    db: Session = Depends(get_db),
) -> AnalysisOut:
    """Distribution, dependence, stationarity and regime-break diagnostics.

    Reported p-values are corrected for the number of tests performed where the
    method supports it; the uncorrected values are not reportable as evidence.
    """
    start = as_of.cutoff - dt.timedelta(days=lookback_days)
    return _run(
        db, identifier, as_of,
        start=start, end=as_of.cutoff, resolution=resolution,
        adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
        methods=STATISTICAL_METHODS, purpose="statistical_analysis",
        include_fundamentals=False, benchmark=benchmark,
    )


@router.get(
    "/{identifier}/fundamentals",
    response_model=AnalysisOut,
    summary="Fundamental analysis",
)
def fundamentals(
    identifier: str,
    as_of: AsOf = Depends(parse_as_of),
    db: Session = Depends(get_db),
) -> AnalysisOut:
    """Margins, balance sheet, cash flow and multiples as reported at the cutoff.

    Each figure is the newest value filed on or before the cutoff, so a later
    restatement is not used -- a 2022 analysis sees FY2021 as it was originally
    reported.
    """
    start = as_of.cutoff - dt.timedelta(days=400)
    return _run(
        db, identifier, as_of,
        start=start, end=as_of.cutoff, resolution="1d",
        adjustment_basis=AdjustmentBasis.RAW,
        methods=FUNDAMENTAL_METHODS, purpose="fundamental_analysis",
        include_benchmark=False,
    )


@router.get("/{identifier}/risk", response_model=AnalysisOut, summary="Risk analysis")
def risk(
    identifier: str,
    lookback_days: int = Query(default=730, ge=180, le=7300),
    target_annual_volatility: float = Query(default=0.15, gt=0, le=2.0),
    as_of: AsOf = Depends(parse_as_of),
    db: Session = Depends(get_db),
) -> AnalysisOut:
    """Volatility, tail risk, gap risk, sizing and scenarios.

    Educational research output, not personalised investment advice. Every
    method in this family carries that disclaimer in its response.
    """
    start = as_of.cutoff - dt.timedelta(days=lookback_days)
    return _run(
        db, identifier, as_of,
        start=start, end=as_of.cutoff, resolution="1d",
        adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
        methods=RISK_METHODS, purpose="risk_analysis",
        include_fundamentals=False, include_benchmark=False,
        params_by_method={
            "position_sizing": {"target_annual_volatility": target_annual_volatility}
        },
    )


class TradeStructureRequest(ApiModel):
    direction: str = "long"
    entry: float | None = None
    stop_atr_multiple: float = 2.0
    target_atr_multiples: list[float] = [2.0, 4.0, 6.0]
    account_value: float | None = None
    account_risk_fraction: float = 0.01
    lookback_days: int = 730


@router.post(
    "/{identifier}/trade-structure",
    response_model=AnalysisOut,
    summary="Trade structure arithmetic",
)
def trade_structure(
    identifier: str,
    request: TradeStructureRequest = Body(),
    as_of: AsOf = Depends(parse_as_of),
    db: Session = Depends(get_db),
) -> AnalysisOut:
    """The arithmetic of a hypothetical entry, stop and targets.

    Not a recommendation. Reports the stop distance in ATR terms, the
    risk/reward ratios (which carry no probability of being reached), and how
    often an overnight gap in this window would have jumped the stop.
    """
    start = as_of.cutoff - dt.timedelta(days=request.lookback_days)
    return _run(
        db, identifier, as_of,
        start=start, end=as_of.cutoff, resolution="1d",
        adjustment_basis=AdjustmentBasis.RAW,
        methods=("trade_structure", "risk_profile"),
        purpose="trade_structure",
        include_fundamentals=False, include_benchmark=False,
        params_by_method={
            "trade_structure": {
                "direction": request.direction,
                "entry": request.entry,
                "stop_atr_multiple": request.stop_atr_multiple,
                "target_atr_multiples": request.target_atr_multiples,
                "account_value": request.account_value,
                "account_risk_fraction": request.account_risk_fraction,
            }
        },
    )


@router.get(
    "/{identifier}/bubble",
    response_model=AnalysisOut,
    summary="Speculative-risk assessment",
)
def bubble(
    identifier: str,
    lookback_days: int = Query(default=1095, ge=365, le=7300),
    as_of: AsOf = Depends(parse_as_of),
    db: Session = Depends(get_db),
) -> AnalysisOut:
    """Multi-signal speculative-risk assessment.

    Ten signals are evaluated independently. Signals that cannot be evaluated
    report why and are excluded from the score's denominator, so a low score
    from three available signals is never presented as a low score from ten.
    No crash date or crash probability is produced.
    """
    start = as_of.cutoff - dt.timedelta(days=lookback_days)
    return _run(
        db, identifier, as_of,
        start=start, end=as_of.cutoff, resolution="1d",
        adjustment_basis=AdjustmentBasis.TOTAL_RETURN,
        methods=("bubble_assessment", "valuation_ratios", "growth_trajectory"),
        purpose="bubble_assessment",
        include_benchmark=False,
    )


class CustomAnalysisRequest(ApiModel):
    methods: list[str]
    start: dt.datetime | None = None
    end: dt.datetime | None = None
    resolution: str = "1d"
    adjustment_basis: str = AdjustmentBasis.TOTAL_RETURN
    benchmark: str | None = None
    params: dict[str, dict] = {}
    include_fundamentals: bool = True


@router.post("/{identifier}/custom", response_model=AnalysisOut, summary="Run chosen methods")
def custom(
    identifier: str,
    request: CustomAnalysisRequest = Body(),
    as_of: AsOf = Depends(parse_as_of),
    db: Session = Depends(get_db),
) -> AnalysisOut:
    """Run an explicit set of registered methods over a window."""
    return _run(
        db, identifier, as_of,
        start=request.start, end=request.end,
        resolution=request.resolution,
        adjustment_basis=request.adjustment_basis,
        methods=tuple(request.methods),
        purpose="custom_analysis",
        include_fundamentals=request.include_fundamentals,
        include_benchmark=request.benchmark is not None,
        benchmark=request.benchmark,
        params_by_method=request.params,
    )
