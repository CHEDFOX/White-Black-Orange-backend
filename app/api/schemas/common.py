"""Shared response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ErrorResponse(ApiModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class Attribution(ApiModel):
    key: str
    name: str
    attribution_required: bool
    attribution_text: str | None = None
    official_api: bool
    redistribution_allowed: bool
    classification: str
    caveats: str | None = None


class DataFreshness(ApiModel):
    latest_bar: str | None = None
    age_days: int | None = None
    is_stale: bool
    note: str


class CoverageEntry(ApiModel):
    data_kind: str
    resolution: str | None = None
    source: str
    first_observation: str | None = None
    last_observation: str | None = None
    observation_count: int
    last_refreshed_at: str | None = None
    known_gap_count: int
    notes: str | None = None


class MethodOutcomeOut(ApiModel):
    """A method's result, or the explained reason it could not run."""

    key: str
    version: str
    display_name: str
    ok: bool
    values: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    observations_used: int = 0
    input_fingerprint: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    unavailable_reason: str | None = None
    unavailable_code: str | None = None


class AnalysisOut(ApiModel):
    instrument_id: str
    symbol: str
    name: str
    as_of: str
    purpose: str
    resolution: str
    adjustment_basis: str
    period_start: str | None = None
    period_end: str | None = None
    observations: int
    currency: str
    methods: dict[str, MethodOutcomeOut]
    methods_available: list[str]
    methods_unavailable: dict[str, str | None]
    data_sources: list[str]
    gaps: list[dict[str, Any]]
    quality_issues: list[dict[str, Any]]
    coverage_notes: list[str]
    benchmark_symbol: str | None = None
    benchmark_note: str | None = None
    fundamentals_note: str | None = None
    # The statements a client must surface alongside any figure it displays.
    disclosures: list[str]


class PaginatedMeta(ApiModel):
    limit: int
    offset: int
    returned: int
    has_more: bool
