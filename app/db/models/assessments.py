"""Risk assessments, bubble assessments, reports and job records."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.models.enums import ClaimKind, DataClassification, EvidenceStance, JobStatus
from app.db.types import JSONType


class BubbleAssessment(Base):
    """Multi-signal speculative-risk assessment.

    No single indicator produces a verdict: ``signals`` holds each evaluated
    warning signal with its own status and evidence, and the composite score is
    a documented aggregation of those, with ``opposing_evidence`` kept
    alongside.
    """

    __tablename__ = "bubble_assessments"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "as_of", "method_version",
            name="uq_bubble_assessments_instrument_id_as_of_method_version",
        ),
        Index("ix_bubble_assessments_lookup", "instrument_id", "as_of"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    as_of: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    method_version: Mapped[str] = mapped_column(String(20), nullable=False)
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # 0..100 vulnerability estimate. Deliberately named "vulnerability", not
    # "crash probability": the platform does not predict crash dates.
    vulnerability_score: Mapped[float | None] = mapped_column(Float)
    vulnerability_band: Mapped[str | None] = mapped_column(String(20))
    signals: Mapped[list] = mapped_column(JSONType, default=list)
    signals_triggered: Mapped[int] = mapped_column(Integer, default=0)
    signals_evaluated: Mapped[int] = mapped_column(Integer, default=0)
    signals_unavailable: Mapped[list] = mapped_column(JSONType, default=list)
    supporting_evidence: Mapped[list] = mapped_column(JSONType, default=list)
    opposing_evidence: Mapped[list] = mapped_column(JSONType, default=list)
    historical_analogues: Mapped[list] = mapped_column(JSONType, default=list)
    uncertainties: Mapped[list] = mapped_column(JSONType, default=list)
    strengthening_conditions: Mapped[list] = mapped_column(JSONType, default=list)
    weakening_conditions: Mapped[list] = mapped_column(JSONType, default=list)
    classification: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DataClassification.PRODUCTION
    )


class RiskAssessment(Base):
    """Volatility-scaled risk framing for an instrument at a point in time.

    This is educational research output, not personalised investment advice;
    the distinction is recorded in ``disclaimer_code`` and surfaced by the API.
    """

    __tablename__ = "risk_assessments"
    __table_args__ = (
        Index("ix_risk_assessments_lookup", "instrument_id", "as_of"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    as_of: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    method_version: Mapped[str] = mapped_column(String(20), nullable=False)
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    realized_volatility_annual: Mapped[float | None] = mapped_column(Float)
    atr: Mapped[float | None] = mapped_column(Float)
    var_95: Mapped[float | None] = mapped_column(Float)
    var_99: Mapped[float | None] = mapped_column(Float)
    expected_shortfall_95: Mapped[float | None] = mapped_column(Float)
    max_historical_drawdown: Mapped[float | None] = mapped_column(Float)
    gap_risk: Mapped[dict] = mapped_column(JSONType, default=dict)
    liquidity: Mapped[dict] = mapped_column(JSONType, default=dict)
    levels: Mapped[dict] = mapped_column(JSONType, default=dict)
    scenarios: Mapped[list] = mapped_column(JSONType, default=list)
    position_sizing: Mapped[dict] = mapped_column(JSONType, default=dict)
    limitations: Mapped[list] = mapped_column(JSONType, default=list)
    disclaimer_code: Mapped[str] = mapped_column(
        String(40), nullable=False, default="educational_research"
    )
    classification: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DataClassification.PRODUCTION
    )


class GeneratedReport(Base):
    """A generated analysis report, reproducible from its recorded inputs."""

    __tablename__ = "generated_reports"
    __table_args__ = (
        Index("ix_generated_reports_lookup", "instrument_id", "report_kind", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    # historical | current | future | bubble | risk
    report_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    information_cutoff: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_start: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    # The narrative is optional and LLM-generated; the structured findings below
    # are the authoritative content and exist without any LLM.
    narrative: Mapped[str | None] = mapped_column(Text)
    narrative_model: Mapped[str | None] = mapped_column(String(80))

    generator_version: Mapped[str] = mapped_column(String(20), nullable=False)
    reproduction_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    inputs: Mapped[dict] = mapped_column(JSONType, default=dict)
    method_versions: Mapped[dict] = mapped_column(JSONType, default=dict)
    data_coverage: Mapped[dict] = mapped_column(JSONType, default=dict)
    limitations: Mapped[list] = mapped_column(JSONType, default=list)
    classification: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DataClassification.PRODUCTION
    )

    findings: Mapped[list["ReportFinding"]] = relationship(
        back_populates="report", cascade="all, delete-orphan", order_by="ReportFinding.ordinal"
    )


class ReportFinding(Base):
    """One labelled conclusion inside a report."""

    __tablename__ = "report_findings"
    __table_args__ = (Index("ix_report_findings_report", "report_id", "ordinal"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("generated_reports.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str] = mapped_column(String(60), nullable=False)
    conclusion: Mapped[str] = mapped_column(Text, nullable=False)
    claim_kind: Mapped[str] = mapped_column(String(32), nullable=False, default=ClaimKind.UNKNOWN)
    confidence_label: Mapped[str | None] = mapped_column(String(24))
    uncertainty: Mapped[str | None] = mapped_column(Text)
    methodology: Mapped[str | None] = mapped_column(Text)
    limitations: Mapped[list] = mapped_column(JSONType, default=list)
    calculations: Mapped[list] = mapped_column(JSONType, default=list)

    report: Mapped[GeneratedReport] = relationship(back_populates="findings")
    evidence: Mapped[list["ReportEvidence"]] = relationship(
        back_populates="finding", cascade="all, delete-orphan"
    )


class ReportEvidence(Base):
    """A citation attached to a finding, pointing at a stored row."""

    __tablename__ = "report_evidence"
    __table_args__ = (Index("ix_report_evidence_finding", "finding_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    finding_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("report_findings.id", ondelete="CASCADE"), nullable=False
    )
    stance: Mapped[str] = mapped_column(
        String(16), nullable=False, default=EvidenceStance.SUPPORTING
    )
    # The table and row this evidence resolves to, so a reader can re-query it.
    source_table: Mapped[str] = mapped_column(String(60), nullable=False)
    source_row_id: Mapped[str | None] = mapped_column(String(64))
    data_source_key: Mapped[str | None] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(String(1000))
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    observed_value: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[dict] = mapped_column(JSONType, default=dict)

    finding: Mapped[ReportFinding] = relationship(back_populates="evidence")


class AnalysisJob(Base):
    """Background-job record for long-running analysis requested via the API."""

    __tablename__ = "analysis_jobs"
    __table_args__ = (
        Index("ix_analysis_jobs_status", "status", "created_at"),
        UniqueConstraint("idempotency_key", name="uq_analysis_jobs_idempotency_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    job_type: Mapped[str] = mapped_column(String(60), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(300), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=JobStatus.PENDING)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str | None] = mapped_column(String(80))
    params: Mapped[dict] = mapped_column(JSONType, default=dict)
    result_ref: Mapped[str | None] = mapped_column(String(100))
    error: Mapped[str | None] = mapped_column(Text)
