"""Findings and evidence: the structures that make a report checkable.

A ``Finding`` is one labelled conclusion. The label is a ``ClaimKind``, and it
is mandatory, so no sentence in a report can present a model estimate as an
observed fact. Every finding carries:

* the calculations behind it, each naming the method and version that produced
  it, so the number can be recomputed;
* supporting *and* contradicting evidence, because a finding with only
  supporting evidence is an argument;
* its own uncertainty and limitations, inherited from the methods used.

``EvidenceItem`` always points at something real -- a stored row, a method
output, a source URL with a publication date. There is no free-text evidence
field an LLM could fill with a plausible-sounding citation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field

from app.db.models.enums import ClaimKind, EvidenceStance

GENERATOR_VERSION = "1.0.0"


@dataclass
class Calculation:
    """One computed number with its full provenance."""

    label: str
    value: float | int | str | None
    unit: str
    method_key: str
    method_version: str
    observations_used: int
    input_fingerprint: str | None = None
    params: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "method_key": self.method_key,
            "method_version": self.method_version,
            "observations_used": self.observations_used,
            "input_fingerprint": self.input_fingerprint,
            "params": self.params,
        }


@dataclass
class EvidenceItem:
    """A citation resolving to a stored row, a method output or a source URL."""

    stance: str
    description: str
    source_table: str
    source_row_id: str | None = None
    data_source_key: str | None = None
    url: str | None = None
    published_at: dt.datetime | None = None
    observed_value: str | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "stance": self.stance,
            "description": self.description,
            "source_table": self.source_table,
            "source_row_id": self.source_row_id,
            "data_source_key": self.data_source_key,
            "url": self.url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "observed_value": self.observed_value,
            "detail": self.detail,
        }


@dataclass
class Finding:
    """One labelled conclusion with its evidence."""

    section: str
    conclusion: str
    claim_kind: str
    confidence_label: str | None = None
    uncertainty: str | None = None
    methodology: str | None = None
    limitations: list[str] = field(default_factory=list)
    calculations: list[Calculation] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.claim_kind not in set(ClaimKind):
            raise ValueError(
                f"claim_kind must be one of {sorted(ClaimKind)}, got "
                f"{self.claim_kind!r}. An unlabelled conclusion would let a model "
                "estimate be read as an observed fact."
            )

    @property
    def supporting(self) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.stance == EvidenceStance.SUPPORTING]

    @property
    def contradicting(self) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.stance == EvidenceStance.CONTRADICTING]

    def as_dict(self) -> dict:
        return {
            "section": self.section,
            "conclusion": self.conclusion,
            "claim_kind": self.claim_kind,
            "confidence_label": self.confidence_label,
            "uncertainty": self.uncertainty,
            "methodology": self.methodology,
            "limitations": self.limitations,
            "calculations": [c.as_dict() for c in self.calculations],
            "evidence": [e.as_dict() for e in self.evidence],
            "supporting_count": len(self.supporting),
            "contradicting_count": len(self.contradicting),
        }


@dataclass
class Report:
    """A complete report, reproducible from its recorded inputs."""

    instrument_id: str
    symbol: str
    name: str
    report_kind: str
    title: str
    summary: str
    created_at: dt.datetime
    information_cutoff: dt.datetime
    period_start: dt.datetime | None = None
    period_end: dt.datetime | None = None
    findings: list[Finding] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    data_coverage: dict = field(default_factory=dict)
    method_versions: dict[str, str] = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    attribution: list[dict] = field(default_factory=list)
    narrative: str | None = None
    narrative_model: str | None = None
    generator_version: str = GENERATOR_VERSION

    @property
    def reproduction_key(self) -> str:
        """Hash of everything needed to reproduce this report.

        Two reports with the same key used the same instrument, cutoff, window,
        method versions and input data. This is what makes "reproduce key
        calculations from recorded inputs" a checkable claim.
        """
        payload = {
            "instrument_id": self.instrument_id,
            "report_kind": self.report_kind,
            "information_cutoff": self.information_cutoff.isoformat(),
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "generator_version": self.generator_version,
            "method_versions": dict(sorted(self.method_versions.items())),
            "inputs": self.inputs,
            "fingerprints": sorted(
                c.input_fingerprint
                for f in self.findings
                for c in f.calculations
                if c.input_fingerprint
            ),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def claim_kind_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.claim_kind] = counts.get(finding.claim_kind, 0) + 1
        return counts

    def as_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "symbol": self.symbol,
            "name": self.name,
            "report_kind": self.report_kind,
            "title": self.title,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "information_cutoff": self.information_cutoff.isoformat(),
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "findings": [f.as_dict() for f in self.findings],
            "limitations": self.limitations,
            "data_coverage": self.data_coverage,
            "method_versions": self.method_versions,
            "inputs": self.inputs,
            "attribution": self.attribution,
            "narrative": self.narrative,
            "narrative_model": self.narrative_model,
            "generator_version": self.generator_version,
            "reproduction_key": self.reproduction_key,
            "claim_kind_counts": self.claim_kind_counts(),
        }


def calculation_from_outcome(
    outcome, label: str, field_name: str, unit: str
) -> Calculation | None:
    """Build a ``Calculation`` from a method outcome's named field."""
    if not outcome.ok:
        return None
    value = outcome.values.get(field_name)
    if value is None:
        return None
    return Calculation(
        label=label,
        value=value,
        unit=unit,
        method_key=outcome.key,
        method_version=outcome.version,
        observations_used=outcome.observations_used,
        input_fingerprint=outcome.input_fingerprint,
        params=outcome.params,
    )


def method_evidence(outcome, description: str, *, stance: str = EvidenceStance.SUPPORTING,
                    value_field: str | None = None) -> EvidenceItem:
    """Evidence pointing at a method output rather than a stored row."""
    observed = None
    if value_field and outcome.ok:
        raw = outcome.values.get(value_field)
        observed = None if raw is None else str(raw)
    return EvidenceItem(
        stance=stance,
        description=description,
        source_table="computed_features",
        source_row_id=outcome.input_fingerprint,
        observed_value=observed,
        detail={
            "method_key": outcome.key,
            "method_version": outcome.version,
            "observations_used": outcome.observations_used,
        },
    )
