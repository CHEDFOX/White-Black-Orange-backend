"""LLM provider abstraction.

The contract, which is the entire point of this layer: **an LLM may only
rephrase findings this platform already computed.** It is never a source of a
figure, a citation, a date or a confidence level. Concretely:

* The prompt contains the findings as structured data and instructs the model
  to add no numbers of its own.
* The response is validated: any number appearing in the generated prose that
  does not appear in the supplied findings causes the narrative to be
  **discarded**, not corrected. A narrative is optional; a wrong figure is not
  acceptable.
* With ``LLM_PROVIDER=null`` no model is called at all and reports are complete
  without prose.

Swapping providers requires implementing ``NarrativeProvider`` only; nothing in
the analysis or reporting path changes.
"""

from __future__ import annotations

import abc
import decimal
import re
from dataclasses import dataclass

from app.core.logging import get_logger

log = get_logger(__name__)

# Matches numbers the way they appear in generated prose, including percentages,
# thousands separators and signs.
_NUMBER_PATTERN = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")

# Numbers too common to be meaningful evidence of fabrication.
_ALLOWED_TRIVIAL = {"0", "1", "2", "3", "4", "5", "10", "100", "0.0", "1.0"}


@dataclass
class NarrativeRequest:
    """What the model is given. Deliberately only findings, never raw data."""

    title: str
    summary: str
    findings: list[dict]
    limitations: list[str]
    max_words: int = 350


@dataclass
class NarrativeResult:
    text: str | None
    model: str | None
    rejected_reason: str | None = None
    fabricated_numbers: list[str] | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None


class NarrativeProvider(abc.ABC):
    """Generates optional explanatory prose from already-computed findings."""

    name: str

    @abc.abstractmethod
    def generate(self, request: NarrativeRequest) -> NarrativeResult:
        ...


def extract_numbers(text: str) -> set[str]:
    """Normalised numeric tokens in a block of text."""
    out: set[str] = set()
    for raw in _NUMBER_PATTERN.findall(text):
        token = raw.replace(",", "").lstrip("+")
        if token.startswith("-"):
            token = token[1:]
        token = token.rstrip(".")
        if not token or token in _ALLOWED_TRIVIAL:
            continue
        out.add(token)
        # Also record a 1-dp rounding, so "18.3%" matches a source "18.34".
        try:
            out.add(f"{float(token):.1f}")
            out.add(f"{float(token):.0f}")
        except ValueError:
            pass
    return out


def _renderings(value: float) -> list[str]:
    """Every string a careful writer might use for this number.

    Both rounding conventions are included. Python's format uses round-half-even,
    so a drawdown of -0.1125 renders as "11.2%", while a writer would usually
    write "11.3%". Permitting only one convention makes the guard reject correct
    prose over a value sitting exactly on a rounding boundary -- a false
    positive that throws away a good narrative.
    """
    out: list[str] = []
    for scaled in (value, value * 100):
        magnitude = abs(scaled)
        for places in (0, 1, 2):
            out.append(f"{magnitude:.{places}f}")
            # Round-half-up, the convention most writers use.
            out.append(
                str(
                    decimal.Decimal(repr(magnitude)).quantize(
                        decimal.Decimal(1).scaleb(-places),
                        rounding=decimal.ROUND_HALF_UP,
                    )
                )
            )
    return out


def permitted_numbers(request: NarrativeRequest) -> set[str]:
    """Every numeric token derivable from the supplied findings."""
    blob_parts = [request.title, request.summary, *request.limitations]
    for finding in request.findings:
        blob_parts.extend(
            str(finding.get(key, ""))
            for key in ("conclusion", "uncertainty", "methodology", "confidence_label")
        )
        blob_parts.extend(str(x) for x in finding.get("limitations", []))
        for calc in finding.get("calculations", []):
            value = calc.get("value")
            blob_parts.append(str(value))
            if isinstance(value, (int, float)):
                blob_parts.extend(_renderings(value))
        for ev in finding.get("evidence", []):
            blob_parts.append(str(ev.get("observed_value", "")))
            blob_parts.append(str(ev.get("description", "")))
    return extract_numbers(" ".join(blob_parts))


def validate_narrative(text: str, request: NarrativeRequest) -> NarrativeResult:
    """Reject a narrative containing a figure not present in the findings.

    Rejection rather than repair: a narrative is a convenience, and silently
    editing a model's prose to remove an invented number would leave the
    surrounding sentence asserting something the data does not support.
    """
    allowed = permitted_numbers(request)
    used = extract_numbers(text)
    fabricated = sorted(used - allowed)
    if fabricated:
        log.warning(
            "narrative_rejected_fabricated_numbers",
            fabricated=fabricated[:10],
            count=len(fabricated),
        )
        return NarrativeResult(
            text=None,
            model=None,
            rejected_reason=(
                "the generated narrative contained "
                f"{len(fabricated)} number(s) not present in the computed findings "
                f"({', '.join(fabricated[:5])}), so it was discarded rather than "
                "published"
            ),
            fabricated_numbers=fabricated,
        )
    return NarrativeResult(text=text, model=None)


class NullNarrativeProvider(NarrativeProvider):
    """The default. Generates nothing, so reports contain no model prose."""

    name = "null"

    def generate(self, request: NarrativeRequest) -> NarrativeResult:
        return NarrativeResult(
            text=None,
            model=None,
            rejected_reason=(
                "no LLM provider is configured (LLM_PROVIDER=null); the report's "
                "structured findings are complete without generated prose"
            ),
        )


_PROVIDER: NarrativeProvider | None = None


def get_narrative_provider() -> NarrativeProvider:
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER

    from app.core.config import settings

    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        from app.reporting.llm.anthropic_provider import AnthropicNarrativeProvider

        _PROVIDER = AnthropicNarrativeProvider()
    else:
        if settings.llm_provider != "null":
            log.warning(
                "llm_provider_unavailable",
                configured=settings.llm_provider,
                reason="no API key configured; falling back to the null provider",
            )
        _PROVIDER = NullNarrativeProvider()
    return _PROVIDER


def reset_narrative_provider() -> None:
    global _PROVIDER
    _PROVIDER = None
