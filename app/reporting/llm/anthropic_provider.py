"""Anthropic narrative provider.

The system prompt forbids introducing figures, and the response is validated
against the supplied findings regardless of what the prompt said -- a prompt is
a request, not a guarantee.
"""

from __future__ import annotations

import json

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.reporting.llm.base import (
    NarrativeProvider,
    NarrativeRequest,
    NarrativeResult,
    validate_narrative,
)

log = get_logger(__name__)

SYSTEM_PROMPT = """\
You are writing an explanatory paragraph for a financial analysis report.

You will be given findings that have ALREADY been computed from verified market
data, each labelled with its epistemic status.

Absolute rules:
1. Do not introduce any number that is not present in the findings you are
   given. No prices, percentages, dates, ratios or counts of your own.
2. Preserve each finding's label. A finding marked model_estimate must not be
   written as though it were an observed fact. A finding marked
   causal_hypothesis must not be written as an established cause.
3. Do not add explanations, causes or context that are not in the findings. If
   the findings do not say why something happened, do not say why.
4. Do not give advice, make recommendations, or predict future prices.
5. Where findings include contradicting evidence, mention it.
6. Write plain prose. No headings, no bullet lists, no markdown.

Your job is to make the computed findings readable, nothing more."""


class AnthropicNarrativeProvider(NarrativeProvider):
    name = "anthropic"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self.model = settings.anthropic_model
        self._client = client or httpx.Client(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": settings.anthropic_api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=60.0,
        )

    def generate(self, request: NarrativeRequest) -> NarrativeResult:
        payload = {
            "model": self.model,
            "max_tokens": settings.llm_max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Write at most {request.max_words} words explaining these "
                        "findings.\n\n"
                        + json.dumps(
                            {
                                "title": request.title,
                                "summary": request.summary,
                                "findings": request.findings,
                                "limitations": request.limitations,
                            },
                            indent=2,
                            default=str,
                        )
                    ),
                }
            ],
        }
        try:
            response = self._client.post("/v1/messages", json=payload)
        except httpx.HTTPError as exc:
            log.warning("anthropic_request_failed", error=str(exc))
            return NarrativeResult(
                text=None, model=self.model,
                rejected_reason=f"the narrative request failed: {exc}",
            )

        if response.status_code >= 400:
            log.warning(
                "anthropic_error", status=response.status_code, body=response.text[:300]
            )
            return NarrativeResult(
                text=None, model=self.model,
                rejected_reason=(
                    f"the narrative provider returned status {response.status_code}"
                ),
            )

        body = response.json()
        blocks = body.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        if not text:
            return NarrativeResult(
                text=None, model=self.model,
                rejected_reason="the narrative provider returned no text",
            )

        # Validate regardless of the prompt. The prompt is a request; this is
        # the guarantee.
        validated = validate_narrative(text, request)
        if not validated.ok:
            return NarrativeResult(
                text=None,
                model=self.model,
                rejected_reason=validated.rejected_reason,
                fabricated_numbers=validated.fabricated_numbers,
            )
        return NarrativeResult(text=text, model=self.model)
