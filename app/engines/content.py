"""Content Engine — LLM-based rubric judge.

INPUT:  transcript TEXT only (no timing, no delivery features)
OUTPUT: ContentResult

STRUCTURAL ISOLATION GUARANTEE:
  - The public entry point analyze() accepts ONLY (transcript_text, rubric_criteria).
  - If ablation mode is enabled, delivery_features dict MAY be passed but is
    clearly marked and tested separately.
  - This module does NOT import DeliveryEngine or DeliveryResult.

Every LLM call is persisted to llm_call_logs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.core import LLMCallLog
from app.services.llm import LLMService


# ─────────────────────────────────────────────────────────────────────────────
# Rubric input schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RubricCriterion:
    criterion_name: str
    criterion_type: str          # correctness | relevance | completeness | concept_coverage | star_structure
    weight: float = 1.0
    expected_concepts: list[str] = field(default_factory=list)
    star_component: bool = False
    scoring_guide: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic output schema — validates LLM JSON strictly
# ─────────────────────────────────────────────────────────────────────────────

class CriterionResult(BaseModel):
    criterion_name: str
    score: float = Field(ge=0.0, le=1.0)
    covered_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    # Must be a verbatim quote or phrase from the transcript
    evidence: str = ""
    # Only populated for STAR criteria
    star_component: str | None = None

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))


class STARDetection(BaseModel):
    situation: bool = False
    task: bool = False
    action: bool = False
    result: bool = False


class ContentEvalOutput(BaseModel):
    """Full structured output from the LLM content judge."""
    criteria: list[CriterionResult]
    overall_score: float = Field(ge=0.0, le=1.0)
    star_detected: STARDetection = Field(default_factory=STARDetection)
    uncertainty: bool = False
    uncertainty_reason: str | None = None

    @field_validator("overall_score", mode="before")
    @classmethod
    def clamp_overall(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))


# ─────────────────────────────────────────────────────────────────────────────
# Result returned to callers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContentResult:
    correctness_score: float | None
    coverage_score: float | None
    relevance_score: float | None
    completeness_score: float | None
    star_score: float | None
    overall_score: float
    evidence_json: dict
    used_delivery_features: bool
    uncertainty: bool
    llm_call_log_id: int | None = None

    def score_by_criterion(self, name: str) -> float | None:
        """Convenience: look up any criterion by name."""
        return {
            "correctness": self.correctness_score,
            "coverage": self.coverage_score,
            "relevance": self.relevance_score,
            "completeness": self.completeness_score,
            "star": self.star_score,
        }.get(name)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_VERSION = "v1"

_RUBRIC_JUDGE_PROMPT = """\
You are a placement interview evaluator. Your job is to assess the CONTENT of a candidate's spoken answer.

## IMPORTANT RULES
1. Evaluate ONLY what was said (semantic content). Ignore speaking pace, pauses, fillers, or grammar.
2. Base your scores strictly on the rubric criteria provided.
3. For each criterion, quote a SHORT piece of evidence directly from the transcript.
4. Return ONLY valid JSON matching the schema below. No prose outside the JSON.

## Question
{question_text}

## Transcript (candidate's answer)
\"\"\"
{transcript_text}
\"\"\"

## Rubric Criteria
{rubric_json}

{delivery_section}

## Output JSON Schema
{{
  "criteria": [
    {{
      "criterion_name": "<name>",
      "score": <float 0.0–1.0>,
      "covered_concepts": ["<concept>", ...],
      "missing_concepts": ["<concept>", ...],
      "evidence": "<verbatim phrase from transcript>",
      "star_component": "<Situation|Task|Action|Result|null>"
    }}
  ],
  "overall_score": <weighted average 0.0–1.0>,
  "star_detected": {{
    "situation": <bool>,
    "task": <bool>,
    "action": <bool>,
    "result": <bool>
  }},
  "uncertainty": <bool>,
  "uncertainty_reason": "<string or null>"
}}
"""

_DELIVERY_SECTION_ABLATION = """\
## Delivery Features (ablation mode — for research comparison only)
{delivery_json}
NOTE: You MAY use these features to adjust scoring. This is the ABLATION condition.
"""


def _build_prompt(
    question_text: str,
    transcript_text: str,
    rubric_criteria: list[RubricCriterion],
    delivery_features: dict | None = None,
) -> str:
    rubric_json = json.dumps(
        [
            {
                "criterion_name": c.criterion_name,
                "criterion_type": c.criterion_type,
                "weight": c.weight,
                "expected_concepts": c.expected_concepts,
                "star_component": c.star_component,
                "scoring_guide": c.scoring_guide,
            }
            for c in rubric_criteria
        ],
        indent=2,
    )

    delivery_section = ""
    if delivery_features:
        delivery_section = _DELIVERY_SECTION_ABLATION.format(
            delivery_json=json.dumps(delivery_features, indent=2)
        )

    return _RUBRIC_JUDGE_PROMPT.format(
        question_text=question_text,
        transcript_text=transcript_text,
        rubric_json=rubric_json,
        delivery_section=delivery_section,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class ContentEngine:
    """LLM rubric judge. Receives transcript text only (no timing data).

    Structural isolation: this class must NEVER receive DeliveryResult
    or WordTimestamp objects unless ablation mode is explicitly enabled.
    """

    def __init__(self, llm: LLMService, db: Session | None = None):
        self._llm = llm
        self._db = db
        self._settings = get_settings()

    def analyze(
        self,
        transcript_text: str,
        question_text: str,
        rubric_criteria: list[RubricCriterion],
        *,
        delivery_features: dict | None = None,
    ) -> ContentResult:
        """Evaluate the content of a transcript against rubric criteria.

        Args:
            transcript_text: The plain text transcript. NO timing data.
            question_text:   The original interview question.
            rubric_criteria: List of rubric criteria for this question.
            delivery_features: Optional dict of delivery metrics (ablation only).
                               Only used when settings.content_engine_include_delivery=True.
        """
        settings = get_settings()
        # Enforce isolation: only pass delivery if ablation is on
        effective_delivery = delivery_features if settings.content_engine_include_delivery else None

        prompt = _build_prompt(
            question_text=question_text,
            transcript_text=transcript_text,
            rubric_criteria=rubric_criteria,
            delivery_features=effective_delivery,
        )

        llm_response, parsed, log_id = self._call_with_retry(prompt)

        return self._build_result(parsed, used_delivery=(effective_delivery is not None), log_id=log_id)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _call_with_retry(self, prompt: str) -> tuple[Any, ContentEvalOutput, int | None]:
        """Call LLM, validate JSON, retry on parse failure."""
        llm_resp = self._llm.complete(prompt, temperature=0.0, max_tokens=2048)
        raw = llm_resp.text

        # Extract JSON block (handle markdown code fences)
        json_str = self._extract_json(raw)

        parsed_data = json.loads(json_str)
        validated = ContentEvalOutput.model_validate(parsed_data)

        log_id = self._persist_call(
            prompt=prompt,
            raw_output=raw,
            parsed_ok=True,
            llm_resp=llm_resp,
        )
        return llm_resp, validated, log_id

    def _extract_json(self, text: str) -> str:
        """Strip markdown fences and extract the JSON object."""
        # Try ```json ... ``` or ``` ... ```
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            return fence.group(1)
        # Fallback: find first { ... }
        brace = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace:
            return brace.group(1)
        raise ValueError(f"No JSON object found in LLM output: {text[:200]}")

    def _persist_call(
        self,
        prompt: str,
        raw_output: str,
        parsed_ok: bool,
        llm_resp: Any,
    ) -> int | None:
        if self._db is None:
            return None
        log = LLMCallLog(
            model_name=self._llm.model_name,
            prompt_version=PROMPT_VERSION,
            call_type="content_eval",
            prompt_text=prompt,
            raw_output=raw_output,
            parsed_ok=parsed_ok,
            temperature=0.0,
            input_tokens=llm_resp.input_tokens,
            output_tokens=llm_resp.output_tokens,
            latency_ms=llm_resp.latency_ms,
        )
        self._db.add(log)
        self._db.flush()
        return log.log_id

    def _build_result(
        self,
        parsed: ContentEvalOutput,
        used_delivery: bool,
        log_id: int | None,
    ) -> ContentResult:
        # Map criterion names to standard slots
        scores: dict[str, float] = {}
        for cr in parsed.criteria:
            scores[cr.criterion_name.lower()] = cr.score

        def _find(*keys: str) -> float | None:
            for k in keys:
                for name, score in scores.items():
                    if k in name:
                        return score
            return None

        evidence_json = {
            "criteria": [c.model_dump() for c in parsed.criteria],
            "star_detected": parsed.star_detected.model_dump(),
            "uncertainty_reason": parsed.uncertainty_reason,
        }

        return ContentResult(
            correctness_score=_find("correct"),
            coverage_score=_find("coverage", "concept"),
            relevance_score=_find("relev"),
            completeness_score=_find("complet"),
            star_score=_find("star"),
            overall_score=parsed.overall_score,
            evidence_json=evidence_json,
            used_delivery_features=used_delivery,
            uncertainty=parsed.uncertainty,
            llm_call_log_id=log_id,
        )
