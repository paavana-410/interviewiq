"""Content Engine — LLM-based rubric judge.

INPUT:  transcript TEXT only (no timing, no delivery features)
OUTPUT: ContentResult

STRUCTURAL ISOLATION GUARANTEE:
  - analyze() accepts ONLY (transcript_text, rubric_criteria).
  - This module does NOT import DeliveryEngine or DeliveryResult.

Truncation handling:
  - Every call records finish_reason and completion_tokens from the provider.
  - finish_reason == "length" → TruncatedOutputError is raised.
  - The caller retries ONCE with max_tokens doubled. Unchanged requests are
    never retried — only after the token budget increases.
  - A failed call stays None in the result; it is never guessed.

Prompt v2 changes (vs v1):
  - Evidence ≤ 15 words.
  - At most 3 covered_concepts and 3 missing_concepts per criterion.
  - No extra prose fields.
  - Instruction: "Return ONLY the JSON object, no other text."
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.core import LLMCallLog
from app.services.llm import LLMResponse, LLMService


# ─────────────────────────────────────────────────────────────────────────────
# Rubric input schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RubricCriterion:
    criterion_name: str
    criterion_type: str
    weight: float = 1.0
    expected_concepts: list[str] = field(default_factory=list)
    star_component: bool = False
    scoring_guide: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Typed exception for truncated output
# ─────────────────────────────────────────────────────────────────────────────

class TruncatedOutputError(Exception):
    """Raised when finish_reason == 'length'. Carries how many tokens were used."""
    def __init__(self, tokens_used: int, max_tokens: int):
        self.tokens_used = tokens_used
        self.max_tokens = max_tokens
        super().__init__(
            f"Output truncated: used {tokens_used} of {max_tokens} max_tokens "
            f"(finish_reason='length'). Will retry with doubled budget."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Call statistics (per ContentEngine instance)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CallStats:
    total: int = 0
    truncated: int = 0   # finish_reason == "length"
    retried: int = 0     # calls that needed a retry (any reason)
    failed: int = 0      # calls that exhausted retries and returned None

    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def record_truncation(self) -> None:
        with self._lock:
            self.truncated += 1

    def record_retry(self) -> None:
        with self._lock:
            self.retried += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failed += 1

    def record_call(self) -> None:
        with self._lock:
            self.total += 1

    def summary(self) -> dict:
        return {
            "total_calls": self.total,
            "truncated": self.truncated,
            "retried": self.retried,
            "failed": self.failed,
        }

    def print_summary(self) -> None:
        print(
            f"\n  Call stats: total={self.total}  truncated={self.truncated}  "
            f"retried={self.retried}  failed={self.failed}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic output schema
# ─────────────────────────────────────────────────────────────────────────────

class CriterionResult(BaseModel):
    criterion_name: str = "general"
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    # v2: at most 3 items each
    covered_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    # v2: ≤ 15 words
    evidence: str = ""
    star_component: str | None = None

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        if v is None:
            return 0.0
        try:
            return max(0.0, min(1.0, float(v)))
        except (ValueError, TypeError):
            return 0.0

    @field_validator("covered_concepts", "missing_concepts", mode="before")
    @classmethod
    def cap_concepts(cls, v: Any) -> list:
        if isinstance(v, list):
            return v[:3]
        return []

    @field_validator("evidence", mode="before")
    @classmethod
    def cap_evidence(cls, v: Any) -> str:
        if not isinstance(v, str):
            return ""
        words = v.split()
        return " ".join(words[:15])


class STARDetection(BaseModel):
    situation: bool = False
    task: bool = False
    action: bool = False
    result: bool = False


class ContentEvalOutput(BaseModel):
    criteria: list[CriterionResult] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    star_detected: STARDetection = Field(default_factory=STARDetection)
    uncertainty: bool = False
    uncertainty_reason: str | None = None

    @field_validator("star_detected", mode="before")
    @classmethod
    def coerce_star_detected(cls, v: Any) -> Any:
        """LLMs sometimes return `false` or `null` instead of the dict."""
        if isinstance(v, dict):
            return v
        # Any non-dict (bool, None, str) → default empty STARDetection
        return {"situation": False, "task": False, "action": False, "result": False}

    @model_validator(mode="before")
    @classmethod
    def compute_default_overall(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "overall_score" not in data or data["overall_score"] is None:
                criteria = data.get("criteria", [])
                if isinstance(criteria, list) and criteria:
                    scores = []
                    for c in criteria:
                        if isinstance(c, dict) and "score" in c and c["score"] is not None:
                            try:
                                scores.append(float(c["score"]))
                            except (ValueError, TypeError):
                                pass
                    data["overall_score"] = (sum(scores) / len(scores)) if scores else 0.0
                else:
                    data["overall_score"] = 0.0
            else:
                try:
                    data["overall_score"] = max(0.0, min(1.0, float(data["overall_score"])))
                except (ValueError, TypeError):
                    data["overall_score"] = 0.0
        return data

    @field_validator("overall_score", mode="before")
    @classmethod
    def clamp_overall(cls, v: Any) -> float:
        if v is None:
            return 0.0
        try:
            return max(0.0, min(1.0, float(v)))
        except (ValueError, TypeError):
            return 0.0


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
    prompt_version: str = "v2"
    llm_call_log_id: int | None = None

    def score_by_criterion(self, name: str) -> float | None:
        return {
            "correctness": self.correctness_score,
            "coverage": self.coverage_score,
            "relevance": self.relevance_score,
            "completeness": self.completeness_score,
            "star": self.star_score,
        }.get(name)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt v2
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_VERSION = "v2"

# v2 is deliberately compact:
# - "Return ONLY the JSON object, no other text." at the top
# - evidence capped at 15 words in the schema comment
# - at most 3 concepts per list
# - no scoring_guide repeated back, no extra prose fields
_RUBRIC_JUDGE_PROMPT_V2 = """\
Return ONLY the JSON object below, no other text.

You are an interview content evaluator. Score ONLY the semantic content \
(what was said). Ignore speaking pace, pauses, fillers, grammar, or delivery.

Question: {question_text}

Transcript: {transcript_text}

Rubric: {rubric_json}
{delivery_section}
JSON schema (fill every field):
{{"criteria":[{{"criterion_name":"<name>","score":<0.0-1.0>,\
"covered_concepts":["<concept>",...MAX 3],"missing_concepts":["<concept>",...MAX 3],\
"evidence":"<≤15 words verbatim from transcript>",\
"star_component":"<Situation|Task|Action|Result|null>"}},...],\
"overall_score":<weighted average 0.0-1.0>,\
"star_detected":{{"situation":<bool>,"task":<bool>,"action":<bool>,"result":<bool>}},\
"uncertainty":<bool>,"uncertainty_reason":"<string or null>"}}"""

_DELIVERY_SECTION_ABLATION = (
    "\nDelivery features (ablation — research only): {delivery_json}\n"
)


def _build_prompt(
    question_text: str,
    transcript_text: str,
    rubric_criteria: list[RubricCriterion],
    delivery_features: dict | None = None,
) -> str:
    # Keep rubric compact — only name, type, weight, concepts (no scoring_guide)
    rubric_json = json.dumps(
        [
            {
                "criterion_name": c.criterion_name,
                "criterion_type": c.criterion_type,
                "weight": c.weight,
                "expected_concepts": c.expected_concepts[:5],  # cap at 5
            }
            for c in rubric_criteria
        ],
        separators=(",", ":"),  # no whitespace → smaller prompt
    )

    delivery_section = ""
    if delivery_features:
        delivery_section = _DELIVERY_SECTION_ABLATION.format(
            delivery_json=json.dumps(delivery_features, separators=(",", ":"))
        )

    return _RUBRIC_JUDGE_PROMPT_V2.format(
        question_text=question_text,
        transcript_text=transcript_text,
        rubric_json=rubric_json,
        delivery_section=delivery_section,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

# Default starting max_tokens for content eval calls
_DEFAULT_MAX_TOKENS = 4096
# Maximum we'll ever double up to before giving up
_MAX_TOKENS_CEILING = 8192


class ContentEngine:
    """LLM rubric judge. Receives transcript text only."""

    def __init__(self, llm: LLMService, db: Session | None = None):
        self._llm = llm
        self._db = db
        self._settings = get_settings()
        # Shared stats across all calls in this engine instance
        self.stats = CallStats()

    def analyze(
        self,
        transcript_text: str,
        question_text: str,
        rubric_criteria: list[RubricCriterion],
        *,
        delivery_features: dict | None = None,
        run_index: int | None = None,
    ) -> ContentResult:
        settings = get_settings()
        effective_delivery = (
            delivery_features if settings.content_engine_include_delivery else None
        )

        prompt = _build_prompt(
            question_text=question_text,
            transcript_text=transcript_text,
            rubric_criteria=rubric_criteria,
            delivery_features=effective_delivery,
        )

        pv = settings.content_engine_prompt_version
        llm_response, parsed, log_id = self._call_with_length_retry(
            prompt, prompt_version=pv, run_index=run_index
        )

        return self._build_result(
            parsed,
            used_delivery=(effective_delivery is not None),
            log_id=log_id,
            prompt_version=pv,
        )

    def _call_with_length_retry(
        self,
        prompt: str,
        prompt_version: str,
        *,
        run_index: int | None = None,
    ) -> tuple[LLMResponse, ContentEvalOutput, int | None]:
        """Call LLM. On finish_reason==length, retry ONCE with doubled budget."""
        max_tokens = _DEFAULT_MAX_TOKENS
        last_exc: Exception | None = None

        for attempt in range(2):  # attempt 0: normal, attempt 1: doubled budget
            if attempt == 1:
                max_tokens = min(max_tokens * 2, _MAX_TOKENS_CEILING)
                self.stats.record_retry()

            self.stats.record_call()
            try:
                result = self._call_with_parse_retry(
                    prompt, prompt_version=prompt_version,
                    max_tokens=max_tokens, run_index=run_index,
                    bypass_cache=(attempt > 0),
                )
                return result
            except TruncatedOutputError as exc:
                self.stats.record_truncation()
                last_exc = exc
                # loop around with doubled budget
                continue
            except Exception as exc:
                last_exc = exc
                break  # non-truncation failure — don't retry here

        # Exhausted attempts
        self.stats.record_failure()
        raise last_exc  # type: ignore[misc]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _call_with_parse_retry(
        self,
        prompt: str,
        *,
        prompt_version: str,
        max_tokens: int,
        run_index: int | None = None,
        bypass_cache: bool = False,
    ) -> tuple[LLMResponse, ContentEvalOutput, int | None]:
        """Call LLM and parse JSON. Retries up to 3x on parse failures.

        Raises TruncatedOutputError immediately (no parse retry useful there).
        """
        llm_resp = self._llm.complete(
            prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
            run_index=run_index,
            bypass_cache=bypass_cache,
        )
        raw = llm_resp.text

        try:
            json_str = self._extract_json(raw)
            parsed_data = json.loads(json_str)
            validated = ContentEvalOutput.model_validate(parsed_data)
        except Exception:
            # Only treat as fatal truncation if we cannot extract valid JSON
            if llm_resp.finish_reason == "length":
                raise TruncatedOutputError(
                    tokens_used=llm_resp.output_tokens or max_tokens,
                    max_tokens=max_tokens,
                )
            raise

        log_id = self._persist_call(
            prompt=prompt,
            raw_output=raw,
            parsed_ok=True,
            llm_resp=llm_resp,
            prompt_version=prompt_version,
        )
        return llm_resp, validated, log_id

    def _extract_json(self, text: str) -> str:
        """Extract outermost JSON object using brace-depth counting.

        Handles: markdown fences, nested objects, truncated output,
        trailing commas.
        """
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

        start = -1
        open_ch = close_ch = ""
        for i, ch in enumerate(text):
            if ch == "{":
                start, open_ch, close_ch = i, "{", "}"
                break
            if ch == "[":
                start, open_ch, close_ch = i, "[", "]"
                break

        if start == -1:
            raise ValueError(
                f"No JSON object in LLM output: {text[:200]}"
            )

        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start=start):
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    end = i
                    break

        candidate = text[start:end + 1] if end != -1 else text[start:]
        # Remove trailing commas
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return candidate

    def _persist_call(
        self,
        prompt: str,
        raw_output: str,
        parsed_ok: bool,
        llm_resp: LLMResponse,
        prompt_version: str,
    ) -> int | None:
        if self._db is None:
            return None
        log = LLMCallLog(
            model_name=self._llm.model_name,
            prompt_version=prompt_version,
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
        prompt_version: str,
    ) -> ContentResult:
        scores: dict[str, float] = {
            cr.criterion_name.lower(): cr.score
            for cr in parsed.criteria
        }

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
            prompt_version=prompt_version,
            llm_call_log_id=log_id,
        )
