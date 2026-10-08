"""Baseline Evaluator A — Single holistic LLM score (no rubric).

Used for ablation comparison: rubric-based content engine vs. this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.core import LLMCallLog
from app.services.llm import LLMService


HOLISTIC_PROMPT_VERSION = "v1"

_HOLISTIC_PROMPT = """\
You are a placement interview evaluator.

Rate the following answer to the interview question on a scale from 0.0 to 1.0.
Consider technical correctness, relevance, completeness, and communication clarity.
Return ONLY a JSON object: {{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}

Question: {question_text}

Answer:
\"\"\"
{transcript_text}
\"\"\"
"""


@dataclass
class HolisticResult:
    score: float
    reasoning: str
    model_name: str
    log_id: int | None = None


class HolisticEvaluator:
    """Single opaque LLM score — baseline for comparison."""

    def __init__(self, llm: LLMService, db: Session | None = None):
        self._llm = llm
        self._db = db

    def evaluate(self, transcript_text: str, question_text: str) -> HolisticResult:
        import json
        prompt = _HOLISTIC_PROMPT.format(
            question_text=question_text,
            transcript_text=transcript_text,
        )
        resp = self._llm.complete(prompt, temperature=0.0)
        raw = resp.text

        # Parse
        try:
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            score = float(data.get("score", 0.5))
            reasoning = data.get("reasoning", "")
        except Exception:
            score = 0.5
            reasoning = raw[:100]

        score = max(0.0, min(1.0, score))
        log_id = self._persist(prompt, raw, resp)

        return HolisticResult(
            score=score,
            reasoning=reasoning,
            model_name=self._llm.model_name,
            log_id=log_id,
        )

    def _persist(self, prompt: str, raw: str, resp) -> int | None:
        if self._db is None:
            return None
        log = LLMCallLog(
            model_name=self._llm.model_name,
            prompt_version=HOLISTIC_PROMPT_VERSION,
            call_type="holistic",
            prompt_text=prompt,
            raw_output=raw,
            parsed_ok=True,
            temperature=0.0,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_ms=resp.latency_ms,
        )
        self._db.add(log)
        self._db.flush()
        return log.log_id
