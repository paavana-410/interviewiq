"""Judge qualification — the single standard for "is this judge usable".

A judge qualifies if a real rubric call returns:
  1. a non-empty reply,
  2. parseable JSON,
  3. finish_reason == "stop" (i.e. the model finished naturally, not truncated).

Used by:
  - qualify_judge.py  (3 calls on 3 fixed answers → full qualification)
  - run_audit_benchmark.py pre-flight (1 call per judge → same standard)

qualify_call() bypasses the disk cache so the call is real, not a cached hit.
API keys are never included in QualifyResult.error (masked by the caller).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.services.llm import LLMResponse, LLMService


QUALIFY_PROMPT_VERSION = "qualify-v1"

# A minimal but real rubric prompt so JSON parsing is a meaningful check.
_QUALIFY_PROMPT = """\
Return ONLY a JSON object, no other text.

You are an interview answer evaluator. Score the answer 0.0-1.0.

Question: {question}

Answer: {answer}

JSON: {{"overall_score": <0.0-1.0>, "one_strength": "<short>"}}"""


@dataclass
class QualifyResult:
    ok: bool
    finish_reason: str | None
    completion_tokens: int | None
    max_tokens_sent: int
    latency_ms: int | None
    error: str = ""   # safe — no key values; caller masks any provider text

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "finish_reason": self.finish_reason,
            "completion_tokens": self.completion_tokens,
            "max_tokens_sent": self.max_tokens_sent,
            "latency_ms": self.latency_ms,
            "error": self.error or None,
        }


def _extract_json_object(text: str) -> str:
    """Pull the outermost {...} out of the reply using brace-depth counting."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    if start == -1:
        return text
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    return text[start:end + 1] if end != -1 else text[start:]


def qualify_call(
    llm_service: LLMService,
    question: str,
    answer: str,
    *,
    max_tokens: int = 512,
) -> QualifyResult:
    """Make ONE real rubric call and qualify it. Bypasses the disk cache.

    Returns a QualifyResult. Never raises — failures become ok=False with error.
    """
    prompt = _QUALIFY_PROMPT.format(question=question, answer=answer)

    try:
        resp: LLMResponse = llm_service.complete(
            prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            prompt_version=QUALIFY_PROMPT_VERSION,
            bypass_cache=True,
        )
    except Exception as exc:
        # Caller is responsible for masking secrets from provider error text.
        return QualifyResult(
            ok=False, finish_reason=None, completion_tokens=None,
            max_tokens_sent=max_tokens, latency_ms=None,
            error=f"call failed: {type(exc).__name__}",
        )

    # 1. non-empty
    if not resp.text or not resp.text.strip():
        return QualifyResult(
            ok=False, finish_reason=resp.finish_reason,
            completion_tokens=resp.output_tokens,
            max_tokens_sent=resp.max_tokens_used, latency_ms=resp.latency_ms,
            error="empty reply",
        )

    # 2. valid JSON
    try:
        json.loads(_extract_json_object(resp.text))
    except Exception:
        return QualifyResult(
            ok=False, finish_reason=resp.finish_reason,
            completion_tokens=resp.output_tokens,
            max_tokens_sent=resp.max_tokens_used, latency_ms=resp.latency_ms,
            error="reply is not valid JSON",
        )

    # 3. finish_reason == "stop"
    if resp.finish_reason != "stop":
        return QualifyResult(
            ok=False, finish_reason=resp.finish_reason,
            completion_tokens=resp.output_tokens,
            max_tokens_sent=resp.max_tokens_used, latency_ms=resp.latency_ms,
            error=f"finish_reason={resp.finish_reason!r} (expected 'stop')",
        )

    return QualifyResult(
        ok=True, finish_reason=resp.finish_reason,
        completion_tokens=resp.output_tokens,
        max_tokens_sent=resp.max_tokens_used, latency_ms=resp.latency_ms,
    )
