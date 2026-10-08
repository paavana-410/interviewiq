"""LLM provider interface — swappable via config.

All LLM calls go through LLMService.complete().
Every call is logged to llm_call_logs for reproducibility.
"""
from __future__ import annotations

import abc
import json
import time
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    text: str
    model_name: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class LLMService(abc.ABC):
    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a prompt and return the completion."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Return a canonical model identifier for logging."""


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI implementation
# ─────────────────────────────────────────────────────────────────────────────

class OpenAILLMService(LLMService):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        import openai  # type: ignore
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 2048) -> LLMResponse:
        t0 = time.monotonic()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = int((time.monotonic() - t0) * 1000)
        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model_name=self._model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic implementation
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicLLMService(LLMService):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        import anthropic  # type: ignore
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 2048) -> LLMResponse:
        t0 = time.monotonic()
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = int((time.monotonic() - t0) * 1000)
        text = message.content[0].text if message.content else ""
        return LLMResponse(
            text=text,
            model_name=self._model,
            input_tokens=message.usage.input_tokens if message.usage else None,
            output_tokens=message.usage.output_tokens if message.usage else None,
            latency_ms=latency,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mock implementation (tests + no-key CI)
# ─────────────────────────────────────────────────────────────────────────────

class MockLLMService(LLMService):
    """Deterministic mock: returns a valid content evaluation JSON for any prompt."""

    def __init__(self, fixed_response: str | None = None):
        self._fixed = fixed_response
        self._model = "mock-llm"

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 2048) -> LLMResponse:
        if self._fixed is not None:
            return LLMResponse(text=self._fixed, model_name=self._model)
        # Generate a plausible rubric evaluation JSON
        response_json = {
            "criteria": [
                {
                    "criterion_name": "technical_correctness",
                    "score": 0.75,
                    "covered_concepts": ["object-oriented programming", "abstraction"],
                    "missing_concepts": ["design patterns"],
                    "evidence": "Candidate mentioned OOP and abstraction.",
                    "star_component": None,
                },
                {
                    "criterion_name": "relevance",
                    "score": 0.80,
                    "covered_concepts": ["role fit"],
                    "missing_concepts": [],
                    "evidence": "Answer is relevant to the question.",
                    "star_component": None,
                },
            ],
            "overall_score": 0.775,
            "star_detected": {
                "situation": True,
                "task": False,
                "action": True,
                "result": False,
            },
            "uncertainty": False,
            "uncertainty_reason": None,
        }
        return LLMResponse(
            text=json.dumps(response_json),
            model_name=self._model,
            input_tokens=len(prompt.split()),
            output_tokens=50,
            latency_ms=5,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_service(override_model: str | None = None) -> LLMService:
    from app.config import get_settings
    s = get_settings()

    if s.llm_provider == "mock":
        return MockLLMService()
    if s.llm_provider == "openai":
        model = override_model or s.openai_model
        return OpenAILLMService(api_key=s.openai_api_key, model=model)
    if s.llm_provider == "anthropic":
        model = override_model or s.anthropic_model
        return AnthropicLLMService(api_key=s.anthropic_api_key, model=model)

    raise ValueError(f"Unknown LLM provider: {s.llm_provider}")
