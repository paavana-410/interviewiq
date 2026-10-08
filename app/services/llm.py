"""LLM service layer.

All providers are accessed through a single OpenAI-compatible client
(OpenAI SDK supports custom base_url for Anthropic, Ollama, Together, etc.).

Features:
  - Per-judge token-bucket rate limiting (RPM configurable per judge)
  - Exponential backoff with jitter on rate-limit / server errors
  - SHA-256 disk cache keyed by (model, prompt_version, input_hash)
  - API key values are NEVER stored in logs, repr, or error messages
  - MockLLMService for tests (no network, no keys)
"""
from __future__ import annotations

import abc
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)
import logging

from app.services.judge_config import JudgeConfig, _mask_key, mask_secrets_in_text

logger = logging.getLogger(__name__)


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
    cache_hit: bool = False


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
        prompt_version: str = "v1",
    ) -> LLMResponse:
        """Send a prompt and return the completion."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier safe for logging."""

    @property
    def provider(self) -> str:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Token-bucket rate limiter
# ─────────────────────────────────────────────────────────────────────────────

class _TokenBucket:
    """Thread-safe token bucket for RPM limiting."""

    def __init__(self, rpm: int):
        self._rpm = max(rpm, 1)
        self._tokens = float(self._rpm)
        self._max_tokens = float(self._rpm)
        self._refill_rate = self._rpm / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._max_tokens,
                    self._tokens + elapsed * self._refill_rate,
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._refill_rate
            time.sleep(wait)


# ─────────────────────────────────────────────────────────────────────────────
# Disk cache
# ─────────────────────────────────────────────────────────────────────────────

class _DiskCache:
    """SHA-256 keyed disk cache for LLM responses.

    Cache key = SHA-256(model + ":" + prompt_version + ":" + prompt_text)
    Stored as JSON files under cache_dir/<first2>/<hash>.json
    """

    def __init__(self, cache_dir: str | Path):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, prompt_version: str, prompt: str) -> str:
        raw = f"{model}:{prompt_version}:{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"

    def get(self, model: str, prompt_version: str, prompt: str) -> LLMResponse | None:
        key = self._key(model, prompt_version, prompt)
        p = self._path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return LLMResponse(
                text=data["text"],
                model_name=data["model_name"],
                input_tokens=data.get("input_tokens"),
                output_tokens=data.get("output_tokens"),
                latency_ms=data.get("latency_ms"),
                cache_hit=True,
            )
        except Exception:
            return None

    def put(self, model: str, prompt_version: str, prompt: str, resp: LLMResponse) -> None:
        key = self._key(model, prompt_version, prompt)
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "text": resp.text,
            "model_name": resp.model_name,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency_ms": resp.latency_ms,
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible service (covers OpenAI, Anthropic via proxy, Ollama, etc.)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAICompatibleService(LLMService):
    """Single implementation for any OpenAI-compatible API endpoint.

    Anthropic is accessed via its OpenAI-compatible endpoint.
    Local models (Ollama, LM Studio) work the same way.
    """

    def __init__(self, config: JudgeConfig, cache_dir: str | Path | None = None):
        import openai  # type: ignore

        self._config = config
        self._model = config.model
        self._provider = config.provider

        # Build client — key value goes directly to SDK, never logged
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or None,
        )

        self._bucket = _TokenBucket(config.rpm)
        self._cache = _DiskCache(cache_dir) if cache_dir else None
        self._secrets = [config.api_key]

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        prompt_version: str = "v1",
    ) -> LLMResponse:
        # Cache lookup (only for temperature=0 deterministic calls)
        if self._cache and temperature == 0.0:
            cached = self._cache.get(self._model, prompt_version, prompt)
            if cached is not None:
                logger.debug("Cache hit for model=%s prompt_version=%s", self._model, prompt_version)
                return cached

        resp = self._complete_with_retry(prompt, temperature=temperature, max_tokens=max_tokens)

        # Cache store
        if self._cache and temperature == 0.0:
            self._cache.put(self._model, prompt_version, prompt, resp)

        return resp

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(multiplier=2, max=60, jitter=4),
        reraise=True,
    )
    def _complete_with_retry(
        self, prompt: str, *, temperature: float, max_tokens: int
    ) -> LLMResponse:
        self._bucket.acquire()
        t0 = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            # Mask any key values that might appear in the error message
            safe_msg = mask_secrets_in_text(str(exc), self._secrets)
            logger.warning("LLM call failed (model=%s): %s", self._model, safe_msg)
            raise

        latency = int((time.monotonic() - t0) * 1000)
        choice = response.choices[0]
        text = choice.message.content or ""

        return LLMResponse(
            text=text,
            model_name=self._model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mock (tests / CI — no network, no keys)
# ─────────────────────────────────────────────────────────────────────────────

class MockLLMService(LLMService):
    """Deterministic mock. Returns valid rubric JSON for any prompt."""

    def __init__(self, fixed_response: str | None = None, name: str = "mock-llm"):
        self._fixed = fixed_response
        self._model = name

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "mock"

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        prompt_version: str = "v1",
    ) -> LLMResponse:
        if self._fixed is not None:
            return LLMResponse(text=self._fixed, model_name=self._model)

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
# Factory functions
# ─────────────────────────────────────────────────────────────────────────────

def _cache_dir() -> Path | None:
    try:
        from app.config import get_settings
        s = get_settings()
        p = Path(s.experiments_dir) / ".llm_cache"
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        return None


def get_llm_service(judge_name: str | None = None) -> LLMService:
    """Get an LLMService by judge name, or the first configured judge.

    Falls back to MockLLMService if no judges are configured (test mode).
    """
    from app.config import get_settings
    settings = get_settings()

    # Explicit mock mode
    if getattr(settings, "llm_provider", "mock") == "mock":
        return MockLLMService()

    from app.services.judge_config import get_judge_registry
    registry = get_judge_registry()

    if not registry.judges:
        return MockLLMService()

    if judge_name is not None:
        config = registry.get_judge(judge_name)
    else:
        config = registry.judges[0]

    return OpenAICompatibleService(config, cache_dir=_cache_dir())


def get_paraphrase_service() -> LLMService:
    """Get the paraphrase LLM service.

    Raises if not configured or if it matches a judge model.
    Falls back to mock in test mode.
    """
    from app.config import get_settings
    settings = get_settings()

    if getattr(settings, "llm_provider", "mock") == "mock":
        return MockLLMService(name="mock-paraphrase")

    from app.services.judge_config import get_judge_registry, JudgeConfig
    registry = get_judge_registry()

    if registry.paraphrase is None:
        raise RuntimeError(
            "PARAPHRASE_* not configured in .env. "
            "Set PARAPHRASE_PROVIDER, PARAPHRASE_API_KEY, PARAPHRASE_MODEL."
        )

    # Validation already ran at registry load time, but assert here for clarity
    registry.validate_paraphrase_differs_from_judges()

    # Build a JudgeConfig-compatible object for the service
    pcfg = registry.paraphrase
    judge_like = JudgeConfig(
        name=pcfg.name,
        provider=pcfg.provider,
        base_url=pcfg.base_url,
        api_key=pcfg.api_key,
        model=pcfg.model,
        rpm=pcfg.rpm,
    )
    return OpenAICompatibleService(judge_like, cache_dir=_cache_dir())


def get_all_judge_services() -> list[tuple[str, LLMService]]:
    """Return [(judge_name, service), ...] for all configured judges."""
    from app.config import get_settings
    settings = get_settings()

    if getattr(settings, "llm_provider", "mock") == "mock":
        return [("mock", MockLLMService())]

    from app.services.judge_config import get_judge_registry
    registry = get_judge_registry()

    if not registry.judges:
        return [("mock", MockLLMService())]

    cache = _cache_dir()
    return [
        (j.name, OpenAICompatibleService(j, cache_dir=cache))
        for j in registry.judges
    ]
