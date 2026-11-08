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
    finish_reason: str | None = None   # "stop" | "length" | "content_filter" | None
    max_tokens_used: int = 2048        # the max_tokens value that was sent


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
        run_index: int | None = None,
        bypass_cache: bool = False,
    ) -> LLMResponse:
        """Send a prompt and return the completion.

        run_index: when set, is mixed into the disk-cache key so that repeated
        calls with different indices are distinct real calls (not cache hits).
        Used by the repeated-run consistency test.

        bypass_cache: when True, skip cache lookup AND store — guarantees a real
        call. Used by qualify_judge.py / pre-flight to test the live judge.
        """

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

    def _key(self, model: str, prompt_version: str, prompt: str, run_index: int | None = None) -> str:
        raw = f"{model}:{prompt_version}:{prompt}"
        if run_index is not None:
            raw = f"{model}:{prompt_version}:run{run_index}:{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"

    def get(self, model: str, prompt_version: str, prompt: str, run_index: int | None = None) -> LLMResponse | None:
        key = self._key(model, prompt_version, prompt, run_index)
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
                finish_reason=data.get("finish_reason"),
                max_tokens_used=data.get("max_tokens_used", 2048),
                cache_hit=True,
            )
        except Exception:
            return None

    def put(self, model: str, prompt_version: str, prompt: str, resp: LLMResponse, run_index: int | None = None) -> None:
        key = self._key(model, prompt_version, prompt, run_index)
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "text": resp.text,
            "model_name": resp.model_name,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency_ms": resp.latency_ms,
            "finish_reason": resp.finish_reason,
            "max_tokens_used": resp.max_tokens_used,
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible service (covers OpenAI, Anthropic via proxy, Ollama, etc.)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAICompatibleService(LLMService):
    """Single implementation for any OpenAI-compatible API endpoint.

    Uses httpx directly instead of the OpenAI SDK to avoid hanging issues
    with certain providers. Supports all providers: OpenAI, Gemini (AQ. keys),
    OpenRouter, Ollama, Anthropic-compat, etc.
    """

    def __init__(self, config: JudgeConfig, cache_dir: str | Path | None = None):
        self._config = config
        self._model = config.model
        self._provider = config.provider
        self._base_url = (config.base_url or "").rstrip("/")
        self._api_key = config.api_key
        self._bucket = _TokenBucket(config.rpm)
        self._cache = _DiskCache(cache_dir) if cache_dir else None
        self._secrets = [config.api_key]

        # Gemini AQ. keys need x-goog-api-key + Authorization: Bearer
        self._is_gemini_aq = (
            config.api_key.startswith("AQ.")
            and "googleapis.com" in self._base_url
        )

    def _build_headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._is_gemini_aq:
            h["Authorization"] = f"Bearer {self._api_key}"
            h["x-goog-api-key"] = self._api_key
        else:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

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
        run_index: int | None = None,
        bypass_cache: bool = False,
    ) -> LLMResponse:
        if self._cache and temperature == 0.0 and not bypass_cache:
            cached = self._cache.get(self._model, prompt_version, prompt, run_index)
            if cached is not None:
                logger.debug("Cache hit model=%s run_index=%s", self._model, run_index)
                return cached

        resp = self._complete_with_retry(prompt, temperature=temperature, max_tokens=max_tokens)

        if self._cache and temperature == 0.0 and not bypass_cache:
            self._cache.put(self._model, prompt_version, prompt, resp, run_index)

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
        import httpx as _httpx

        self._bucket.acquire()
        url = self._base_url + "/chat/completions"
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t0 = time.monotonic()
        try:
            with _httpx.Client(timeout=30.0) as client:
                r = client.post(url, headers=self._build_headers(), json=body)
        except _httpx.TimeoutException:
            raise TimeoutError(f"Request to {self._model} timed out after 30s")
        except _httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to {self._base_url}: {e}")

        latency = int((time.monotonic() - t0) * 1000)

        # Non-retryable errors
        if r.status_code in (400, 401, 403, 404):
            try:
                data = r.json()
                err = data.get("error", {})
                if isinstance(err, list):
                    err = err[0].get("error", {}) if err else {}
                msg = err.get("message", r.text[:200])
            except Exception:
                msg = r.text[:200]
            safe = mask_secrets_in_text(msg, self._secrets)
            exc_map = {
                400: ValueError(f"Bad request (400): {safe}"),
                401: PermissionError(f"Auth failed (401): {safe}"),
                403: PermissionError(f"Forbidden (403): {safe}"),
                404: LookupError(f"Model not found (404): {self._model}"),
            }
            raise exc_map[r.status_code]

        # Retryable errors
        if r.status_code in (429, 502, 503, 504):
            try:
                msg = r.json().get("error", {}).get("message", "")
            except Exception:
                msg = ""
            safe = mask_secrets_in_text(msg, self._secrets)
            logger.warning("Retryable error %s model=%s: %s", r.status_code, self._model, safe[:100])
            raise IOError(f"HTTP {r.status_code}: {safe[:100]}")

        if r.status_code != 200:
            raise IOError(f"Unexpected HTTP {r.status_code}: {r.text[:100]}")

        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise ValueError(f"Unexpected response shape: {str(data)[:200]}")

        usage = data.get("usage", {})
        finish_reason = None
        try:
            finish_reason = data["choices"][0].get("finish_reason")
        except (KeyError, IndexError):
            pass

        return LLMResponse(
            text=text,
            model_name=self._model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency,
            finish_reason=finish_reason,
            max_tokens_used=max_tokens,
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
        run_index: int | None = None,
        bypass_cache: bool = False,
    ) -> LLMResponse:
        if self._fixed is not None:
            return LLMResponse(
                text=self._fixed, model_name=self._model, finish_reason="stop",
                max_tokens_used=max_tokens,
            )

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
            finish_reason="stop",
            max_tokens_used=max_tokens,
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


# ─────────────────────────────────────────────────────────────────────────────
# Ollama service — local model, small context, JSON-repair fallback
# ─────────────────────────────────────────────────────────────────────────────

def _repair_json(text: str) -> str:
    """Best-effort JSON repair for malformed LLM output.

    Handles the most common failure modes from small local models:
    - Trailing commas
    - Missing closing braces/brackets
    - Text before/after the JSON object
    """
    import re as _re
    # Strip markdown fences
    text = _re.sub(r"```(?:json)?", "", text).strip()
    # Find first { ... } or [ ... ]
    for start_ch, end_ch in (('{', '}'), ('[', ']')):
        start = text.find(start_ch)
        if start == -1:
            continue
        # Walk backwards to find the last matching close
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start=start):
            if ch == start_ch:
                depth += 1
            elif ch == end_ch:
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            candidate = text[start:end + 1]
            # Remove trailing commas before } or ]
            candidate = _re.sub(r',\s*([}\]])', r'\1', candidate)
            return candidate
    return text


class OllamaService(LLMService):
    """Local Ollama judge. Uses httpx directly, small context window.

    Loaded only during the local_judge stage. Never co-loaded with Whisper.
    Config comes from .env: OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_BASE_URL.
    """

    def __init__(self, cache_dir: Path | None = None):
        from app.config import get_settings
        s = get_settings()
        self._base_url = s.ollama_base_url.rstrip("/")
        self._model = s.ollama_model
        self._num_ctx = min(max(s.ollama_num_ctx, 512), 4096)  # clamp 512–4096
        self._cache = _DiskCache(cache_dir) if cache_dir else None
        self._bucket = _TokenBucket(60)  # local: 60 rpm ceiling

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"

    @property
    def provider(self) -> str:
        return "ollama"

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        prompt_version: str = "v1",
        run_index: int | None = None,
        bypass_cache: bool = False,
    ) -> LLMResponse:
        if self._cache and temperature == 0.0 and not bypass_cache:
            cached = self._cache.get(self.model_name, prompt_version, prompt, run_index)
            if cached is not None:
                return cached

        resp = self._complete_with_retry(
            prompt, temperature=temperature, max_tokens=max_tokens
        )

        if self._cache and temperature == 0.0 and not bypass_cache:
            self._cache.put(self.model_name, prompt_version, prompt, resp, run_index)

        return resp

    @retry(
        retry=retry_if_exception_type((IOError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(multiplier=2, max=30, jitter=2),
        reraise=True,
    )
    def _complete_with_retry(
        self, prompt: str, *, temperature: float, max_tokens: int
    ) -> LLMResponse:
        import httpx as _httpx

        self._bucket.acquire()
        url = self._base_url + "/api/chat"
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self._num_ctx,
            },
        }

        t0 = time.monotonic()
        try:
            with _httpx.Client(timeout=120.0) as client:
                r = client.post(url, json=body)
        except _httpx.TimeoutException:
            raise TimeoutError(f"Ollama timed out after 120s (model={self._model})")
        except _httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Is `ollama serve` running?"
            )

        latency = int((time.monotonic() - t0) * 1000)

        if r.status_code == 404:
            raise LookupError(
                f"Ollama model '{self._model}' not found. "
                f"Run: ollama pull {self._model}"
            )
        if r.status_code != 200:
            raise IOError(f"Ollama HTTP {r.status_code}: {r.text[:150]}")

        data = r.json()
        text = data.get("message", {}).get("content", "") or ""

        # Attempt JSON repair if the response looks like broken JSON
        if text.strip().startswith(("{", "[")):
            try:
                json.loads(text)
            except json.JSONDecodeError:
                text = _repair_json(text)

        # Ollama signals truncation via done_reason
        finish_reason = data.get("done_reason")  # "stop" | "length" | None

        return LLMResponse(
            text=text,
            model_name=self.model_name,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_ms=latency,
            finish_reason=finish_reason,
            max_tokens_used=max_tokens,
        )


def get_ollama_service() -> OllamaService:
    """Return an OllamaService using settings from .env."""
    return OllamaService(cache_dir=_cache_dir())
