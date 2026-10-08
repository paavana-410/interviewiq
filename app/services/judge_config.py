"""Judge configuration — reads JUDGE_N_* and PARAPHRASE_* from environment.

Variable naming convention in .env:
    JUDGE_1_NAME=judge_gpt4o
    JUDGE_1_PROVIDER=openai
    JUDGE_1_BASE_URL=https://api.openai.com/v1
    JUDGE_1_API_KEY=sk-...
    JUDGE_1_MODEL=gpt-4o
    JUDGE_1_RPM=500          # requests per minute (optional, default 60)

    JUDGE_2_NAME=judge_claude
    JUDGE_2_PROVIDER=anthropic
    JUDGE_2_BASE_URL=https://api.anthropic.com/v1
    JUDGE_2_API_KEY=sk-ant-...
    JUDGE_2_MODEL=claude-3-5-sonnet-20241022
    JUDGE_2_RPM=60

    JUDGE_3_NAME=judge_local
    JUDGE_3_PROVIDER=openai_compatible
    JUDGE_3_BASE_URL=http://localhost:11434/v1
    JUDGE_3_API_KEY=ollama
    JUDGE_3_MODEL=llama3.2:3b
    JUDGE_3_RPM=120

    PARAPHRASE_NAME=paraphrase_gpt4o
    PARAPHRASE_PROVIDER=openai
    PARAPHRASE_BASE_URL=https://api.openai.com/v1
    PARAPHRASE_API_KEY=sk-...
    PARAPHRASE_MODEL=gpt-4o

Key design rules:
  - API key values are NEVER stored in repr, __str__, or logs.
  - `safe_info()` returns a dict with the key replaced by a masked string.
  - Paraphrase model must differ from every judge model (enforced at load time).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator


# ─────────────────────────────────────────────────────────────────────────────
# Key masking helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mask_key(key: str) -> str:
    """Return a masked version showing only first 4 and last 4 chars."""
    if not key or len(key) < 10:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def mask_secrets_in_text(text: str, secrets: list[str]) -> str:
    """Replace any occurrence of a secret value in text with [REDACTED]."""
    result = text
    for secret in secrets:
        if secret and len(secret) >= 8:
            result = result.replace(secret, "[REDACTED]")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# JudgeConfig dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeConfig:
    name: str           # logical name used in CLI and config.json
    provider: str       # openai | anthropic | openai_compatible | mock
    base_url: str       # full API base URL
    api_key: str        # secret — never logged
    model: str          # exact model ID sent to the API
    rpm: int = 60       # requests per minute for rate limiting

    # ── Safe representation (no key value) ───────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"JudgeConfig(name={self.name!r}, provider={self.provider!r}, "
            f"model={self.model!r}, base_url={self.base_url!r}, "
            f"api_key={_mask_key(self.api_key)!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def safe_info(self) -> dict:
        """Dict safe for logging and config.json — no key value."""
        return {
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_masked": _mask_key(self.api_key),
            "rpm": self.rpm,
        }


@dataclass
class ParaphraseConfig:
    name: str
    provider: str
    base_url: str
    api_key: str
    model: str
    rpm: int = 60

    def __repr__(self) -> str:
        return (
            f"ParaphraseConfig(name={self.name!r}, provider={self.provider!r}, "
            f"model={self.model!r}, api_key={_mask_key(self.api_key)!r})"
        )

    def safe_info(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_masked": _mask_key(self.api_key),
            "rpm": self.rpm,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registry — loaded once from environment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeRegistry:
    judges: list[JudgeConfig] = field(default_factory=list)
    paraphrase: ParaphraseConfig | None = None

    def get_judge(self, name: str) -> JudgeConfig:
        for j in self.judges:
            if j.name == name:
                return j
        available = [j.name for j in self.judges]
        raise KeyError(f"Judge {name!r} not found. Available: {available}")

    def judge_names(self) -> list[str]:
        return [j.name for j in self.judges]

    def all_secrets(self) -> list[str]:
        """Return all secret values for masking in logs."""
        secrets = [j.api_key for j in self.judges]
        if self.paraphrase:
            secrets.append(self.paraphrase.api_key)
        return [s for s in secrets if s]

    def validate_paraphrase_differs_from_judges(self) -> None:
        """Raise if paraphrase model matches any judge model (conflict of interest)."""
        if self.paraphrase is None:
            return
        for j in self.judges:
            if self.paraphrase.model == j.model:
                raise ValueError(
                    f"Paraphrase model '{self.paraphrase.model}' is the same as "
                    f"judge '{j.name}' model. Paraphrase must use a different model "
                    f"to avoid circular self-evaluation."
                )


def _read_judge_block(prefix: str) -> JudgeConfig | None:
    """Read JUDGE_N_* variables for a given numeric prefix (e.g. 'JUDGE_1')."""
    name = os.environ.get(f"{prefix}_NAME", "").strip()
    provider = os.environ.get(f"{prefix}_PROVIDER", "").strip()
    base_url = os.environ.get(f"{prefix}_BASE_URL", "").strip()
    api_key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    model = os.environ.get(f"{prefix}_MODEL", "").strip()
    rpm_str = os.environ.get(f"{prefix}_RPM", "60").strip()

    # A block is valid only if name, provider, api_key, and model are all set
    if not all([name, provider, api_key, model]):
        return None

    # Apply default base_urls if not explicitly set
    if not base_url:
        if provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif provider == "anthropic":
            base_url = "https://api.anthropic.com/v1"
        else:
            base_url = ""

    try:
        rpm = int(rpm_str)
    except ValueError:
        rpm = 60

    return JudgeConfig(
        name=name,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        rpm=rpm,
    )


def _read_paraphrase_block() -> ParaphraseConfig | None:
    prefix = "PARAPHRASE"
    name = os.environ.get(f"{prefix}_NAME", "paraphrase").strip()
    provider = os.environ.get(f"{prefix}_PROVIDER", "").strip()
    base_url = os.environ.get(f"{prefix}_BASE_URL", "").strip()
    api_key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    model = os.environ.get(f"{prefix}_MODEL", "").strip()
    rpm_str = os.environ.get(f"{prefix}_RPM", "60").strip()

    if not all([provider, api_key, model]):
        return None

    if not base_url:
        if provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif provider == "anthropic":
            base_url = "https://api.anthropic.com/v1"

    try:
        rpm = int(rpm_str)
    except ValueError:
        rpm = 60

    return ParaphraseConfig(
        name=name,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        rpm=rpm,
    )


def load_judge_registry(env_file: str = ".env") -> JudgeRegistry:
    """Load all JUDGE_N_* and PARAPHRASE_* blocks from environment.

    Call this after dotenv has already loaded the .env file.
    Supports up to 10 judges (JUDGE_1 through JUDGE_10).
    """
    judges: list[JudgeConfig] = []
    for n in range(1, 11):
        cfg = _read_judge_block(f"JUDGE_{n}")
        if cfg is not None:
            judges.append(cfg)

    paraphrase = _read_paraphrase_block()

    registry = JudgeRegistry(judges=judges, paraphrase=paraphrase)

    # Validate paraphrase ≠ any judge
    if paraphrase is not None:
        registry.validate_paraphrase_differs_from_judges()

    return registry


# ─────────────────────────────────────────────────────────────────────────────
# Cached singleton
# ─────────────────────────────────────────────────────────────────────────────

_registry: JudgeRegistry | None = None


def get_judge_registry(reload: bool = False) -> JudgeRegistry:
    """Return the cached JudgeRegistry, loading from env if needed."""
    global _registry
    if _registry is None or reload:
        # Ensure .env is loaded
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(override=False)
        except ImportError:
            pass
        _registry = load_judge_registry()
    return _registry
