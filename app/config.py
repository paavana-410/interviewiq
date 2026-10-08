"""Central configuration loaded from .env / environment variables."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql://interviewiq:interviewiq@localhost:5432/interviewiq"

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic", "mock"] = "mock"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # ── Prompt versions ───────────────────────────────────────────────────────
    content_engine_prompt_version: str = "v1"
    holistic_engine_prompt_version: str = "v1"

    # ── STT ───────────────────────────────────────────────────────────────────
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # ── Delivery Engine ───────────────────────────────────────────────────────
    long_pause_threshold_sec: float = 1.5
    filler_words: str = "um,uh,like,you know,so,basically,actually,literally"

    @property
    def filler_words_list(self) -> list[str]:
        return [w.strip().lower() for w in self.filler_words.split(",")]

    # ── Audit thresholds ──────────────────────────────────────────────────────
    audit_delivery_invariance_max: float = 0.08
    audit_content_sensitivity_min: float = 0.15
    audit_paraphrase_max_deviation: float = 0.10
    audit_repeatability_max_variance: float = 0.02
    audit_stt_max_drift: float = 0.08
    audit_repeated_runs: int = 5

    # ── Ablation ──────────────────────────────────────────────────────────────
    content_engine_include_delivery: bool = False

    # ── Experiment tracking ───────────────────────────────────────────────────
    experiments_dir: str = "experiments"

    # ── Misc ──────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    audio_upload_dir: str = "uploads"
    max_audio_size_mb: int = 50

    @field_validator("audio_upload_dir", "experiments_dir", mode="after")
    @classmethod
    def ensure_dirs(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
