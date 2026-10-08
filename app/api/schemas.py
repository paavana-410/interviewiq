"""Pydantic request/response schemas for API endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Sessions
# ─────────────────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    user_id: int | None = None
    question_id: int | None = None
    mode: str = "practice"


class SessionResponse(BaseModel):
    session_id: int
    user_id: int | None
    question_id: int | None
    mode: str
    status: str
    started_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Responses (audio upload)
# ─────────────────────────────────────────────────────────────────────────────

class ResponseCreated(BaseModel):
    response_id: int
    session_id: int
    audio_path: str | None
    status: str = "processing"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation result
# ─────────────────────────────────────────────────────────────────────────────

class DeliveryMetricsSchema(BaseModel):
    speaking_rate_wpm: float | None
    pause_count: int | None
    long_pause_count: int | None
    filler_count: int | None
    filler_density: float | None
    repetition_count: int | None
    duration_sec: int | None
    detail: dict[str, Any] = Field(default_factory=dict)


class ContentEvalSchema(BaseModel):
    overall_score: float | None
    correctness_score: float | None
    coverage_score: float | None
    relevance_score: float | None
    completeness_score: float | None
    star_score: float | None
    evidence: dict[str, Any] = Field(default_factory=dict)
    used_delivery_features: bool = False
    uncertainty: bool = False


class AuditResultSchema(BaseModel):
    delivery_invariance_error: float | None
    content_sensitivity: float | None
    paraphrase_deviation: float | None
    repeatability_variance: float | None
    stt_sensitivity: float | None
    uncertainty_flag: bool
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationResponse(BaseModel):
    response_id: int
    session_id: int
    transcript: str
    delivery: DeliveryMetricsSchema | None
    content: ContentEvalSchema | None
    audit: AuditResultSchema | None
    uncertainty_flag: bool
