"""All ORM models for InterviewIQ."""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class QuestionCategory(str, enum.Enum):
    HR = "hr"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"


class SessionMode(str, enum.Enum):
    PRACTICE = "practice"
    AUDIT = "audit"
    BENCHMARK = "benchmark"


class SessionStatus(str, enum.Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"


class CriterionType(str, enum.Enum):
    CORRECTNESS = "correctness"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CONCEPT_COVERAGE = "concept_coverage"
    STAR_STRUCTURE = "star_structure"


# ─────────────────────────────────────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="student")
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    sessions: Mapped[list["InterviewSession"]] = relationship(back_populates="user")
    annotations: Mapped[list["HumanAnnotation"]] = relationship(back_populates="annotator")


# ─────────────────────────────────────────────────────────────────────────────
# Questions + Rubrics
# ─────────────────────────────────────────────────────────────────────────────

class Question(Base):
    __tablename__ = "questions"

    question_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[QuestionCategory] = mapped_column(
        Enum(QuestionCategory), nullable=False
    )
    domain: Mapped[str | None] = mapped_column(String(100))
    difficulty: Mapped[str | None] = mapped_column(String(50))
    company_tag: Mapped[str | None] = mapped_column(String(100))
    active_flag: Mapped[bool] = mapped_column(Boolean, default=True)

    rubrics: Mapped[list["Rubric"]] = relationship(back_populates="question")
    sessions: Mapped[list["InterviewSession"]] = relationship(back_populates="question")


class Rubric(Base):
    __tablename__ = "rubrics"

    rubric_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("questions.question_id", ondelete="CASCADE"), nullable=False
    )
    criterion_name: Mapped[str] = mapped_column(String(100), nullable=False)
    criterion_type: Mapped[CriterionType] = mapped_column(
        Enum(CriterionType), nullable=False
    )
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    # JSON list of expected concept strings
    expected_concepts: Mapped[dict | list | None] = mapped_column(JSONB)
    star_component: Mapped[bool] = mapped_column(Boolean, default=False)
    # Full scoring guide text
    scoring_guide: Mapped[str | None] = mapped_column(Text)

    question: Mapped["Question"] = relationship(back_populates="rubrics")


# ─────────────────────────────────────────────────────────────────────────────
# Sessions + Responses
# ─────────────────────────────────────────────────────────────────────────────

class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    session_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True
    )
    question_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("questions.question_id", ondelete="SET NULL"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    mode: Mapped[SessionMode] = mapped_column(
        Enum(SessionMode), default=SessionMode.PRACTICE
    )
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus), default=SessionStatus.ACTIVE
    )

    user: Mapped["User | None"] = relationship(back_populates="sessions")
    question: Mapped["Question | None"] = relationship(back_populates="sessions")
    responses: Mapped[list["Response"]] = relationship(back_populates="session")


class Response(Base):
    __tablename__ = "responses"

    response_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("interview_sessions.session_id", ondelete="CASCADE"), nullable=False
    )
    audio_path: Mapped[str | None] = mapped_column(String(255))
    duration_sec: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped["InterviewSession"] = relationship(back_populates="responses")
    transcripts: Mapped[list["Transcript"]] = relationship(back_populates="response")
    content_evaluations: Mapped[list["ContentEvaluation"]] = relationship(back_populates="response")
    delivery_metrics: Mapped[list["DeliveryMetrics"]] = relationship(back_populates="response")
    audit_results: Mapped[list["AuditResult"]] = relationship(back_populates="response")
    feedback_reports: Mapped[list["FeedbackReport"]] = relationship(back_populates="response")
    annotations: Mapped[list["HumanAnnotation"]] = relationship(back_populates="response")


# ─────────────────────────────────────────────────────────────────────────────
# Transcript
# ─────────────────────────────────────────────────────────────────────────────

class Transcript(Base):
    __tablename__ = "transcripts"

    transcript_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False
    )
    transcript_text: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON list of {word, start, end, probability} dicts
    word_timestamps: Mapped[list | None] = mapped_column(JSONB)
    stt_model: Mapped[str | None] = mapped_column(String(100))
    confidence: Mapped[float | None] = mapped_column(Float)
    # "whisper" | "gold" | "mock"
    source: Mapped[str] = mapped_column(String(50), default="whisper")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    response: Mapped["Response"] = relationship(back_populates="transcripts")


# ─────────────────────────────────────────────────────────────────────────────
# Content Evaluation
# ─────────────────────────────────────────────────────────────────────────────

class ContentEvaluation(Base):
    __tablename__ = "content_evaluations"

    content_eval_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False
    )
    # Per-criterion scores
    correctness_score: Mapped[float | None] = mapped_column(Float)
    coverage_score: Mapped[float | None] = mapped_column(Float)
    relevance_score: Mapped[float | None] = mapped_column(Float)
    completeness_score: Mapped[float | None] = mapped_column(Float)
    star_score: Mapped[float | None] = mapped_column(Float)
    # Composite weighted score
    overall_score: Mapped[float | None] = mapped_column(Float)
    # Full structured evidence: {criterion: {score, covered, missing, evidence, star_component}}
    evidence_json: Mapped[dict | None] = mapped_column(JSONB)
    # Whether delivery features were fed in (ablation flag)
    used_delivery_features: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_call_log_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("llm_call_logs.log_id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    response: Mapped["Response"] = relationship(back_populates="content_evaluations")
    llm_call_log: Mapped["LLMCallLog | None"] = relationship()


# ─────────────────────────────────────────────────────────────────────────────
# Delivery Metrics
# ─────────────────────────────────────────────────────────────────────────────

class DeliveryMetrics(Base):
    __tablename__ = "delivery_metrics"

    delivery_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False
    )
    speaking_rate: Mapped[float | None] = mapped_column(Float)        # WPM
    pause_count: Mapped[int | None] = mapped_column(Integer)
    long_pause_count: Mapped[int | None] = mapped_column(Integer)     # > threshold
    filler_count: Mapped[int | None] = mapped_column(Integer)
    filler_density: Mapped[float | None] = mapped_column(Float)       # fillers per 100 words
    repetition_count: Mapped[int | None] = mapped_column(Integer)
    duration_sec: Mapped[int | None] = mapped_column(Integer)
    # Detailed breakdown: {pause_positions, filler_positions, repetitions_detail}
    detail_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    response: Mapped["Response"] = relationship(back_populates="delivery_metrics")


# ─────────────────────────────────────────────────────────────────────────────
# Audit Results
# ─────────────────────────────────────────────────────────────────────────────

class AuditResult(Base):
    __tablename__ = "audit_results"

    audit_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False
    )
    delivery_invariance_error: Mapped[float | None] = mapped_column(Float)
    content_sensitivity: Mapped[float | None] = mapped_column(Float)
    paraphrase_deviation: Mapped[float | None] = mapped_column(Float)
    repeatability_variance: Mapped[float | None] = mapped_column(Float)
    stt_sensitivity: Mapped[float | None] = mapped_column(Float)
    uncertainty_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    # Full per-test details
    details_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    response: Mapped["Response"] = relationship(back_populates="audit_results")


# ─────────────────────────────────────────────────────────────────────────────
# Feedback Reports
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackReport(Base):
    __tablename__ = "feedback_reports"

    report_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False
    )
    summary_text: Mapped[str | None] = mapped_column(Text)
    strengths: Mapped[str | None] = mapped_column(Text)
    improvements: Mapped[str | None] = mapped_column(Text)
    # Full criterion-level structured feedback
    criterion_feedback_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    response: Mapped["Response"] = relationship(back_populates="feedback_reports")


# ─────────────────────────────────────────────────────────────────────────────
# Progress History
# ─────────────────────────────────────────────────────────────────────────────

class ProgressHistory(Base):
    __tablename__ = "progress_history"

    progress_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("interview_sessions.session_id", ondelete="CASCADE"), nullable=False
    )
    trend_metric: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g. "overall_score"
    value: Mapped[float] = mapped_column(Float, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Call Log  (reproducibility)
# ─────────────────────────────────────────────────────────────────────────────

class LLMCallLog(Base):
    __tablename__ = "llm_call_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    call_type: Mapped[str] = mapped_column(String(50), nullable=False)  # content_eval | holistic | paraphrase
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_output: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_ok: Mapped[bool] = mapped_column(Boolean, default=True)
    temperature: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Human Annotations  (inter-annotator agreement)
# ─────────────────────────────────────────────────────────────────────────────

class HumanAnnotation(Base):
    __tablename__ = "human_annotations"

    annotation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    annotator_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    response_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False
    )
    criterion: Mapped[str] = mapped_column(String(100), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    annotator: Mapped["User"] = relationship(back_populates="annotations")
    response: Mapped["Response"] = relationship(back_populates="annotations")
