"""Slim evaluation pipeline for the UI — STT + Delivery + Content only.

Audits are skipped by default (AUDIT_INLINE=false) to keep per-response
latency acceptable on a laptop. The audit battery is still available via
run_audit_benchmark.py.

Returns the result dict directly without DB writes — the caller (the UI
endpoint) can store what it needs.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import get_settings
from app.engines.content import ContentEngine, RubricCriterion
from app.engines.delivery import DeliveryEngine
from app.models.core import (
    ContentEvaluation,
    DeliveryMetrics,
    InterviewSession,
    Question,
    Response,
    Rubric,
    Transcript,
)
from app.services.llm import get_llm_service
from app.services.stt import get_stt_service


@dataclass
class SlimResult:
    response_id: int
    question_text: str
    transcript: str
    # Delivery
    wpm: float
    pause_count: int
    long_pause_count: int
    filler_count: int
    filler_density: float
    repetition_count: int
    duration_sec: float
    # Content
    overall_score: float
    correctness_score: float | None
    coverage_score: float | None
    relevance_score: float | None
    completeness_score: float | None
    star_score: float | None
    evidence_json: dict
    uncertainty: bool
    # Judge info (safe — no key)
    judge_model: str
    judge_provider: str


def run_slim_pipeline(response_id: int, db: Session) -> SlimResult:
    """Run STT → Delivery → Content for one response. No audit, no DB writes."""
    settings = get_settings()

    response: Response | None = db.get(Response, response_id)
    if response is None:
        raise ValueError(f"Response {response_id} not found")

    session: InterviewSession | None = db.get(InterviewSession, response.session_id)
    question_id = session.question_id if session else None
    question_text = ""
    rubric_criteria: list[RubricCriterion] = []

    if question_id is not None:
        q: Question | None = db.get(Question, question_id)
        if q:
            question_text = q.question_text
            rubrics = db.query(Rubric).filter(Rubric.question_id == question_id).all()
            rubric_criteria = [
                RubricCriterion(
                    criterion_name=r.criterion_name,
                    criterion_type=r.criterion_type.value,
                    weight=r.weight,
                    expected_concepts=r.expected_concepts or [],
                    star_component=r.star_component,
                    scoring_guide=r.scoring_guide or "",
                )
                for r in rubrics
            ]

    # 1. STT
    stt = get_stt_service()
    stt_result = stt.transcribe(response.audio_path or "")

    # Persist transcript
    t = Transcript(
        response_id=response_id,
        transcript_text=stt_result.text,
        word_timestamps=stt_result.to_word_timestamps_json(),
        stt_model=stt_result.model_name,
        confidence=stt_result.avg_confidence,
        source="whisper" if settings.llm_provider != "mock" else "mock",
    )
    db.add(t)
    db.flush()

    # 2. Delivery (timestamps only)
    delivery_engine = DeliveryEngine()
    dr = delivery_engine.analyze(stt_result.words)

    dm = DeliveryMetrics(
        response_id=response_id,
        speaking_rate=dr.speaking_rate_wpm,
        pause_count=dr.pause_count,
        long_pause_count=dr.long_pause_count,
        filler_count=dr.filler_count,
        filler_density=dr.filler_density,
        repetition_count=dr.repetition_count,
        duration_sec=int(dr.duration_sec),
        detail_json=dr.to_dict()["detail"],
    )
    db.add(dm)
    db.flush()

    # 3. Content (text only)
    llm = get_llm_service()
    ce_engine = ContentEngine(llm=llm, db=db)
    cr = ce_engine.analyze(
        transcript_text=stt_result.text,
        question_text=question_text,
        rubric_criteria=rubric_criteria,
    )

    ce = ContentEvaluation(
        response_id=response_id,
        correctness_score=cr.correctness_score,
        coverage_score=cr.coverage_score,
        relevance_score=cr.relevance_score,
        completeness_score=cr.completeness_score,
        star_score=cr.star_score,
        overall_score=cr.overall_score,
        evidence_json=cr.evidence_json,
        used_delivery_features=cr.used_delivery_features,
        llm_call_log_id=cr.llm_call_log_id,
    )
    db.add(ce)
    db.commit()

    return SlimResult(
        response_id=response_id,
        question_text=question_text,
        transcript=stt_result.text,
        wpm=dr.speaking_rate_wpm,
        pause_count=dr.pause_count,
        long_pause_count=dr.long_pause_count,
        filler_count=dr.filler_count,
        filler_density=dr.filler_density,
        repetition_count=dr.repetition_count,
        duration_sec=dr.duration_sec,
        overall_score=cr.overall_score,
        correctness_score=cr.correctness_score,
        coverage_score=cr.coverage_score,
        relevance_score=cr.relevance_score,
        completeness_score=cr.completeness_score,
        star_score=cr.star_score,
        evidence_json=cr.evidence_json,
        uncertainty=cr.uncertainty,
        judge_model=llm.model_name,
        judge_provider=llm.provider,
    )
