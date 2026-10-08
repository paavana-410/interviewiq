"""POST /sessions/{id}/responses  — audio upload + full evaluation pipeline.
GET  /responses/{id}/evaluation  — fetch persisted evaluation result.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.schemas import (
    AuditResultSchema,
    ContentEvalSchema,
    DeliveryMetricsSchema,
    EvaluationResponse,
    ResponseCreated,
)
from app.config import get_settings
from app.database import get_db
from app.models.core import (
    AuditResult,
    ContentEvaluation,
    DeliveryMetrics,
    FeedbackReport,
    InterviewSession,
    Response,
    Transcript,
)
from app.orchestration.pipeline import run_evaluation_pipeline

router = APIRouter(tags=["responses"])


# ─────────────────────────────────────────────────────────────────────────────
# POST /sessions/{session_id}/responses
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/responses", response_model=ResponseCreated, status_code=201)
async def submit_response(
    session_id: int,
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    settings = get_settings()

    # Validate session exists
    session = db.get(InterviewSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check file size
    content = await audio.read()
    max_bytes = settings.max_audio_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds {settings.max_audio_size_mb} MB limit",
        )

    # Persist audio file
    ext = Path(audio.filename or "audio.wav").suffix or ".wav"
    filename = f"{uuid.uuid4().hex}{ext}"
    audio_path = Path(settings.audio_upload_dir) / filename
    audio_path.write_bytes(content)

    # Create response record
    response = Response(
        session_id=session_id,
        audio_path=str(audio_path),
    )
    db.add(response)
    db.commit()
    db.refresh(response)

    # Run evaluation pipeline synchronously
    # (for production you'd push this to a task queue, but per spec we do it inline)
    try:
        run_evaluation_pipeline(response_id=response.response_id, db=db)
    except Exception as exc:
        # Don't fail the upload if evaluation has an issue — caller can retry via GET
        import traceback
        traceback.print_exc()

    return ResponseCreated(
        response_id=response.response_id,
        session_id=session_id,
        audio_path=str(audio_path),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /responses/{response_id}/evaluation
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/responses/{response_id}/evaluation", response_model=EvaluationResponse)
def get_evaluation(response_id: int, db: Session = Depends(get_db)):
    response = db.get(Response, response_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Response not found")

    # Latest transcript
    transcript = (
        db.query(Transcript)
        .filter(Transcript.response_id == response_id)
        .order_by(Transcript.created_at.desc())
        .first()
    )
    transcript_text = transcript.transcript_text if transcript else ""

    # Latest content evaluation
    content_eval = (
        db.query(ContentEvaluation)
        .filter(ContentEvaluation.response_id == response_id)
        .order_by(ContentEvaluation.created_at.desc())
        .first()
    )

    # Latest delivery metrics
    delivery = (
        db.query(DeliveryMetrics)
        .filter(DeliveryMetrics.response_id == response_id)
        .order_by(DeliveryMetrics.created_at.desc())
        .first()
    )

    # Latest audit result
    audit = (
        db.query(AuditResult)
        .filter(AuditResult.response_id == response_id)
        .order_by(AuditResult.created_at.desc())
        .first()
    )

    content_schema = None
    if content_eval:
        content_schema = ContentEvalSchema(
            overall_score=content_eval.overall_score,
            correctness_score=content_eval.correctness_score,
            coverage_score=content_eval.coverage_score,
            relevance_score=content_eval.relevance_score,
            completeness_score=content_eval.completeness_score,
            star_score=content_eval.star_score,
            evidence=content_eval.evidence_json or {},
            used_delivery_features=content_eval.used_delivery_features,
            uncertainty=False,
        )

    delivery_schema = None
    if delivery:
        detail = delivery.detail_json or {}
        delivery_schema = DeliveryMetricsSchema(
            speaking_rate_wpm=delivery.speaking_rate,
            pause_count=delivery.pause_count,
            long_pause_count=delivery.long_pause_count,
            filler_count=delivery.filler_count,
            filler_density=delivery.filler_density,
            repetition_count=delivery.repetition_count,
            duration_sec=delivery.duration_sec,
            detail=detail,
        )

    audit_schema = None
    if audit:
        audit_schema = AuditResultSchema(
            delivery_invariance_error=audit.delivery_invariance_error,
            content_sensitivity=audit.content_sensitivity,
            paraphrase_deviation=audit.paraphrase_deviation,
            repeatability_variance=audit.repeatability_variance,
            stt_sensitivity=audit.stt_sensitivity,
            uncertainty_flag=audit.uncertainty_flag,
            details=audit.details_json or {},
        )

    return EvaluationResponse(
        response_id=response_id,
        session_id=response.session_id,
        transcript=transcript_text,
        delivery=delivery_schema,
        content=content_schema,
        audit=audit_schema,
        uncertainty_flag=audit.uncertainty_flag if audit else False,
    )
