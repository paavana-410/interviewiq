"""UI-facing interview endpoints.

POST /interview/sessions
    Create a new interview session with mode + n_questions.
    Selects questions from the seeded bank; stores their IDs on the session.

GET  /interview/sessions/{id}/next-question
    Return the next question in order, or {done: true} at the end.

POST /interview/sessions/{id}/responses   (audio upload)
    Runs STT + Delivery + Content (audit OFF by default via AUDIT_INLINE).
    Returns full SlimResult JSON.

GET  /interview/sessions/{id}/summary
    All answers for the session: per-criterion scores, evidence,
    delivery metrics, transcript, judge info.

GET  /interview/judge-info
    Safe (no keys) info about the active judge model — for the UI banner.
"""
from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models.core import (
    ContentEvaluation,
    DeliveryMetrics,
    InterviewSession,
    Question,
    QuestionCategory,
    Response,
    SessionMode,
    SessionStatus,
    Transcript,
)
from app.orchestration.slim_pipeline import (
    SlimResult,
    UIJudgeNotConfigured,
    WhisperUnavailable,
    run_slim_pipeline,
)
from app.utils.audio import audio_dep_info, save_upload

router = APIRouter(prefix="/interview", tags=["interview-ui"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_MODE_MAP = {
    "behavioral": [QuestionCategory.BEHAVIORAL, QuestionCategory.HR],
    "technical":  [QuestionCategory.TECHNICAL],
    "mixed":      [QuestionCategory.BEHAVIORAL, QuestionCategory.HR, QuestionCategory.TECHNICAL],
}


def _pick_questions(db: Session, mode: str, n: int, seed: int) -> list[Question]:
    categories = _MODE_MAP.get(mode, list(QuestionCategory))
    qs = (
        db.query(Question)
        .filter(Question.category.in_(categories), Question.active_flag == True)  # noqa: E712
        .all()
    )
    if not qs:
        return []
    rng = random.Random(seed)
    return rng.sample(qs, min(n, len(qs)))


def _session_question_ids(session: InterviewSession) -> list[int]:
    """Read the ordered question IDs stored in the session's JSON metadata field."""
    raw = getattr(session, "meta_json", None) or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    return raw.get("question_ids", [])


def _set_session_question_ids(session: InterviewSession, ids: list[int]) -> None:
    raw = getattr(session, "meta_json", None) or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    raw["question_ids"] = ids
    session.meta_json = raw  # type: ignore[assignment]


def _answered_question_ids(session_id: int, db: Session) -> set[int]:
    """Return the set of question IDs that already have a response + transcript."""
    rows = (
        db.query(Response, Transcript)
        .join(Transcript, Transcript.response_id == Response.response_id)
        .filter(Response.session_id == session_id)
        .all()
    )
    result = set()
    for resp, _ in rows:
        if resp.question_id:
            result.add(resp.question_id)
    return result


def _get_judge_info() -> dict:
    from app.config import get_settings
    s = get_settings()
    if s.llm_provider == "mock":
        return {"provider": "mock", "model": "mock-llm", "is_mock": True}
    if not s.ui_judge:
        return {"provider": "unconfigured", "model": "UI_JUDGE not set", "is_mock": False, "error": True}
    try:
        from app.services.judge_config import get_judge_registry
        reg = get_judge_registry()
        cfg = reg.get_judge(s.ui_judge)
        return {"provider": cfg.provider, "model": cfg.model, "is_mock": False}
    except Exception as exc:
        return {"provider": "error", "model": str(exc)[:80], "is_mock": False, "error": True}


# ─────────────────────────────────────────────────────────────────────────────
# Session creation
# ─────────────────────────────────────────────────────────────────────────────

class InterviewSessionCreate(BaseModel):
    mode: str = "mixed"          # behavioral | technical | mixed
    n_questions: int = 5         # 3–6


class InterviewSessionResponse(BaseModel):
    session_id: int
    mode: str
    n_questions: int
    question_ids: list[int]


@router.post("/sessions", response_model=InterviewSessionResponse, status_code=201)
def create_interview_session(
    body: InterviewSessionCreate,
    db: Session = Depends(get_db),
):
    mode = body.mode.lower()
    if mode not in _MODE_MAP:
        raise HTTPException(status_code=422, detail=f"mode must be one of {list(_MODE_MAP)}")
    n = max(1, min(body.n_questions, 10))

    questions = _pick_questions(db, mode, n, seed=random.randint(0, 999999))
    if not questions:
        raise HTTPException(
            status_code=503,
            detail="No questions found. Run: python -m app.seed.questions",
        )

    session = InterviewSession(
        mode=SessionMode.PRACTICE,
        status=SessionStatus.ACTIVE,
    )
    db.add(session)
    db.flush()

    # Store ordered question IDs as JSON in the session — reuse meta_json
    _set_session_question_ids(session, [q.question_id for q in questions])
    db.commit()
    db.refresh(session)

    return InterviewSessionResponse(
        session_id=session.session_id,
        mode=mode,
        n_questions=len(questions),
        question_ids=[q.question_id for q in questions],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Next question
# ─────────────────────────────────────────────────────────────────────────────

class NextQuestionResponse(BaseModel):
    done: bool
    question_id: int | None = None
    question_text: str | None = None
    category: str | None = None
    question_number: int | None = None    # 1-based
    total_questions: int | None = None


@router.get("/sessions/{session_id}/next-question", response_model=NextQuestionResponse)
def next_question(session_id: int, db: Session = Depends(get_db)):
    session = db.get(InterviewSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    all_ids = _session_question_ids(session)
    if not all_ids:
        raise HTTPException(status_code=400, detail="Session has no questions — recreate it.")

    answered = _answered_question_ids(session_id, db)
    remaining = [qid for qid in all_ids if qid not in answered]

    if not remaining:
        return NextQuestionResponse(done=True)

    next_qid = remaining[0]
    q = db.get(Question, next_qid)
    if q is None:
        return NextQuestionResponse(done=True)

    return NextQuestionResponse(
        done=False,
        question_id=q.question_id,
        question_text=q.question_text,
        category=q.category.value,
        question_number=len(answered) + 1,
        total_questions=len(all_ids),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Audio response upload → slim pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SlimResultResponse(BaseModel):
    response_id: int
    question_text: str
    transcript: str
    transcript_source: str        # "whisper" | "synthetic" | "mock"
    wpm: float
    pause_count: int
    long_pause_count: int
    filler_count: int
    filler_density: float
    repetition_count: int
    duration_sec: float
    overall_score: float
    correctness_score: float | None
    coverage_score: float | None
    relevance_score: float | None
    completeness_score: float | None
    star_score: float | None
    evidence_json: dict[str, Any]
    uncertainty: bool
    judge_model: str
    judge_provider: str
    is_mock: bool


@router.post(
    "/sessions/{session_id}/responses",
    response_model=SlimResultResponse,
    status_code=201,
)
async def submit_interview_response(
    session_id: int,
    question_id: int,
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    settings = get_settings()

    session = db.get(InterviewSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    raw = await audio.read()
    max_bytes = settings.max_audio_size_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Audio exceeds {settings.max_audio_size_mb} MB")

    audio_path = save_upload(raw, Path(settings.audio_upload_dir), audio.filename or "audio.webm")

    response = Response(
        session_id=session_id,
        question_id=question_id,
        audio_path=str(audio_path),
    )
    db.add(response)
    db.commit()
    db.refresh(response)

    try:
        result = run_slim_pipeline(response_id=response.response_id, db=db)
    except UIJudgeNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except WhisperUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")

    return SlimResultResponse(
        response_id=result.response_id,
        question_text=result.question_text,
        transcript=result.transcript,
        transcript_source=result.transcript_source,
        wpm=result.wpm,
        pause_count=result.pause_count,
        long_pause_count=result.long_pause_count,
        filler_count=result.filler_count,
        filler_density=result.filler_density,
        repetition_count=result.repetition_count,
        duration_sec=result.duration_sec,
        overall_score=result.overall_score,
        correctness_score=result.correctness_score,
        coverage_score=result.coverage_score,
        relevance_score=result.relevance_score,
        completeness_score=result.completeness_score,
        star_score=result.star_score,
        evidence_json=result.evidence_json,
        uncertainty=result.uncertainty,
        judge_model=result.judge_model,
        judge_provider=result.judge_provider,
        is_mock=result.is_mock,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session summary
# ─────────────────────────────────────────────────────────────────────────────

class QuestionSummary(BaseModel):
    question_id: int
    question_text: str
    transcript: str
    overall_score: float | None
    correctness_score: float | None
    coverage_score: float | None
    relevance_score: float | None
    completeness_score: float | None
    star_score: float | None
    evidence_json: dict[str, Any]
    wpm: float | None
    pause_count: int | None
    long_pause_count: int | None
    filler_count: int | None
    filler_density: float | None
    duration_sec: int | None


class SessionSummary(BaseModel):
    session_id: int
    total_questions: int
    answers: list[QuestionSummary]
    judge_provider: str
    judge_model: str
    is_mock: bool


@router.get("/sessions/{session_id}/summary", response_model=SessionSummary)
def session_summary(session_id: int, db: Session = Depends(get_db)):
    session = db.get(InterviewSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    all_ids = _session_question_ids(session)
    responses = (
        db.query(Response)
        .filter(Response.session_id == session_id)
        .all()
    )

    answers = []
    for resp in responses:
        q = db.get(Question, resp.question_id) if resp.question_id else None
        transcript = (
            db.query(Transcript)
            .filter(Transcript.response_id == resp.response_id)
            .order_by(Transcript.created_at.desc())
            .first()
        )
        ce = (
            db.query(ContentEvaluation)
            .filter(ContentEvaluation.response_id == resp.response_id)
            .order_by(ContentEvaluation.created_at.desc())
            .first()
        )
        dm = (
            db.query(DeliveryMetrics)
            .filter(DeliveryMetrics.response_id == resp.response_id)
            .order_by(DeliveryMetrics.created_at.desc())
            .first()
        )
        answers.append(QuestionSummary(
            question_id=resp.question_id or 0,
            question_text=q.question_text if q else "",
            transcript=transcript.transcript_text if transcript else "",
            overall_score=ce.overall_score if ce else None,
            correctness_score=ce.correctness_score if ce else None,
            coverage_score=ce.coverage_score if ce else None,
            relevance_score=ce.relevance_score if ce else None,
            completeness_score=ce.completeness_score if ce else None,
            star_score=ce.star_score if ce else None,
            evidence_json=ce.evidence_json if ce else {},
            wpm=dm.speaking_rate if dm else None,
            pause_count=dm.pause_count if dm else None,
            long_pause_count=dm.long_pause_count if dm else None,
            filler_count=dm.filler_count if dm else None,
            filler_density=dm.filler_density if dm else None,
            duration_sec=dm.duration_sec if dm else None,
        ))

    ji = _get_judge_info()
    return SessionSummary(
        session_id=session_id,
        total_questions=len(all_ids),
        answers=answers,
        judge_provider=ji["provider"],
        judge_model=ji["model"],
        is_mock=ji["is_mock"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Judge info (for UI banner)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/judge-info")
def judge_info():
    return _get_judge_info()


# ─────────────────────────────────────────────────────────────────────────────
# Audio dependency status (for debug/health)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/audio-deps")
def get_audio_deps():
    return audio_dep_info()
