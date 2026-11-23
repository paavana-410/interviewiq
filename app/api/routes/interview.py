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
import hashlib
import random
import secrets
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

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
    User,
)
from app.orchestration.slim_pipeline import (
    SlimResult,
    UIJudgeNotConfigured,
    WhisperUnavailable,
    run_slim_pipeline,
)
from app.utils.audio import audio_dep_info, save_upload

router = APIRouter(prefix="/interview", tags=["interview-ui"])


_MODE_MAP = {
    "behavioral": [QuestionCategory.BEHAVIORAL, QuestionCategory.HR],
    "technical":  [QuestionCategory.TECHNICAL],
    "mixed":      [QuestionCategory.BEHAVIORAL, QuestionCategory.HR, QuestionCategory.TECHNICAL],
}


def _build_interview_flow(db: Session, mode: str, difficulty: str = "easy", seed: int = 42) -> list[Question]:
    """Construct an interview flow adhering strictly to target category & followup proportions."""
    rng = random.Random(seed)
    diff = (difficulty or "easy").lower()
    mode_clean = (mode or "mixed").lower()

    def get_followups(base_qid: int) -> list[Question]:
        return (
            db.query(Question)
            .filter(
                Question.company_tag == f"followup_for_{base_qid}",
                Question.active_flag == True
            )
            .all()
        )

    def fetch_bases(cats: list[QuestionCategory], target_diff: str) -> list[Question]:
        matching = (
            db.query(Question)
            .filter(
                Question.category.in_(cats),
                Question.active_flag == True,
                Question.question_id != 1,
                (Question.difficulty == target_diff),
                (Question.company_tag == None) | (~Question.company_tag.like("followup_for_%"))
            )
            .all()
        )
        rng.shuffle(matching)
        fallback = (
            db.query(Question)
            .filter(
                Question.category.in_(cats),
                Question.active_flag == True,
                Question.question_id != 1,
                (Question.company_tag == None) | (~Question.company_tag.like("followup_for_%")),
                Question.question_id.notin_([q.question_id for q in matching])
            )
            .all()
        )
        rng.shuffle(fallback)
        return matching + fallback

    flow: list[Question] = []

    if mode_clean == "mixed":
        # Target proportions:
        # Easy:   2 Tech + 2 Follow-ups + 2 Beh = 6 Qs
        # Medium: 3 Tech + 2 Follow-ups + 3 Beh = 8 Qs
        # Hard:   3 Tech + 3 Follow-ups + 3 Beh = 9 Qs
        num_tech = 2 if diff == "easy" else 3
        num_fol = 2 if diff in ("easy", "medium") else 3
        num_beh = 2 if diff == "easy" else 3

        tech_bases = fetch_bases([QuestionCategory.TECHNICAL], diff)
        beh_bases = fetch_bases([QuestionCategory.BEHAVIORAL, QuestionCategory.HR], diff)

        # 1. Add Technical questions with interleaving followups
        added_fol = 0
        for i in range(num_tech):
            if i < len(tech_bases):
                tb = tech_bases[i]
                flow.append(tb)
                fols = get_followups(tb.question_id)
                if fols and added_fol < num_fol:
                    flow.append(fols[0])
                    added_fol += 1

        # Fill any remaining followups if bases didn't have enough
        while added_fol < num_fol:
            for tb in tech_bases:
                fols = get_followups(tb.question_id)
                for f in fols:
                    if f not in flow and added_fol < num_fol:
                        flow.append(f)
                        added_fol += 1
            break

        # 2. Add Behavioral questions
        for i in range(num_beh):
            if i < len(beh_bases):
                flow.append(beh_bases[i])

    elif mode_clean == "behavioral":
        # Target proportions:
        # Easy:   3 Core Beh + 2 Follow-ups = 5 Qs
        # Medium: 4 Core Beh + 2 Follow-ups = 6 Qs
        # Hard:   4 Core Beh + 3 Follow-ups = 7 Qs
        num_beh = 3 if diff == "easy" else 4
        num_fol = 2 if diff in ("easy", "medium") else 3

        beh_bases = fetch_bases([QuestionCategory.BEHAVIORAL, QuestionCategory.HR], diff)
        added_fol = 0
        for i in range(num_beh):
            if i < len(beh_bases):
                bb = beh_bases[i]
                flow.append(bb)
                fols = get_followups(bb.question_id)
                if fols and added_fol < num_fol:
                    flow.append(fols[0])
                    added_fol += 1

        # If any followups left, append
        while added_fol < num_fol:
            for bb in beh_bases:
                fols = get_followups(bb.question_id)
                for f in fols:
                    if f not in flow and added_fol < num_fol:
                        flow.append(f)
                        added_fol += 1
            break

    else:  # "technical"
        # Target proportions:
        # Easy:   3 Core Tech + 2 Follow-ups = 5 Qs
        # Medium: 4 Core Tech + 2 Follow-ups = 6 Qs
        # Hard:   4 Core Tech + 3 Follow-ups = 7 Qs
        num_tech = 3 if diff == "easy" else 4
        num_fol = 2 if diff in ("easy", "medium") else 3

        tech_bases = fetch_bases([QuestionCategory.TECHNICAL], diff)
        added_fol = 0
        for i in range(num_tech):
            if i < len(tech_bases):
                tb = tech_bases[i]
                flow.append(tb)
                fols = get_followups(tb.question_id)
                if fols and added_fol < num_fol:
                    flow.append(fols[0])
                    added_fol += 1

        # If any followups left, append
        while added_fol < num_fol:
            for tb in tech_bases:
                fols = get_followups(tb.question_id)
                for f in fols:
                    if f not in flow and added_fol < num_fol:
                        flow.append(f)
                        added_fol += 1
            break

    return flow


def _pick_questions(db: Session, mode: str, n: int, seed: int) -> list[Question]:
    return _build_interview_flow(db, mode, "easy", seed)[:n]


def _session_question_ids(session: InterviewSession) -> list[int]:
    """Read the ordered question IDs stored in the session's JSON metadata field."""
    raw = getattr(session, "meta_json", None) or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    return raw.get("question_ids", [])


def _set_session_question_ids(session: InterviewSession, ids: list[int], intro_qid: int = 1) -> None:
    raw = getattr(session, "meta_json", None) or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    raw["question_ids"] = ids
    raw["intro_qid"] = intro_qid
    session.meta_json = raw  # type: ignore[assignment]


def _answered_question_ids(session_id: int, db: Session) -> set[int]:
    """Return the set of question IDs that have been answered in this session."""
    rows = (
        db.query(Response.question_id)
        .filter(Response.session_id == session_id, Response.question_id.isnot(None))
        .distinct()
        .all()
    )
    return {r[0] for r in rows if r[0] is not None}


def _get_judge_info() -> dict:
    from app.config import get_settings
    s = get_settings()
    if s.llm_provider == "mock":
        return {"provider": "mock", "model": "mock-llm", "is_mock": True}
    if getattr(s, "ui_judge_router", "static") == "fastest_race" or s.ui_judge == "race":
        return {"provider": "adaptive_race", "model": "fastest_race (liquid + nemotron)", "is_mock": False}
    if not s.ui_judge:
        return {"provider": "unconfigured", "model": "UI_JUDGE not set", "is_mock": False, "error": True}
    try:
        from app.services.judge_config import get_judge_registry
        reg = get_judge_registry()
        cfg = reg.get_judge(s.ui_judge)
        return {"provider": cfg.provider, "model": cfg.model, "is_mock": False}
    except Exception as exc:
        return {"provider": "error", "model": str(exc)[:80], "is_mock": False, "error": True}


def _hash_password(password: str, salt: str | None = None) -> str:
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${hashed}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, _ = stored_hash.split("$", 1)
        return _hash_password(password, salt) == stored_hash
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Session creation
# ─────────────────────────────────────────────────────────────────────────────

class InterviewSessionCreate(BaseModel):
    mode: str = "mixed"          # behavioral | technical | mixed
    difficulty: str = "easy"     # easy | medium | hard
    n_questions: int | None = None
    user_id: int | None = None


class InterviewSessionResponse(BaseModel):
    session_id: int
    mode: str
    difficulty: str
    n_questions: int
    question_ids: list[int]
    user_id: int | None = None


@router.post("/sessions", response_model=InterviewSessionResponse, status_code=201)
def create_interview_session(
    body: InterviewSessionCreate,
    db: Session = Depends(get_db),
):
    mode = body.mode.lower()
    if mode not in _MODE_MAP:
        raise HTTPException(status_code=422, detail=f"mode must be one of {list(_MODE_MAP)}")
    diff = (body.difficulty or "easy").lower()
    if diff not in ("easy", "medium", "hard"):
        diff = "easy"

    questions = _build_interview_flow(db, mode, diff, seed=random.randint(0, 999999))
    if not questions:
        raise HTTPException(
            status_code=503,
            detail="No questions found. Run: python -m app.seed.questions",
        )

    session = InterviewSession(
        user_id=body.user_id,
        mode=SessionMode.PRACTICE,
        status=SessionStatus.ACTIVE,
    )
    db.add(session)
    db.flush()

    # Store ordered question IDs as JSON in the session
    _set_session_question_ids(session, [q.question_id for q in questions])
    db.commit()
    db.refresh(session)

    return InterviewSessionResponse(
        session_id=session.session_id,
        mode=mode,
        difficulty=diff,
        n_questions=len(questions),
        question_ids=[q.question_id for q in questions],
        user_id=session.user_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Next question
# ─────────────────────────────────────────────────────────────────────────────

class NextQuestionResponse(BaseModel):
    done: bool
    question_id: int | None = None
    question_text: str | None = None
    category: str | None = None
    difficulty: str | None = None
    is_intro: bool = False
    is_followup: bool = False
    question_number: int | None = None    # 0 for intro stage, 1..N for core questions
    total_questions: int | None = None


@router.get("/sessions/{session_id}/next-question", response_model=NextQuestionResponse)
def next_question(session_id: int, db: Session = Depends(get_db)):
    session = db.get(InterviewSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    all_ids = _session_question_ids(session)
    if not all_ids:
        raise HTTPException(status_code=400, detail="Session has no questions — recreate it.")

    meta = getattr(session, "meta_json", None) or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    intro_qid = meta.get("intro_qid", 1)
    answered = _answered_question_ids(session_id, db)

    # 1. Candidate Introduction Stage (Question 0 - outside of question count)
    if intro_qid not in answered:
        intro_q = db.get(Question, intro_qid)
        intro_text = (
            intro_q.question_text if intro_q else "Please introduce yourself: Tell me about your background, key strengths, and career aspirations."
        )
        return NextQuestionResponse(
            done=False,
            question_id=intro_qid,
            question_text=intro_text,
            category="CANDIDATE INTRODUCTION",
            difficulty="general",
            is_intro=True,
            is_followup=False,
            question_number=0,
            total_questions=len(all_ids),
        )

    # 2. Core Interview Questions (Questions 1 through N)
    remaining = [qid for qid in all_ids if qid not in answered]

    if not remaining:
        return NextQuestionResponse(done=True)

    # Smart follow-up skipping: if the last response was passed, skip any follow-up tied to it
    last_resp = (
        db.query(Response)
        .filter(Response.session_id == session_id)
        .order_by(Response.created_at.desc(), Response.response_id.desc())
        .first()
    )
    if last_resp and last_resp.question_id:
        t = (
            db.query(Transcript)
            .filter(Transcript.response_id == last_resp.response_id)
            .order_by(Transcript.created_at.desc())
            .first()
        )
        ce = (
            db.query(ContentEvaluation)
            .filter(ContentEvaluation.response_id == last_resp.response_id)
            .order_by(ContentEvaluation.created_at.desc())
            .first()
        )
        is_passed = False
        if ce and ce.evidence_json and ce.evidence_json.get("verdict") in ("passed", "passed_zero_score"):
            is_passed = True
        elif t and "passed this question" in (t.transcript_text or "").lower():
            is_passed = True
        elif t and any(w in (t.transcript_text or "").lower() for w in ("shall pass", "i pass", "skip this", "don't know", "dont know")):
            is_passed = True

        if is_passed:
            parent_id = last_resp.question_id
            filtered = []
            for qid in remaining:
                cand_q = db.get(Question, qid)
                if cand_q and cand_q.company_tag == f"followup_for_{parent_id}":
                    continue  # skip follow-up for a passed question
                filtered.append(qid)
            if filtered:
                remaining = filtered

    next_qid = remaining[0]
    q = db.get(Question, next_qid)
    if q is None:
        return NextQuestionResponse(done=True)

    is_fol = bool(
        (q.company_tag and q.company_tag.startswith("followup_for_"))
        or q.question_text.startswith("Follow-up:")
    )

    answered_core_count = len([qid for qid in answered if qid != intro_qid])
    return NextQuestionResponse(
        done=False,
        question_id=q.question_id,
        question_text=q.question_text,
        category=q.category.value,
        difficulty=q.difficulty or "medium",
        is_intro=False,
        is_followup=is_fol,
        question_number=answered_core_count + 1,
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
    is_conversational: bool = False
    intent: str = "answer"
    interviewer_reply: str = ""


@router.post(
    "/sessions/{session_id}/responses",
    response_model=SlimResultResponse,
    status_code=201,
)
def submit_interview_response(
    session_id: int,
    question_id: int,
    audio: UploadFile | None = File(None),
    transcript_text: str | None = Form(None),
    db: Session = Depends(get_db),
):
    settings = get_settings()

    session = db.get(InterviewSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    audio_path = None
    if audio and audio.filename:
        raw = audio.file.read()
        max_bytes = settings.max_audio_size_mb * 1024 * 1024
        if len(raw) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Audio exceeds {settings.max_audio_size_mb} MB")
        audio_path = save_upload(raw, Path(settings.audio_upload_dir), audio.filename or "audio.webm")

    response = Response(
        session_id=session_id,
        question_id=question_id,
        audio_path=str(audio_path) if audio_path else None,
    )
    db.add(response)
    db.commit()
    db.refresh(response)

    try:
        result = run_slim_pipeline(
            response_id=response.response_id,
            db=db,
            transcript_override=transcript_text,
        )
    except UIJudgeNotConfigured as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc))
    except WhisperUnavailable as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        db.rollback()
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
        is_conversational=getattr(result, "is_conversational", False),
        intent=getattr(result, "intent", "answer"),
        interviewer_reply=getattr(result, "interviewer_reply", ""),
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


class AggregatedPerformance(BaseModel):
    overall_score: float             # 0 to 100
    content_score: float             # 0 to 100
    delivery_score: float            # 0 to 100
    average_wpm: float
    total_fillers: int
    total_long_pauses: int
    total_duration_sec: int
    verdict: str


class SessionSummary(BaseModel):
    session_id: int
    total_questions: int
    introduction: QuestionSummary | None = None
    answers: list[QuestionSummary]
    code_submissions: list[dict] = []
    aggregated: AggregatedPerformance | None = None
    improvement_suggestions: list[str] = []
    judge_provider: str
    judge_model: str
    is_mock: bool


@router.get("/sessions/{session_id}/summary", response_model=SessionSummary)
def session_summary(session_id: int, db: Session = Depends(get_db)):
    session = db.get(InterviewSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    meta = getattr(session, "meta_json", None) or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    intro_qid = meta.get("intro_qid", 1)
    all_ids = _session_question_ids(session)

    responses = (
        db.query(Response)
        .filter(Response.session_id == session_id)
        .order_by(Response.created_at.asc(), Response.response_id.asc())
        .all()
    )

    intro_summary: QuestionSummary | None = None
    core_answers: list[QuestionSummary] = []
    all_content_scores: list[float] = []
    all_wpms: list[float] = []
    tot_fillers = 0
    tot_pauses = 0
    tot_duration = 0

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
        qs = QuestionSummary(
            question_id=resp.question_id or 0,
            question_text=q.question_text if q else ("Candidate Introduction" if resp.question_id == intro_qid else ""),
            transcript=transcript.transcript_text if transcript else "",
            overall_score=ce.overall_score if ce else None,
            correctness_score=ce.correctness_score if ce else None,
            coverage_score=ce.coverage_score if ce else None,
            relevance_score=ce.relevance_score if ce else None,
            completeness_score=ce.completeness_score if ce else None,
            star_score=ce.star_score if ce else None,
            evidence_json=(ce.evidence_json if (ce and ce.evidence_json is not None) else {}),
            wpm=dm.speaking_rate if dm else None,
            pause_count=dm.pause_count if dm else None,
            long_pause_count=dm.long_pause_count if dm else None,
            filler_count=dm.filler_count if dm else None,
            filler_density=dm.filler_density if dm else None,
            duration_sec=dm.duration_sec if dm else None,
        )

        if resp.question_id == intro_qid and intro_summary is None:
            intro_summary = qs
        else:
            core_answers.append(qs)

        if ce and ce.overall_score is not None:
            all_content_scores.append(ce.overall_score)
        if dm and dm.speaking_rate:
            all_wpms.append(dm.speaking_rate)
        if dm:
            tot_fillers += dm.filler_count or 0
            tot_pauses += dm.long_pause_count or 0
            tot_duration += dm.duration_sec or 0

    avg_content = (sum(all_content_scores) / len(all_content_scores)) if all_content_scores else 0.0
    avg_wpm = (sum(all_wpms) / len(all_wpms)) if all_wpms else 0.0

    # Delivery score: only meaningful if there are actual spoken answers
    # If no one spoke (all passed / no WPM data), delivery is 0
    if not all_wpms or avg_wpm == 0:
        delivery_score = 0.0
    else:
        delivery_score = 100.0
        if avg_wpm < 110 or avg_wpm > 175:
            delivery_score -= 15.0
        if tot_fillers > 4:
            delivery_score -= min(25.0, tot_fillers * 3.0)
        if tot_pauses > 2:
            delivery_score -= min(15.0, tot_pauses * 3.0)
        delivery_score = max(20.0, min(100.0, delivery_score))

    content_score = round(avg_content * 100, 1)
    base_overall = round((content_score * 0.6) + (delivery_score * 0.4), 1)
    tot_penalty = float(meta.get("total_security_penalty", 0.0))
    overall_agg = max(0.0, round(base_overall - tot_penalty, 1))

    verdict = (
        "Outstanding Placement Ready" if overall_agg >= 80 else
        "Strong Performance" if overall_agg >= 65 else
        "Good Potential — Needs Practice" if overall_agg >= 45 else
        "Needs Dedicated Preparation"
    )

    agg = AggregatedPerformance(
        overall_score=overall_agg,
        content_score=content_score,
        delivery_score=round(delivery_score, 1),
        average_wpm=round(avg_wpm, 1),
        total_fillers=tot_fillers,
        total_long_pauses=tot_pauses,
        total_duration_sec=tot_duration,
        verdict=verdict,
    )

    suggestions = []
    if tot_penalty > 0:
        sec_count = len(meta.get("security_violations", []))
        suggestions.append(f"Security & Integrity: {sec_count} proctoring warning(s) incurred a total -{tot_penalty:.0f}% penalty on final assessment score.")
    if avg_wpm > 165:
        suggestions.append(f"Pacing: Your pace averaged {avg_wpm:.0f} WPM, which is slightly rushed. Aim for 130–150 WPM so your technical explanations carry clarity.")
    elif avg_wpm < 115 and avg_wpm > 0:
        suggestions.append(f"Pacing: Speaking speed was {avg_wpm:.0f} WPM. Work on steady rhythm and forward momentum.")
    else:
        suggestions.append(f"Pacing: Balanced speaking pace ({avg_wpm:.0f} WPM) within the ideal professional interview range.")

    if tot_fillers >= 4:
        suggestions.append(f"Filler Control: {tot_fillers} filler words detected across answers. Replace 'um/like' with intentional 1-second pauses while collecting thoughts.")
    else:
        suggestions.append("Vocal Polish: Clean articulation with minimal filler words.")

    if any(s < 0.5 for s in all_content_scores):
        suggestions.append("Technical Framework: On difficult questions, define the core mechanism first, then outline a real-world scenario and design trade-offs.")

    suggestions.append("STAR Method: For behavioral answers, clearly signpost Situation, Task, your specific Action, and measurable Result.")

    # Save final performance into session metadata for placement leaderboard
    code_subs = meta.get("code_submissions", [])
    avg_code_score = (sum(s.get("score_percent", 0) for s in code_subs) / len(code_subs)) if code_subs else None

    meta["final_performance"] = {
        "overall_score": overall_agg,
        "content_score": content_score,
        "delivery_score": round(delivery_score, 1),
        "code_score": round(avg_code_score, 1) if avg_code_score is not None else None,
        "verdict": verdict,
        "total_duration_sec": tot_duration,
        "answered_count": len(core_answers),
    }
    session.meta_json = dict(meta)
    flag_modified(session, "meta_json")
    db.commit()

    ji = _get_judge_info()
    return SessionSummary(
        session_id=session_id,
        total_questions=len(all_ids),
        introduction=intro_summary,
        answers=core_answers,
        code_submissions=meta.get("code_submissions", []),
        aggregated=agg,
        improvement_suggestions=suggestions,
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


# ─────────────────────────────────────────────────────────────────────────────
# User Authentication & Campus Placement Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str | None = None


class LoginRequest(BaseModel):
    password: str
    username: str | None = None
    username_or_email: str | None = None


class LinkUserRequest(BaseModel):
    user_id: int


class SecurityEventRequest(BaseModel):
    event_type: str  # "phone_detected" | "multiple_faces" | "no_face" | "multiple_voices" | "tab_switch" | "screenshot" | "suspicious_gaze"
    description: str | None = None
    score_penalty: float = 5.0


class PassQuestionRequest(BaseModel):
    question_id: int | None = None


def _seed_demo_leaderboard_if_empty(db: Session):
    existing = db.query(InterviewSession).filter(InterviewSession.user_id != None).first()
    if existing:
        return

    demo_users = [
        {"name": "Ananya Rao", "email": "ananya.rao@campus.edu", "overall": 92.5, "content": 94.0, "delivery": 90.0, "code": 100.0, "verdict": "Outstanding Placement Ready"},
        {"name": "Rahul Sharma", "email": "rahul.s@campus.edu", "overall": 86.0, "content": 88.0, "delivery": 83.0, "code": 83.3, "verdict": "Outstanding Placement Ready"},
        {"name": "Vikramaditya S", "email": "vikram.s@campus.edu", "overall": 74.5, "content": 76.0, "delivery": 72.0, "code": 66.7, "verdict": "Strong Performance"},
    ]
    for d in demo_users:
        u = db.query(User).filter(User.email == d["email"]).first()
        if not u:
            u = User(
                name=d["name"],
                email=d["email"],
                password_hash=_hash_password("DemoPass123!"),
                role="student"
            )
            db.add(u)
            db.flush()

        s = InterviewSession(
            user_id=u.user_id,
            mode=SessionMode.PRACTICE,
            status=SessionStatus.COMPLETED,
            meta_json={
                "final_performance": {
                    "overall_score": d["overall"],
                    "content_score": d["content"],
                    "delivery_score": d["delivery"],
                    "code_score": d["code"],
                    "verdict": d["verdict"],
                    "total_duration_sec": 340,
                    "answered_count": 5,
                }
            }
        )
        db.add(s)
    db.commit()


@router.post("/auth/register")
def register_user(body: RegisterRequest, db: Session = Depends(get_db)):
    username = body.username.strip()
    if len(username) < 2:
        raise HTTPException(status_code=400, detail="Username must be at least 2 characters long.")
    if len(body.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters long.")

    email = (body.email or f"{username}@example.com").strip().lower()

    existing_name = db.query(User).filter(func.lower(User.name) == username.lower()).first()
    if existing_name:
        raise HTTPException(status_code=400, detail="This username is already taken.")

    existing_email = db.query(User).filter(func.lower(User.email) == email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    pwd_hash = _hash_password(body.password)
    user = User(
        name=username,
        email=email,
        password_hash=pwd_hash,
        role="student"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "success": True,
        "message": "User registered successfully",
        "user_id": user.user_id,
        "username": user.name,
        "email": user.email,
        "user": {
            "user_id": user.user_id,
            "username": user.name,
            "email": user.email,
            "role": user.role,
        }
    }


@router.get("/auth/ping")
def ping():
    return {"status": "ok"}


@router.post("/auth/login")
def login_user(body: LoginRequest, db: Session = Depends(get_db)):
    ident = (body.username or body.username_or_email or "").strip()
    if not ident or not body.password:
        raise HTTPException(status_code=400, detail="Username and password required")

    user = (
        db.query(User)
        .filter((func.lower(User.email) == ident.lower()) | (func.lower(User.name) == ident.lower()))
        .first()
    )
    if not user or not _verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username/email or password.")

    return {
        "success": True,
        "message": "Login successful",
        "user_id": user.user_id,
        "username": user.name,
        "email": user.email,
        "user": {
            "user_id": user.user_id,
            "username": user.name,
            "email": user.email,
            "role": user.role,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Security Event & Fullscreen Integrity Recording
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/security-event")
def record_security_event(session_id: int, body: SecurityEventRequest, db: Session = Depends(get_db)):
    session = db.get(InterviewSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    meta = getattr(session, "meta_json", None) or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}

    violations = meta.get("security_violations", [])
    violations.append({
        "event_type": body.event_type,
        "description": body.description,
        "score_penalty": body.score_penalty,
    })
    meta["security_violations"] = violations
    tot_penalty = meta.get("total_security_penalty", 0.0) + body.score_penalty
    meta["total_security_penalty"] = tot_penalty

    # If final performance is present, update the score immediately
    if "final_performance" in meta and isinstance(meta["final_performance"], dict):
        base = meta["final_performance"].get("base_overall_score")
        if base is None:
            base = meta["final_performance"].get("overall_score", 0.0)
            meta["final_performance"]["base_overall_score"] = base
        meta["final_performance"]["overall_score"] = max(0.0, round(base - tot_penalty, 1))

    session.meta_json = dict(meta)
    flag_modified(session, "meta_json")
    db.commit()
    return {
        "success": True,
        "session_id": session_id,
        "event_type": body.event_type,
        "violation_count": len(violations),
        "total_penalty": tot_penalty
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dedicated Pass Question Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/pass")
def pass_question_endpoint(
    session_id: int,
    body: PassQuestionRequest | None = None,
    db: Session = Depends(get_db),
):
    session = db.get(InterviewSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    qid = body.question_id if body and body.question_id else None
    if not qid:
        all_ids = _session_question_ids(session)
        answered = _answered_question_ids(session_id, db)
        meta = getattr(session, "meta_json", None) or {}
        if isinstance(meta, str):
            try: meta = json.loads(meta)
            except Exception: meta = {}
        intro_qid = meta.get("intro_qid", 1)
        if intro_qid not in answered:
            qid = intro_qid
        else:
            remaining = [q for q in all_ids if q not in answered]
            if not remaining:
                return {"done": True, "message": "All questions answered"}
            qid = remaining[0]

    resp = Response(
        session_id=session_id,
        question_id=qid,
        audio_path=None,
    )
    db.add(resp)
    db.flush()

    t = Transcript(
        response_id=resp.response_id,
        transcript_text="(Candidate passed this question without answering)",
        word_timestamps=[],
        stt_model="none",
        confidence=1.0,
        source="passed",
    )
    db.add(t)

    dm = DeliveryMetrics(
        response_id=resp.response_id,
        speaking_rate=0.0,
        pause_count=0,
        long_pause_count=0,
        filler_count=0,
        filler_density=0.0,
        repetition_count=0,
        duration_sec=0,
        detail_json={"delivery_score": 0.0, "status": "passed_zero_score"},
    )
    db.add(dm)

    ce = ContentEvaluation(
        response_id=resp.response_id,
        correctness_score=0.0,
        coverage_score=0.0,
        relevance_score=0.0,
        completeness_score=0.0,
        star_score=0.0,
        overall_score=0.0,
        evidence_json={
            "verdict": "passed_zero_score",
            "reason": "Candidate explicitly passed question — evaluated as 0 score for delivery and content",
            "interviewer_reply": "No worries at all — let's move forward to the next question.",
        },
        used_delivery_features=False,
    )
    db.add(ce)
    db.commit()

    return {
        "success": True,
        "response_id": resp.response_id,
        "question_id": qid,
        "overall_score": 0.0,
        "verdict": "passed_zero_score",
        "interviewer_reply": "No worries at all — let's move forward to the next question.",
        "transcript": "(Candidate passed this question without answering)",
    }


@router.post("/sessions/{session_id}/link-user")
def link_session_user(session_id: int, body: LinkUserRequest, db: Session = Depends(get_db)):
    session = db.get(InterviewSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    user = db.get(User, body.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    session.user_id = user.user_id
    db.commit()
    return {"success": True, "session_id": session_id, "user_id": user.user_id, "username": user.name}


@router.get("/leaderboard")
def get_leaderboard(db: Session = Depends(get_db)):
    """Return student scores ranked on the placement leaderboard."""
    _seed_demo_leaderboard_if_empty(db)

    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id != None)
        .order_by(InterviewSession.started_at.desc())
        .all()
    )

    user_best: dict[int, dict] = {}
    for s in sessions:
        meta = getattr(s, "meta_json", None) or {}
        if isinstance(meta, str):
            try: meta = json.loads(meta)
            except: meta = {}

        perf = meta.get("final_performance")
        if not perf:
            continue

        uid = s.user_id
        score = perf.get("overall_score", 0)
        user = db.get(User, uid)
        if not user:
            continue

        if uid not in user_best or score > user_best[uid]["overall_score"]:
            user_best[uid] = {
                "user_id": uid,
                "username": user.name,
                "email": user.email,
                "overall_score": score,
                "content_score": perf.get("content_score", 0),
                "delivery_score": perf.get("delivery_score", 0),
                "code_score": perf.get("code_score"),
                "verdict": perf.get("verdict", "Evaluated"),
                "session_id": s.session_id,
                "mode": s.mode.value if hasattr(s.mode, "value") else str(s.mode),
                "completed_at": s.started_at.strftime("%Y-%m-%d %H:%M") if s.started_at else "Recently",
            }

    ranked = sorted(
        user_best.values(),
        key=lambda x: (x["overall_score"], x["code_score"] or 0),
        reverse=True,
    )

    for idx, item in enumerate(ranked):
        item["rank"] = idx + 1

    return ranked



