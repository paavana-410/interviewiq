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

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
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


def _build_interview_flow(db: Session, mode: str, difficulty: str = "easy", seed: int = 42) -> list[Question]:
    categories = _MODE_MAP.get(mode, list(QuestionCategory))

    # 1. Base questions (exclude Question 1 which is reserved for candidate introduction)
    base_qs = (
        db.query(Question)
        .filter(
            Question.category.in_(categories),
            Question.active_flag == True,
            Question.question_id != 1,
            (Question.company_tag == None) | (~Question.company_tag.like("followup_for_%"))
        )
        .all()
    )
    if not base_qs:
        base_qs = db.query(Question).filter(Question.active_flag == True, Question.question_id != 1).all()

    rng = random.Random(seed)

    # Prioritize base questions matching the selected difficulty
    matching = [q for q in base_qs if (q.difficulty or "").lower() == difficulty.lower()]
    other = [q for q in base_qs if q not in matching]
    rng.shuffle(matching)
    rng.shuffle(other)
    candidate_bases = matching + other

    def get_followups(base_qid: int) -> list[Question]:
        return (
            db.query(Question)
            .filter(
                Question.company_tag == f"followup_for_{base_qid}",
                Question.active_flag == True
            )
            .all()
        )

    flow: list[Question] = []
    diff = (difficulty or "easy").lower()

    if diff == "easy":
        # Target: 5 questions -> 1q + fol + 1q + fol + fol = 5q
        bases_with_fol = [b for b in candidate_bases if get_followups(b.question_id)]
        bases_without = [b for b in candidate_bases if b not in bases_with_fol]
        ordered_bases = bases_with_fol + bases_without

        if len(ordered_bases) >= 2:
            b1 = ordered_bases[0]
            b2 = ordered_bases[1]
            f1 = get_followups(b1.question_id)
            f2 = get_followups(b2.question_id)

            flow.append(b1)
            if f1: flow.append(f1[0])
            flow.append(b2)
            if f2: flow.append(f2[0])
            if len(f2) > 1:
                flow.append(f2[1])
            elif len(f1) > 1:
                flow.append(f1[1])
            elif len(ordered_bases) > 2:
                flow.append(ordered_bases[2])
        else:
            flow.extend(ordered_bases[:5])

    elif diff == "medium":
        # Target: 6 questions -> 1q + fol + 1q + fol + 1q + fol = 6q
        bases_with_fol = [b for b in candidate_bases if get_followups(b.question_id)]
        bases_without = [b for b in candidate_bases if b not in bases_with_fol]
        ordered_bases = bases_with_fol + bases_without

        for b in ordered_bases:
            if len(flow) >= 6: break
            flow.append(b)
            f = get_followups(b.question_id)
            if f and len(flow) < 6:
                flow.append(f[0])

    else:  # "hard"
        # Target: 7 questions -> 1q + fol + fol + 1q + fol + fol + 1q = 7q
        bases_with_fol = [b for b in candidate_bases if get_followups(b.question_id)]
        bases_without = [b for b in candidate_bases if b not in bases_with_fol]
        ordered_bases = bases_with_fol + bases_without

        for b in ordered_bases:
            if len(flow) >= 7: break
            flow.append(b)
            f = get_followups(b.question_id)
            for fol in f:
                if len(flow) >= 7: break
                flow.append(fol)

    # Ensure minimum 5 questions if pool permits
    if len(flow) < 5:
        all_qs = db.query(Question).filter(Question.active_flag == True).all()
        rng.shuffle(all_qs)
        for q in all_qs:
            if q not in flow:
                flow.append(q)
            if len(flow) >= 5:
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


# ─────────────────────────────────────────────────────────────────────────────
# Session creation
# ─────────────────────────────────────────────────────────────────────────────

class InterviewSessionCreate(BaseModel):
    mode: str = "mixed"          # behavioral | technical | mixed
    difficulty: str = "easy"     # easy | medium | hard
    n_questions: int | None = None


class InterviewSessionResponse(BaseModel):
    session_id: int
    mode: str
    difficulty: str
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
    overall_agg = round((content_score * 0.6) + (delivery_score * 0.4), 1)

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
# Coding & SQL IDE Endpoints (CS Student Practice & Technical Assessments)
# ─────────────────────────────────────────────────────────────────────────────

class CodeRunRequest(BaseModel):
    challenge_id: str
    language: str
    source_code: str
    session_id: int | None = None


class CodeSubmitRequest(BaseModel):
    session_id: int
    challenge_id: str
    language: str
    source_code: str


@router.get("/code/challenges")
def list_code_challenges():
    from app.services.code_executor import CHALLENGES, get_challenge_metadata
    return [get_challenge_metadata(cid) for cid in CHALLENGES]


@router.get("/code/challenges/{challenge_id}")
def get_code_challenge(challenge_id: str):
    from app.services.code_executor import get_challenge_metadata
    try:
        return get_challenge_metadata(challenge_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/code/run")
def run_code(body: CodeRunRequest, db: Session = Depends(get_db)):
    from app.services.code_executor import execute_challenge

    if body.session_id is not None:
        session = db.get(InterviewSession, body.session_id)
        if session:
            meta = getattr(session, "meta_json", None) or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            runs = meta.get("code_run_count", 0)
            if runs >= 15:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit reached: Maximum 15 test runs per session reached. Please review your solution and Submit."
                )
            meta["code_run_count"] = runs + 1
            session.meta_json = meta
            db.commit()

    try:
        return execute_challenge(
            challenge_id=body.challenge_id,
            language=body.language,
            source_code=body.source_code,
            mode="run",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/code/submit")
def submit_code(body: CodeSubmitRequest, db: Session = Depends(get_db)):
    from app.services.code_executor import execute_challenge
    session = db.get(InterviewSession, body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    try:
        res = execute_challenge(
            challenge_id=body.challenge_id,
            language=body.language,
            source_code=body.source_code,
            mode="submit",
        )
        meta = getattr(session, "meta_json", None) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        code_submissions = meta.get("code_submissions", [])
        code_submissions.append({
            "challenge_id": body.challenge_id,
            "language": body.language,
            "score_percent": res["score_percent"],
            "all_passed": res["all_passed"],
            "passed_tests": res["passed_tests"],
            "total_tests": res["total_tests"],
            "source_code": body.source_code,
        })
        meta["code_submissions"] = code_submissions
        session.meta_json = meta
        db.commit()
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
