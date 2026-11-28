"""Admin & Faculty Portal API Routes for InterviewIQ.

Provides endpoints for:
1. Ingesting faculty question documents (Technical & Behavioral).
2. Monitoring student performance & readiness scores.
3. Exporting student results to Excel / CSV.
4. Serving flashcards and study material for the tutoring hub.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response as FastAPIResponse, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.core import (
    ContentEvaluation,
    DeliveryMetrics,
    InterviewSession,
    Question,
    QuestionCategory,
    Response,
    Rubric,
)
from app.services.doc_ingestion import ingest_questions_from_text

router = APIRouter(prefix="/admin", tags=["admin"])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Document Upload & Ingestion
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload-questions")
async def upload_questions(
    category: str = Form("technical"),
    domain: str = Form("General"),
    difficulty: str = Form("medium"),
    text_content: str = Form(""),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    """Ingest questions and rubrics from an uploaded document (TXT, CSV, PDF) or pasted text."""
    content = text_content or ""

    if file and file.filename:
        raw_bytes = await file.read()
        try:
            # Try utf-8 decoding first
            content = raw_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not content.strip():
        raise HTTPException(status_code=400, detail="No question content provided. Please upload a file or paste text.")

    res = ingest_questions_from_text(
        db=db,
        content=content,
        category=category,
        domain_override=domain if domain != "General" else None,
        default_difficulty=difficulty,
    )

    if not res["success"]:
        raise HTTPException(status_code=422, detail=res["message"])

    return res


# ─────────────────────────────────────────────────────────────────────────────
# 2. Database Stats & Question Bank Summary
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stats")
def get_admin_stats(db: Session = Depends(get_db)):
    """Return counts of questions by category & difficulty, plus student session totals."""
    tech_count = db.query(Question).filter(Question.category == QuestionCategory.TECHNICAL, Question.active_flag == True).count()
    beh_count = db.query(Question).filter(Question.category == QuestionCategory.BEHAVIORAL, Question.active_flag == True).count()
    hr_count = db.query(Question).filter(Question.category == QuestionCategory.HR, Question.active_flag == True).count()

    easy_count = db.query(Question).filter(func.lower(Question.difficulty) == "easy", Question.active_flag == True).count()
    med_count = db.query(Question).filter(func.lower(Question.difficulty) == "medium", Question.active_flag == True).count()
    hard_count = db.query(Question).filter(func.lower(Question.difficulty) == "hard", Question.active_flag == True).count()

    total_sessions = db.query(InterviewSession).count()

    return {
        "questions": {
            "total": tech_count + beh_count + hr_count,
            "technical": tech_count,
            "behavioral": beh_count,
            "hr": hr_count,
            "easy": easy_count,
            "medium": med_count,
            "hard": hard_count,
        },
        "total_sessions": total_sessions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Student Performance List & Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/students")
def list_student_sessions(db: Session = Depends(get_db)):
    """List completed interview sessions with student scores, WPM, and proctoring status."""
    sessions = (
        db.query(InterviewSession)
        .order_by(InterviewSession.started_at.desc())
        .limit(100)
        .all()
    )

    results = []
    for s in sessions:
        meta = s.meta_json if isinstance(s.meta_json, dict) else {}
        candidate_name = meta.get("candidate_name") or f"Student #{s.session_id}"
        track = meta.get("track") or meta.get("mode") or "Mixed"
        diff = meta.get("difficulty") or "Medium"

        # Aggregate scores from responses
        responses = db.query(Response).filter(Response.session_id == s.session_id).all()
        q_count = len(responses)

        content_scores = []
        delivery_scores = []
        wpms = []
        fillers = 0

        for r in responses:
            ce = db.query(ContentEvaluation).filter(ContentEvaluation.response_id == r.response_id).first()
            if ce and ce.overall_score is not None:
                content_scores.append(ce.overall_score * 100.0)

            dm = db.query(DeliveryMetrics).filter(DeliveryMetrics.response_id == r.response_id).first()
            if dm:
                if dm.speaking_rate:
                    wpms.append(dm.speaking_rate)
                fillers += (dm.filler_count or 0)
                # Compute delivery score (100 - penalties)
                score_del = max(0.0, 100.0 - ((dm.filler_count or 0) * 4.0))
                delivery_scores.append(score_del)

        avg_content = round(sum(content_scores) / len(content_scores), 1) if content_scores else 0.0
        avg_delivery = round(sum(delivery_scores) / len(delivery_scores), 1) if delivery_scores else 0.0
        overall = round((avg_content * 0.7) + (avg_delivery * 0.3), 1) if (avg_content or avg_delivery) else 0.0
        avg_wpm = round(sum(wpms) / len(wpms), 0) if wpms else 0

        # Proctor violations from session metadata
        sec_violations = meta.get("security_violations", []) if isinstance(meta, dict) else []
        p_count = len(sec_violations)
        proctor_status = "Flagged" if p_count > 0 else "Clean"

        results.append({
            "session_id": s.session_id,
            "candidate_name": candidate_name,
            "track": track.capitalize(),
            "difficulty": diff.capitalize(),
            "questions_answered": q_count,
            "overall_score": overall,
            "content_score": avg_content,
            "delivery_score": avg_delivery,
            "avg_wpm": int(avg_wpm),
            "total_fillers": fillers,
            "proctor_violations": p_count,
            "proctor_status": proctor_status,
            "date": s.started_at.strftime("%Y-%m-%d %H:%M") if s.started_at else "—",
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Export to Excel / CSV
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/export-csv")
def export_results_csv(db: Session = Depends(get_db)):
    """Export all student performance data directly as a downloadable CSV for Excel."""
    students = list_student_sessions(db=db)

    output = io.StringIO()
    writer = csv.writer(output)

    # Write Header Row
    writer.writerow([
        "Session ID",
        "Candidate Name",
        "Interview Track",
        "Difficulty",
        "Questions Answered",
        "Overall Score (%)",
        "Content Score (%)",
        "Delivery Score (%)",
        "Avg Pace (WPM)",
        "Total Fillers",
        "Proctor Violations",
        "Proctor Status",
        "Date & Time",
    ])

    for st in students:
        writer.writerow([
            st["session_id"],
            st["candidate_name"],
            st["track"],
            st["difficulty"],
            st["questions_answered"],
            st["overall_score"],
            st["content_score"],
            st["delivery_score"],
            st["avg_wpm"],
            st["total_fillers"],
            st["proctor_violations"],
            st["proctor_status"],
            st["date"],
        ])

    output.seek(0)
    filename = f"InterviewIQ_Student_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Flashcards & Study Prep Hub Data
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/flashcards")
def get_flashcards(category: str = "all", db: Session = Depends(get_db)):
    """Return flashcards mapped from verified question rubrics for student tutoring."""
    query = db.query(Question).filter(Question.active_flag == True)
    if category.lower() in ("technical", "behavioral"):
        cat_enum = QuestionCategory.TECHNICAL if category.lower() == "technical" else QuestionCategory.BEHAVIORAL
        query = query.filter(Question.category == cat_enum)

    questions = query.limit(60).all()
    cards = []

    for q in questions:
        rubrics = db.query(Rubric).filter(Rubric.question_id == q.question_id).all()
        expected = []
        guide = ""
        for r in rubrics:
            if r.expected_concepts:
                expected.extend(r.expected_concepts if isinstance(r.expected_concepts, list) else [])
            if r.scoring_guide and not guide:
                guide = r.scoring_guide

        cards.append({
            "id": q.question_id,
            "category": q.category.value.capitalize(),
            "domain": q.domain or "Core Engineering",
            "difficulty": (q.difficulty or "Medium").capitalize(),
            "question": q.question_text,
            "key_concepts": expected[:5],
            "study_notes": guide or f"Review key terminology and architectural trade-offs for {q.domain or 'this topic'}.",
        })

    return cards
