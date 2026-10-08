"""POST /sessions — create a new interview session."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.schemas import SessionCreate, SessionResponse
from app.database import get_db
from app.models.core import InterviewSession, SessionMode, SessionStatus

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=201)
def create_session(body: SessionCreate, db: Session = Depends(get_db)):
    try:
        mode = SessionMode(body.mode)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid mode '{body.mode}'")

    session = InterviewSession(
        user_id=body.user_id,
        question_id=body.question_id,
        mode=mode,
        status=SessionStatus.ACTIVE,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session
