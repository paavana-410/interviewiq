"""POST /responses/{id}/annotations — submit a human annotation for a response criterion."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.core import HumanAnnotation, Response, User

router = APIRouter(tags=["annotations"])


class AnnotationCreate(BaseModel):
    annotator_id: int
    criterion: str
    score: float = Field(ge=0.0, le=1.0)
    notes: str | None = None


class AnnotationResponse(BaseModel):
    annotation_id: int
    response_id: int
    annotator_id: int
    criterion: str
    score: float

    model_config = {"from_attributes": True}


@router.post("/responses/{response_id}/annotations", response_model=AnnotationResponse, status_code=201)
def create_annotation(
    response_id: int,
    body: AnnotationCreate,
    db: Session = Depends(get_db),
):
    response = db.get(Response, response_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Response not found")

    annotation = HumanAnnotation(
        annotator_id=body.annotator_id,
        response_id=response_id,
        criterion=body.criterion,
        score=body.score,
        notes=body.notes,
    )
    db.add(annotation)
    db.commit()
    db.refresh(annotation)
    return annotation
