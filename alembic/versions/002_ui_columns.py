"""Add UI columns: interview_sessions.meta_json, responses.question_id

Revision ID: 002
Revises: 001
Create Date: 2026-11-08
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_sessions",
        sa.Column("meta_json", sa.JSON, nullable=True),
    )
    op.add_column(
        "responses",
        sa.Column(
            "question_id",
            sa.Integer,
            sa.ForeignKey("questions.question_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("responses", "question_id")
    op.drop_column("interview_sessions", "meta_json")
