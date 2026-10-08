"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2026-10-07
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # users
    op.create_table(
        "users",
        sa.Column("user_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("role", sa.String(50), server_default="student"),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # questions
    op.create_table(
        "questions",
        sa.Column("question_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("question_text", sa.Text, nullable=False),
        sa.Column(
            "category",
            sa.Enum("hr", "behavioral", "technical", name="questioncategory"),
            nullable=False,
        ),
        sa.Column("domain", sa.String(100)),
        sa.Column("difficulty", sa.String(50)),
        sa.Column("company_tag", sa.String(100)),
        sa.Column("active_flag", sa.Boolean, server_default="true"),
    )

    # rubrics
    op.create_table(
        "rubrics",
        sa.Column("rubric_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("question_id", sa.Integer, sa.ForeignKey("questions.question_id", ondelete="CASCADE"), nullable=False),
        sa.Column("criterion_name", sa.String(100), nullable=False),
        sa.Column(
            "criterion_type",
            sa.Enum("correctness", "relevance", "completeness", "concept_coverage", "star_structure", name="criteriontype"),
            nullable=False,
        ),
        sa.Column("weight", sa.Float, server_default="1.0"),
        sa.Column("expected_concepts", JSONB),
        sa.Column("star_component", sa.Boolean, server_default="false"),
        sa.Column("scoring_guide", sa.Text),
    )

    # interview_sessions
    op.create_table(
        "interview_sessions",
        sa.Column("session_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.user_id", ondelete="SET NULL")),
        sa.Column("question_id", sa.Integer, sa.ForeignKey("questions.question_id", ondelete="SET NULL")),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column(
            "mode",
            sa.Enum("practice", "audit", "benchmark", name="sessionmode"),
            server_default="practice",
        ),
        sa.Column(
            "status",
            sa.Enum("active", "completed", "aborted", name="sessionstatus"),
            server_default="active",
        ),
    )

    # responses
    op.create_table(
        "responses",
        sa.Column("response_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.Integer, sa.ForeignKey("interview_sessions.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("audio_path", sa.String(255)),
        sa.Column("duration_sec", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # llm_call_logs (created before content_evaluations that FK into it)
    op.create_table(
        "llm_call_logs",
        sa.Column("log_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("prompt_version", sa.String(50), nullable=False),
        sa.Column("call_type", sa.String(50), nullable=False),
        sa.Column("prompt_text", sa.Text, nullable=False),
        sa.Column("raw_output", sa.Text, nullable=False),
        sa.Column("parsed_ok", sa.Boolean, server_default="true"),
        sa.Column("temperature", sa.Float, server_default="0.0"),
        sa.Column("input_tokens", sa.Integer),
        sa.Column("output_tokens", sa.Integer),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # transcripts
    op.create_table(
        "transcripts",
        sa.Column("transcript_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("response_id", sa.Integer, sa.ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False),
        sa.Column("transcript_text", sa.Text, nullable=False),
        sa.Column("word_timestamps", JSONB),
        sa.Column("stt_model", sa.String(100)),
        sa.Column("confidence", sa.Float),
        sa.Column("source", sa.String(50), server_default="whisper"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # content_evaluations
    op.create_table(
        "content_evaluations",
        sa.Column("content_eval_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("response_id", sa.Integer, sa.ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False),
        sa.Column("correctness_score", sa.Float),
        sa.Column("coverage_score", sa.Float),
        sa.Column("relevance_score", sa.Float),
        sa.Column("completeness_score", sa.Float),
        sa.Column("star_score", sa.Float),
        sa.Column("overall_score", sa.Float),
        sa.Column("evidence_json", JSONB),
        sa.Column("used_delivery_features", sa.Boolean, server_default="false"),
        sa.Column("llm_call_log_id", sa.Integer, sa.ForeignKey("llm_call_logs.log_id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # delivery_metrics
    op.create_table(
        "delivery_metrics",
        sa.Column("delivery_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("response_id", sa.Integer, sa.ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False),
        sa.Column("speaking_rate", sa.Float),
        sa.Column("pause_count", sa.Integer),
        sa.Column("long_pause_count", sa.Integer),
        sa.Column("filler_count", sa.Integer),
        sa.Column("filler_density", sa.Float),
        sa.Column("repetition_count", sa.Integer),
        sa.Column("duration_sec", sa.Integer),
        sa.Column("detail_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # audit_results
    op.create_table(
        "audit_results",
        sa.Column("audit_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("response_id", sa.Integer, sa.ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False),
        sa.Column("delivery_invariance_error", sa.Float),
        sa.Column("content_sensitivity", sa.Float),
        sa.Column("paraphrase_deviation", sa.Float),
        sa.Column("repeatability_variance", sa.Float),
        sa.Column("stt_sensitivity", sa.Float),
        sa.Column("uncertainty_flag", sa.Boolean, server_default="false"),
        sa.Column("details_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # feedback_reports
    op.create_table(
        "feedback_reports",
        sa.Column("report_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("response_id", sa.Integer, sa.ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False),
        sa.Column("summary_text", sa.Text),
        sa.Column("strengths", sa.Text),
        sa.Column("improvements", sa.Text),
        sa.Column("criterion_feedback_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # progress_history
    op.create_table(
        "progress_history",
        sa.Column("progress_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.Integer, sa.ForeignKey("interview_sessions.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("trend_metric", sa.String(100), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # human_annotations
    op.create_table(
        "human_annotations",
        sa.Column("annotation_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("annotator_id", sa.Integer, sa.ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
        sa.Column("response_id", sa.Integer, sa.ForeignKey("responses.response_id", ondelete="CASCADE"), nullable=False),
        sa.Column("criterion", sa.String(100), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("human_annotations")
    op.drop_table("progress_history")
    op.drop_table("feedback_reports")
    op.drop_table("audit_results")
    op.drop_table("delivery_metrics")
    op.drop_table("content_evaluations")
    op.drop_table("transcripts")
    op.drop_table("llm_call_logs")
    op.drop_table("responses")
    op.drop_table("interview_sessions")
    op.drop_table("rubrics")
    op.drop_table("questions")
    op.drop_table("users")

    op.execute("DROP TYPE IF EXISTS questioncategory")
    op.execute("DROP TYPE IF EXISTS criteriontype")
    op.execute("DROP TYPE IF EXISTS sessionmode")
    op.execute("DROP TYPE IF EXISTS sessionstatus")
