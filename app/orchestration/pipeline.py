"""Full evaluation pipeline for a submitted audio response.

Order of operations:
  1. STT  → Transcript row
  2. Delivery Engine  → DeliveryMetrics row
  3. Content Engine   → ContentEvaluation row
  4. Audit Layer      → AuditResult row
  5. Feedback Report  → FeedbackReport row

ISOLATION ENFORCED:
  - Content Engine receives transcript TEXT only (no timestamps, no DeliveryResult)
  - Delivery Engine receives word timestamps only (no text passed to its analyze())
  - The two engines never share a data structure
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.audit.combiner import AuditCombiner
from app.audit.tests.delivery_invariance import DeliveryInvarianceTest
from app.audit.tests.paraphrase_stability import ParaphraseStabilityTest
from app.audit.tests.repeated_run import RepeatedRunTest
from app.audit.variants import VariantGenerator
from app.config import get_settings
from app.engines.content import ContentEngine, RubricCriterion
from app.engines.delivery import DeliveryEngine
from app.models.core import (
    AuditResult,
    ContentEvaluation,
    DeliveryMetrics,
    FeedbackReport,
    InterviewSession,
    Response,
    Rubric,
    Transcript,
)
from app.services.llm import get_llm_service
from app.services.stt import get_stt_service


def run_evaluation_pipeline(response_id: int, db: Session) -> None:
    """Execute the full evaluation pipeline for a single response."""
    settings = get_settings()

    # ── Fetch response + session + question ───────────────────────────────────
    response: Response = db.get(Response, response_id)
    if response is None:
        raise ValueError(f"Response {response_id} not found")

    session: InterviewSession = db.get(InterviewSession, response.session_id)
    question_id = session.question_id if session else None
    question_text = ""
    rubric_criteria: list[RubricCriterion] = []

    if question_id is not None:
        from app.models.core import Question
        question_obj = db.get(Question, question_id)
        if question_obj:
            question_text = question_obj.question_text
            rubrics: list[Rubric] = (
                db.query(Rubric).filter(Rubric.question_id == question_id).all()
            )
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

    # ── 1. STT ────────────────────────────────────────────────────────────────
    stt_service = get_stt_service()
    stt_result = stt_service.transcribe(response.audio_path or "")

    transcript = Transcript(
        response_id=response_id,
        transcript_text=stt_result.text,
        word_timestamps=stt_result.to_word_timestamps_json(),
        stt_model=stt_result.model_name,
        confidence=stt_result.avg_confidence,
        source="whisper" if settings.llm_provider != "mock" else "mock",
    )
    db.add(transcript)
    db.flush()

    # ── 2. Delivery Engine (timestamps only — no text) ────────────────────────
    delivery_engine = DeliveryEngine()
    delivery_result = delivery_engine.analyze(stt_result.words)   # ← only words

    dm = DeliveryMetrics(
        response_id=response_id,
        speaking_rate=delivery_result.speaking_rate_wpm,
        pause_count=delivery_result.pause_count,
        long_pause_count=delivery_result.long_pause_count,
        filler_count=delivery_result.filler_count,
        filler_density=delivery_result.filler_density,
        repetition_count=delivery_result.repetition_count,
        duration_sec=int(delivery_result.duration_sec),
        detail_json=delivery_result.to_dict()["detail"],
    )
    db.add(dm)
    db.flush()

    # ── 3. Content Engine (transcript TEXT only — no timestamps) ──────────────
    llm = get_llm_service()
    content_engine = ContentEngine(llm=llm, db=db)

    # Ablation: optionally pass delivery features
    delivery_features_for_ablation = (
        {
            "speaking_rate_wpm": delivery_result.speaking_rate_wpm,
            "filler_density": delivery_result.filler_density,
            "pause_count": delivery_result.pause_count,
        }
        if settings.content_engine_include_delivery
        else None
    )

    content_result = content_engine.analyze(
        transcript_text=stt_result.text,   # ← text only
        question_text=question_text,
        rubric_criteria=rubric_criteria,
        delivery_features=delivery_features_for_ablation,
    )

    ce = ContentEvaluation(
        response_id=response_id,
        correctness_score=content_result.correctness_score,
        coverage_score=content_result.coverage_score,
        relevance_score=content_result.relevance_score,
        completeness_score=content_result.completeness_score,
        star_score=content_result.star_score,
        overall_score=content_result.overall_score,
        evidence_json=content_result.evidence_json,
        used_delivery_features=content_result.used_delivery_features,
        llm_call_log_id=content_result.llm_call_log_id,
    )
    db.add(ce)
    db.flush()

    # ── 4. Audit Layer ────────────────────────────────────────────────────────
    audit_metrics = []
    generator = VariantGenerator(seed=42)

    # Test 1 — Delivery Invariance (filler + pace variants)
    try:
        filler_v = generator.filler_injection(stt_result.words)
        pace_v = generator.pace_change(stt_result.words, factor=1.5)
        di_test = DeliveryInvarianceTest(content_engine)
        di_metric = di_test.run(
            original_transcript=stt_result.text,
            question_text=question_text,
            rubric_criteria=rubric_criteria,
            delivery_variants=[filler_v, pace_v],
        )
        audit_metrics.append(di_metric)
    except Exception:
        pass

    # Test 3 — Paraphrase Stability
    try:
        para_v = generator.paraphrase(stt_result.text, stt_result.words, llm_service=llm)
        ps_test = ParaphraseStabilityTest(content_engine)
        ps_metric = ps_test.run(
            original_transcript=stt_result.text,
            question_text=question_text,
            rubric_criteria=rubric_criteria,
            paraphrase_variants=[para_v],
        )
        audit_metrics.append(ps_metric)
    except Exception:
        pass

    # Test 4 — Repeated Run
    try:
        rr_test = RepeatedRunTest(content_engine, n_runs=settings.audit_repeated_runs)
        rr_metric = rr_test.run(
            transcript_text=stt_result.text,
            question_text=question_text,
            rubric_criteria=rubric_criteria,
        )
        audit_metrics.append(rr_metric)
    except Exception:
        pass

    audit_report = AuditCombiner().combine(audit_metrics)

    ar = AuditResult(
        response_id=response_id,
        delivery_invariance_error=audit_report.delivery_invariance_error,
        content_sensitivity=audit_report.content_sensitivity,
        paraphrase_deviation=audit_report.paraphrase_deviation,
        repeatability_variance=audit_report.repeatability_variance,
        stt_sensitivity=audit_report.stt_sensitivity,
        uncertainty_flag=audit_report.uncertainty_flag,
        details_json=audit_report.details,
    )
    db.add(ar)
    db.flush()

    # ── 5. Feedback Report ────────────────────────────────────────────────────
    strengths, improvements = _build_feedback_text(content_result, delivery_result)

    fr = FeedbackReport(
        response_id=response_id,
        summary_text=(
            f"Overall content score: {content_result.overall_score:.2f}. "
            f"Speaking rate: {delivery_result.speaking_rate_wpm:.0f} WPM. "
            f"{'Evaluation flagged as uncertain.' if audit_report.uncertainty_flag else 'Evaluation is stable.'}"
        ),
        strengths=strengths,
        improvements=improvements,
        criterion_feedback_json=content_result.evidence_json,
    )
    db.add(fr)
    db.commit()


def _build_feedback_text(content_result, delivery_result) -> tuple[str, str]:
    """Build simple textual feedback from evaluation results."""
    strengths: list[str] = []
    improvements: list[str] = []

    if content_result.overall_score and content_result.overall_score >= 0.7:
        strengths.append("Strong overall content quality.")
    else:
        improvements.append("Improve answer completeness and technical depth.")

    if delivery_result.filler_density and delivery_result.filler_density < 5:
        strengths.append("Low filler word usage — clear delivery.")
    elif delivery_result.filler_density and delivery_result.filler_density >= 10:
        improvements.append("Reduce filler words (um, uh, like) for more polished delivery.")

    wpm = delivery_result.speaking_rate_wpm
    if 120 <= wpm <= 160:
        strengths.append(f"Good speaking pace ({wpm:.0f} WPM).")
    elif wpm < 100:
        improvements.append(f"Speaking pace is slow ({wpm:.0f} WPM). Aim for 120–160 WPM.")
    elif wpm > 180:
        improvements.append(f"Speaking pace is fast ({wpm:.0f} WPM). Slow down slightly.")

    if delivery_result.long_pause_count and delivery_result.long_pause_count > 3:
        improvements.append("Reduce long pauses; they disrupt fluency.")

    return "; ".join(strengths), "; ".join(improvements)
