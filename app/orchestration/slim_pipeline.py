"""Slim evaluation pipeline for the UI — STT + Delivery + Content only.

Audits are skipped by default (AUDIT_INLINE=false).

Rules enforced here:
  1. UI_JUDGE setting: in real mode, the pipeline uses exactly the judge
     named in UI_JUDGE. If unset or the judge isn't configured, raises
     UIJudgeNotConfigured so the app can return a 503 with a clear message.

  2. Whisper guard: in real mode, if faster-whisper is not installed,
     raises WhisperUnavailable instead of returning a synthetic transcript.

  3. Mock synthetic transcript: per-question, not a single constant string.
     The text carries the question_id so the UI can label it as synthetic.
     transcript_source == "synthetic" on SlimResult in mock mode.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import get_settings
from app.engines.content import ContentEngine, RubricCriterion
from app.engines.delivery import DeliveryEngine
from app.models.core import (
    ContentEvaluation,
    DeliveryMetrics,
    InterviewSession,
    Question,
    Response,
    Rubric,
    Transcript,
)


# ─────────────────────────────────────────────────────────────────────────────
# Typed errors the route layer converts to HTTP responses
# ─────────────────────────────────────────────────────────────────────────────

class UIJudgeNotConfigured(Exception):
    """Raised when UI_JUDGE is unset or names a judge that doesn't exist."""


class WhisperUnavailable(Exception):
    """Raised in real mode when faster-whisper is not installed."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _whisper_installed() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _get_ui_llm_service():
    """Return the LLMService for the UI judge.

    In mock mode: returns MockLLMService (UI_JUDGE ignored).
    In real mode: looks up UI_JUDGE by name in the registry.
    Raises UIJudgeNotConfigured if the setting is missing or the judge
    is not in the registry.
    """
    from app.config import get_settings
    s = get_settings()

    if s.llm_provider == "mock":
        from app.services.llm import MockLLMService
        return MockLLMService()

    if not s.ui_judge:
        raise UIJudgeNotConfigured(
            "UI_JUDGE is not set in .env. "
            "Add UI_JUDGE=<judge_name> where <judge_name> matches one of your JUDGE_N_NAME values."
        )

    from app.services.judge_config import get_judge_registry
    from app.services.llm import OpenAICompatibleService, _cache_dir

    registry = get_judge_registry()
    try:
        cfg = registry.get_judge(s.ui_judge)
    except KeyError:
        available = registry.judge_names()
        raise UIJudgeNotConfigured(
            f"UI_JUDGE={s.ui_judge!r} is not a configured judge. "
            f"Available judges: {available}. "
            f"Check your .env JUDGE_N_NAME values."
        )

    return OpenAICompatibleService(cfg, cache_dir=_cache_dir())


def _synthetic_transcript(question_id: int | None, question_text: str) -> str:
    """Return a per-question synthetic transcript for mock/demo mode.

    The text is different for each question so the UI can show it meaningfully.
    The question_id is mixed in via a hash so it's deterministic but distinct.
    Prefixed with [SYNTHETIC] so the UI can detect and label it.
    """
    seed_str = f"{question_id}:{question_text[:30]}"
    h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)

    templates = [
        "I have experience with {topic} and have worked on several projects. "
        "I believe my background makes me a strong candidate for this role. "
        "I always try to apply best practices and learn from each challenge.",

        "In my previous work I dealt with {topic} quite often. "
        "I collaborated with my team to deliver results on time. "
        "I think communication and technical skills are equally important.",

        "I understand {topic} from both a theoretical and practical perspective. "
        "I have implemented solutions in real-world settings. "
        "I am confident in my ability to contribute effectively.",

        "My approach to {topic} involves careful planning and clear documentation. "
        "I have led small teams and mentored junior colleagues. "
        "I always focus on quality and maintainability.",

        "I first encountered {topic} during my studies and later in internships. "
        "I enjoy solving complex problems and explaining them clearly. "
        "I am eager to keep learning and growing in this field.",
    ]

    keywords = ["this area", "technical concepts", "problem solving", "software development", "data structures"]
    template = templates[h % len(templates)]
    topic = keywords[(h // len(templates)) % len(keywords)]
    text = template.format(topic=topic)
    return f"[SYNTHETIC TRANSCRIPT (demo)] {text}"


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SlimResult:
    response_id: int
    question_text: str
    transcript: str
    transcript_source: str     # "whisper" | "synthetic" | "mock"
    # Delivery
    wpm: float
    pause_count: int
    long_pause_count: int
    filler_count: int
    filler_density: float
    repetition_count: int
    duration_sec: float
    # Content
    overall_score: float
    correctness_score: float | None
    coverage_score: float | None
    relevance_score: float | None
    completeness_score: float | None
    star_score: float | None
    evidence_json: dict
    uncertainty: bool
    # Judge info (safe — no key)
    judge_model: str
    judge_provider: str
    is_mock: bool


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_slim_pipeline(response_id: int, db: Session) -> SlimResult:
    """Run STT → Delivery → Content for one response. No audit."""
    settings = get_settings()
    is_mock = settings.llm_provider == "mock"

    response: Response | None = db.get(Response, response_id)
    if response is None:
        raise ValueError(f"Response {response_id} not found")

    # Resolve question
    session: InterviewSession | None = db.get(InterviewSession, response.session_id)
    question_id = response.question_id or (session.question_id if session else None)
    question_text = ""
    rubric_criteria: list[RubricCriterion] = []

    if question_id is not None:
        q: Question | None = db.get(Question, question_id)
        if q:
            question_text = q.question_text
            rubrics = db.query(Rubric).filter(Rubric.question_id == question_id).all()
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

    # 1. Get the LLM (validates UI_JUDGE in real mode)
    llm = _get_ui_llm_service()

    # 2. STT
    transcript_source: str
    if is_mock:
        # Per-question synthetic transcript — different text per question
        transcript_text = _synthetic_transcript(question_id, question_text)
        transcript_source = "synthetic"
        from app.services.stt import MockSTTService
        stt_result = MockSTTService(fixed_transcript=transcript_text).transcribe(
            response.audio_path or ""
        )
    else:
        # Real mode: require faster-whisper
        if not _whisper_installed():
            raise WhisperUnavailable(
                "Speech recognition unavailable: faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            )
        from app.services.stt import WhisperSTTService
        stt = WhisperSTTService(
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        stt_result = stt.transcribe(response.audio_path or "")
        transcript_source = "whisper"

    # Persist transcript
    t = Transcript(
        response_id=response_id,
        transcript_text=stt_result.text,
        word_timestamps=stt_result.to_word_timestamps_json(),
        stt_model=stt_result.model_name,
        confidence=stt_result.avg_confidence,
        source=transcript_source,
    )
    db.add(t)
    db.flush()

    # 3. Delivery (timestamps only)
    delivery_engine = DeliveryEngine()
    dr = delivery_engine.analyze(stt_result.words)

    dm = DeliveryMetrics(
        response_id=response_id,
        speaking_rate=dr.speaking_rate_wpm,
        pause_count=dr.pause_count,
        long_pause_count=dr.long_pause_count,
        filler_count=dr.filler_count,
        filler_density=dr.filler_density,
        repetition_count=dr.repetition_count,
        duration_sec=int(dr.duration_sec),
        detail_json=dr.to_dict()["detail"],
    )
    db.add(dm)
    db.flush()

    # 4. Content (text only)
    ce_engine = ContentEngine(llm=llm, db=db)
    cr = ce_engine.analyze(
        transcript_text=stt_result.text,
        question_text=question_text,
        rubric_criteria=rubric_criteria,
    )

    ce = ContentEvaluation(
        response_id=response_id,
        correctness_score=cr.correctness_score,
        coverage_score=cr.coverage_score,
        relevance_score=cr.relevance_score,
        completeness_score=cr.completeness_score,
        star_score=cr.star_score,
        overall_score=cr.overall_score,
        evidence_json=cr.evidence_json,
        used_delivery_features=cr.used_delivery_features,
        llm_call_log_id=cr.llm_call_log_id,
    )
    db.add(ce)
    db.commit()

    return SlimResult(
        response_id=response_id,
        question_text=question_text,
        transcript=stt_result.text,
        transcript_source=transcript_source,
        wpm=dr.speaking_rate_wpm,
        pause_count=dr.pause_count,
        long_pause_count=dr.long_pause_count,
        filler_count=dr.filler_count,
        filler_density=dr.filler_density,
        repetition_count=dr.repetition_count,
        duration_sec=dr.duration_sec,
        overall_score=cr.overall_score,
        correctness_score=cr.correctness_score,
        coverage_score=cr.coverage_score,
        relevance_score=cr.relevance_score,
        completeness_score=cr.completeness_score,
        star_score=cr.star_score,
        evidence_json=cr.evidence_json,
        uncertainty=cr.uncertainty,
        judge_model=llm.model_name,
        judge_provider=llm.provider,
        is_mock=is_mock,
    )
