"""Content Engine tests — mocked LLM."""
from __future__ import annotations

import json

import pytest

from app.engines.content import ContentEngine, RubricCriterion
from app.services.llm import MockLLMService, LLMResponse


def test_content_engine_returns_valid_result(content_engine, rubric_criteria):
    result = content_engine.analyze(
        transcript_text="I have strong OOP experience and know design patterns.",
        question_text="Tell me about your technical background.",
        rubric_criteria=rubric_criteria,
    )
    assert 0.0 <= result.overall_score <= 1.0
    assert isinstance(result.evidence_json, dict)
    assert result.used_delivery_features is False


def test_content_engine_retries_on_bad_json(rubric_criteria):
    """Engine must retry when LLM returns malformed JSON."""
    call_count = {"n": 0}

    class BrokenThenFixedLLM(MockLLMService):
        def complete(self, prompt, *, temperature=0.0, max_tokens=2048):
            call_count["n"] += 1
            if call_count["n"] < 2:
                return LLMResponse(text="not json at all {{broken", model_name="broken")
            # Return valid JSON on second attempt
            resp = {
                "criteria": [{"criterion_name": "c", "score": 0.8,
                               "covered_concepts": [], "missing_concepts": [], "evidence": "ok"}],
                "overall_score": 0.8,
                "star_detected": {"situation": False, "task": False, "action": False, "result": False},
                "uncertainty": False, "uncertainty_reason": None,
            }
            return LLMResponse(text=json.dumps(resp), model_name="fixed")

    engine = ContentEngine(llm=BrokenThenFixedLLM(), db=None)
    result = engine.analyze("test transcript", "test question", rubric_criteria)
    assert result.overall_score == pytest.approx(0.8)
    assert call_count["n"] == 2


def test_content_engine_clamps_scores_out_of_range(rubric_criteria):
    """Scores outside [0,1] in LLM output are clamped."""
    resp = {
        "criteria": [{"criterion_name": "c", "score": 1.5,  # out of range
                       "covered_concepts": [], "missing_concepts": [], "evidence": "x"}],
        "overall_score": -0.2,  # out of range
        "star_detected": {"situation": False, "task": False, "action": False, "result": False},
        "uncertainty": False, "uncertainty_reason": None,
    }
    engine = ContentEngine(llm=MockLLMService(fixed_response=json.dumps(resp)), db=None)
    result = engine.analyze("text", "question", rubric_criteria)
    assert result.overall_score == pytest.approx(0.0)  # clamped from -0.2


def test_content_engine_uncertainty_flag(rubric_criteria):
    """Uncertainty flag is propagated from LLM output."""
    resp = {
        "criteria": [{"criterion_name": "c", "score": 0.5,
                       "covered_concepts": [], "missing_concepts": [], "evidence": "x"}],
        "overall_score": 0.5,
        "star_detected": {"situation": False, "task": False, "action": False, "result": False},
        "uncertainty": True,
        "uncertainty_reason": "Answer is off-topic.",
    }
    engine = ContentEngine(llm=MockLLMService(fixed_response=json.dumps(resp)), db=None)
    result = engine.analyze("irrelevant", "What is OOP?", rubric_criteria)
    assert result.uncertainty is True


def test_content_engine_extracts_json_from_markdown(rubric_criteria):
    """Engine handles ```json ... ``` fenced LLM output."""
    resp = {
        "criteria": [{"criterion_name": "c", "score": 0.6,
                       "covered_concepts": ["OOP"], "missing_concepts": [], "evidence": "OOP"}],
        "overall_score": 0.6,
        "star_detected": {"situation": False, "task": False, "action": False, "result": False},
        "uncertainty": False, "uncertainty_reason": None,
    }
    fenced = f"Sure! Here's the evaluation:\n```json\n{json.dumps(resp)}\n```"
    engine = ContentEngine(llm=MockLLMService(fixed_response=fenced), db=None)
    result = engine.analyze("text", "question", rubric_criteria)
    assert result.overall_score == pytest.approx(0.6)


def test_ablation_delivery_features_not_passed_by_default(content_engine, rubric_criteria):
    """Without ablation flag, delivery features are NOT passed to content engine."""
    result = content_engine.analyze(
        transcript_text="Some answer.",
        question_text="Some question.",
        rubric_criteria=rubric_criteria,
        delivery_features={"speaking_rate_wpm": 150},  # passed in, but ablation=False
    )
    assert result.used_delivery_features is False
