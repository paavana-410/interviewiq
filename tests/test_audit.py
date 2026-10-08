"""Audit layer tests with mocked LLM."""
from __future__ import annotations

import pytest

from app.audit.combiner import AuditCombiner
from app.audit.tests.delivery_invariance import DeliveryInvarianceTest
from app.audit.tests.paraphrase_stability import ParaphraseStabilityTest
from app.audit.tests.repeated_run import RepeatedRunTest
from app.audit.tests.stt_sensitivity import STTSensitivityTest
from app.audit.variants import VariantGenerator
from app.services.llm import MockLLMService
from app.engines.content import ContentEngine, RubricCriterion
from app.services.stt import WordTimestamp


TRANSCRIPT = "I have strong experience with object-oriented programming and design patterns."
QUESTION = "Describe your technical background."


@pytest.fixture
def rubric():
    return [
        RubricCriterion(
            criterion_name="technical_correctness",
            criterion_type="correctness",
            weight=1.0,
            expected_concepts=["OOP", "design patterns"],
            star_component=False,
            scoring_guide="1.0 if OOP mentioned.",
        )
    ]


@pytest.fixture
def mock_engine():
    return ContentEngine(llm=MockLLMService(), db=None)


# ─────────────────────────────────────────────────────────────────────────────
# Delivery Invariance
# ─────────────────────────────────────────────────────────────────────────────

def test_delivery_invariance_passes_with_mock(mock_engine, rubric, variant_generator, sample_words):
    filler_v = variant_generator.filler_injection(sample_words)
    pace_v = variant_generator.pace_change(sample_words, factor=1.5)

    test = DeliveryInvarianceTest(mock_engine, threshold=0.5)  # generous threshold for mock
    metric = test.run(
        original_transcript=TRANSCRIPT,
        question_text=QUESTION,
        rubric_criteria=rubric,
        delivery_variants=[filler_v, pace_v],
    )
    assert metric.test_name == "delivery_invariance"
    assert isinstance(metric.metric_value, float)
    assert metric.passed  # mock LLM returns same score every time → delta ≈ 0


def test_delivery_invariance_detects_contamination(rubric, variant_generator, sample_words):
    """LLM that returns different scores for different inputs should fail the test."""
    import json
    call_count = {"n": 0}

    class FluctuatingMockLLM(MockLLMService):
        def complete(self, prompt, *, temperature=0.0, max_tokens=2048):
            call_count["n"] += 1
            score = 0.9 if call_count["n"] == 1 else 0.1  # huge delta
            resp_json = {
                "criteria": [{"criterion_name": "technical_correctness", "score": score,
                               "covered_concepts": [], "missing_concepts": [], "evidence": "ok"}],
                "overall_score": score,
                "star_detected": {"situation": False, "task": False, "action": False, "result": False},
                "uncertainty": False, "uncertainty_reason": None,
            }
            from app.services.llm import LLMResponse
            return LLMResponse(text=json.dumps(resp_json), model_name="fluctuating-mock")

    engine = ContentEngine(llm=FluctuatingMockLLM(), db=None)
    filler_v = variant_generator.filler_injection(sample_words)
    test = DeliveryInvarianceTest(engine, threshold=0.08)
    metric = test.run(TRANSCRIPT, QUESTION, rubric, [filler_v])
    assert not metric.passed, "Should fail when content scores change on delivery-only variants"


# ─────────────────────────────────────────────────────────────────────────────
# Paraphrase Stability
# ─────────────────────────────────────────────────────────────────────────────

def test_paraphrase_stability(mock_engine, rubric, variant_generator, sample_words):
    para_v = variant_generator.paraphrase(TRANSCRIPT, sample_words, llm_service=None)
    test = ParaphraseStabilityTest(mock_engine, threshold=0.5)
    metric = test.run(TRANSCRIPT, QUESTION, rubric, [para_v])
    assert metric.test_name == "paraphrase_stability"
    assert isinstance(metric.metric_value, float)
    assert metric.passed  # mock returns same score


def test_paraphrase_needs_human_verification(variant_generator, sample_words):
    """Paraphrase variants from LLM should be flagged for human review."""
    para_v = variant_generator.paraphrase(TRANSCRIPT, sample_words, llm_service=None)
    assert para_v.needs_human_verification is True


# ─────────────────────────────────────────────────────────────────────────────
# Repeated Run
# ─────────────────────────────────────────────────────────────────────────────

def test_repeated_run_variance(mock_engine, rubric):
    test = RepeatedRunTest(mock_engine, n_runs=5, threshold=0.02)
    metric = test.run(TRANSCRIPT, QUESTION, rubric)
    assert metric.test_name == "repeated_run"
    assert metric.metric_value == pytest.approx(0.0, abs=1e-6)  # mock always returns same score
    assert metric.passed


def test_repeated_run_high_variance_fails(rubric):
    """LLM with high variance should fail the repeated-run test."""
    import json
    import random
    rng = random.Random(99)

    class NoisyMockLLM(MockLLMService):
        def complete(self, prompt, *, temperature=0.0, max_tokens=2048):
            score = rng.uniform(0.0, 1.0)
            resp_json = {
                "criteria": [{"criterion_name": "c", "score": score,
                               "covered_concepts": [], "missing_concepts": [], "evidence": "x"}],
                "overall_score": score,
                "star_detected": {"situation": False, "task": False, "action": False, "result": False},
                "uncertainty": False, "uncertainty_reason": None,
            }
            from app.services.llm import LLMResponse
            return LLMResponse(text=json.dumps(resp_json), model_name="noisy-mock")

    engine = ContentEngine(llm=NoisyMockLLM(), db=None)
    test = RepeatedRunTest(engine, n_runs=10, threshold=0.01)
    metric = test.run(TRANSCRIPT, QUESTION, rubric)
    assert not metric.passed, "High-variance LLM should fail repeatability"


# ─────────────────────────────────────────────────────────────────────────────
# STT Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def test_stt_sensitivity_same_transcripts(mock_engine, rubric):
    """Identical gold and whisper transcripts → zero drift."""
    test = STTSensitivityTest(mock_engine, threshold=0.08)
    metric = test.run(TRANSCRIPT, TRANSCRIPT, QUESTION, rubric)
    assert metric.metric_value == pytest.approx(0.0, abs=1e-6)
    assert metric.passed


def test_stt_sensitivity_different_transcripts(mock_engine, rubric):
    """With mock (same score always), even different transcripts → zero drift."""
    gold = TRANSCRIPT
    whisper = "I know a bit about programming and some computer stuff."
    test = STTSensitivityTest(mock_engine, threshold=0.08)
    metric = test.run(gold, whisper, QUESTION, rubric)
    # Mock always returns same score, so drift = 0
    assert metric.metric_value == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Combiner
# ─────────────────────────────────────────────────────────────────────────────

def test_combiner_all_pass(mock_engine, rubric, variant_generator, sample_words):
    metrics = []
    # Delivery invariance
    filler_v = variant_generator.filler_injection(sample_words)
    di_test = DeliveryInvarianceTest(mock_engine, threshold=0.5)
    metrics.append(di_test.run(TRANSCRIPT, QUESTION, rubric, [filler_v]))
    # Paraphrase
    para_v = variant_generator.paraphrase(TRANSCRIPT, sample_words)
    ps_test = ParaphraseStabilityTest(mock_engine, threshold=0.5)
    metrics.append(ps_test.run(TRANSCRIPT, QUESTION, rubric, [para_v]))
    # Repeated run
    rr_test = RepeatedRunTest(mock_engine, n_runs=3, threshold=0.5)
    metrics.append(rr_test.run(TRANSCRIPT, QUESTION, rubric))

    report = AuditCombiner().combine(metrics)
    assert report.all_tests_passed is True
    assert report.uncertainty_flag is False


def test_combiner_flags_uncertainty_on_failure(rubric, variant_generator, sample_words):
    import json
    call_count = {"n": 0}

    class FluctuatingMockLLM(MockLLMService):
        def complete(self, prompt, *, temperature=0.0, max_tokens=2048):
            call_count["n"] += 1
            score = 0.9 if call_count["n"] == 1 else 0.2
            resp_json = {
                "criteria": [{"criterion_name": "c", "score": score,
                               "covered_concepts": [], "missing_concepts": [], "evidence": "x"}],
                "overall_score": score,
                "star_detected": {"situation": False, "task": False, "action": False, "result": False},
                "uncertainty": False, "uncertainty_reason": None,
            }
            from app.services.llm import LLMResponse
            return LLMResponse(text=json.dumps(resp_json), model_name="fluct")

    engine = ContentEngine(llm=FluctuatingMockLLM(), db=None)
    filler_v = variant_generator.filler_injection(sample_words)
    di_test = DeliveryInvarianceTest(engine, threshold=0.08)
    metric = di_test.run(TRANSCRIPT, QUESTION, rubric, [filler_v])

    report = AuditCombiner().combine([metric])
    assert report.uncertainty_flag is True


# ─────────────────────────────────────────────────────────────────────────────
# Variant Generator
# ─────────────────────────────────────────────────────────────────────────────

def test_filler_injection_preserves_content(variant_generator, sample_words):
    """Filler injection must not change content words."""
    original_content = [w.word for w in sample_words]
    variant = variant_generator.filler_injection(sample_words)
    # All original words must still appear in the variant
    variant_words = [w.word for w in variant.variant_words]
    for word in original_content:
        assert word in variant_words, f"Word '{word}' lost after filler injection"


def test_pace_change_preserves_word_count(variant_generator, sample_words):
    variant = variant_generator.pace_change(sample_words, factor=0.5)
    assert len(variant.variant_words) == len(sample_words)


def test_pace_change_scales_timestamps(variant_generator, sample_words):
    factor = 2.0
    variant = variant_generator.pace_change(sample_words, factor=factor)
    orig_dur = sample_words[-1].end - sample_words[0].start
    new_dur = variant.variant_words[-1].end - variant.variant_words[0].start
    assert abs(new_dur - orig_dur * factor) < 0.01


def test_pause_insertion_shifts_timestamps(variant_generator, sample_words):
    variant = variant_generator.pause_insertion(sample_words, num_pauses=2, pause_duration_sec=3.0)
    # Total duration should be longer than original
    orig_dur = sample_words[-1].end - sample_words[0].start
    new_dur = variant.variant_words[-1].end - variant.variant_words[0].start
    assert new_dur > orig_dur


def test_quality_degradation_changes_text(variant_generator, sample_words):
    variant = variant_generator.quality_degradation(
        text=TRANSCRIPT,
        words=sample_words,
        concepts_to_remove=["object-oriented", "design patterns"],
    )
    assert "object-oriented" not in variant.variant_text
    assert variant.variant_text != TRANSCRIPT


def test_seeded_variants_are_reproducible(sample_words):
    gen1 = VariantGenerator(seed=7)
    gen2 = VariantGenerator(seed=7)
    v1 = gen1.filler_injection(sample_words)
    v2 = gen2.filler_injection(sample_words)
    assert [w.word for w in v1.variant_words] == [w.word for w in v2.variant_words]
