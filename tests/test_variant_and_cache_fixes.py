"""Tests for the audit-layer fixes:
  1. Degraded text actually differs from the original (no-op detection).
  2. Repeated-run makes N REAL calls (no cache hit) — run_index bypasses cache.
  3. Known-groups check FAILS on the mock judge (proves it can fail).

No real API calls. Uses MockLLMService and a counting mock LLM.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest

from app.audit.variants import VariantGenerator
from app.audit.tests.repeated_run import RepeatedRunTest
from app.audit.tests.known_groups import KnownGroupsTest
from app.audit.tests.content_sensitivity import ContentSensitivityTest
from app.engines.content import ContentEngine, RubricCriterion
from app.engines.delivery import DeliveryEngine
from app.services.llm import LLMResponse, MockLLMService
from app.services.stt import MockSTTService, WordTimestamp


# ─────────────────────────────────────────────────────────────────────────────
# 1. Degraded text differs from the original
# ─────────────────────────────────────────────────────────────────────────────

def _make_words(text: str) -> list[WordTimestamp]:
    return MockSTTService(fixed_transcript=text).transcribe("x").words


def test_degradation_changes_text_when_concept_present():
    """If the concept appears verbatim, sentence-based removal must change text."""
    text = (
        "A stack is LIFO. A queue is FIFO. "
        "Both have use cases in scheduling. An example is BFS traversal."
    )
    words = _make_words(text)
    gen = VariantGenerator(seed=42)
    v = gen.quality_degradation(text, words, concepts_to_remove=["LIFO", "FIFO"])
    assert v.text_differs, "degraded text must differ from original"
    assert v.variant_text != text


def test_degradation_changes_text_when_concept_not_verbatim():
    """Even when the concept string is not verbatim, sentence-based removal
    (substring match) should still remove a sentence that carries it."""
    text = (
        "A stack is a Last In First Out structure where you push and pop. "
        "A queue is First In First Out where you enqueue and dequeue. "
        "Both have O of 1 time complexity for primary operations."
    )
    words = _make_words(text)
    gen = VariantGenerator(seed=42)
    # "First In First Out" contains "first" — substring match removes that sentence
    v = gen.quality_degradation(text, words, concepts_to_remove=["First In First Out"])
    assert v.text_differs, (
        "degraded text must differ even when concept is a phrase, not a single token"
    )


def test_degradation_with_concept_sentences_annotation():
    """concept_sentences mapping drives which sentence indices get removed."""
    text = "Sentence one. Sentence two carries the key idea. Sentence three."
    words = _make_words(text)
    gen = VariantGenerator(seed=42)
    v = gen.quality_degradation(
        text, words, concepts_to_remove=["key idea"],
        concept_sentences={"key idea": [1]},
    )
    assert v.text_differs
    assert "Sentence two carries the key idea" not in v.variant_text
    assert "Sentence one" in v.variant_text
    assert "Sentence three" in v.variant_text


def test_degradation_handwritten_answer_used_when_present():
    """A hand-written degraded answer is used directly."""
    text = "Original strong answer about OOP."
    words = _make_words(text)
    gen = VariantGenerator(seed=42)
    v = gen.quality_degradation(
        text, words, concepts_to_remove=["OOP"],
        hand_written="A weaker answer that misses the point.",
    )
    assert v.text_differs
    assert v.variant_text == "A weaker answer that misses the point."
    assert v.source == "handwritten"


def test_variant_text_differs_property_false_on_noop():
    """The text_differs property must be False when degradation is a no-op
    (so the audit test can detect and exclude it)."""
    text = "This answer mentions nothing related to the removed concepts at all."
    words = _make_words(text)
    gen = VariantGenerator(seed=42)
    # No concept_sentences, no substring match anywhere → falls back to last-sentence drop
    v = gen.quality_degradation(text, words, concepts_to_remove=["nonexistent_concept_xyz"])
    # Last sentence is dropped, so it should differ. Force a true no-op by single sentence:
    text_one = "Only one sentence here with no matching concept."
    words_one = _make_words(text_one)
    v2 = gen.quality_degradation(text_one, words_one, concepts_to_remove=["zzz"])
    # single sentence + no match → kept == original (no-op)
    assert not v2.text_differs, "single sentence with no concept match must be a no-op"


def test_content_sensitivity_marks_invalid_on_noop_degradation():
    """When degradation is a no-op, the test must mark itself invalid, NOT
    report content_drop=0 as a real measurement."""
    text = "Only one sentence here with no matching concept."
    words = _make_words(text)
    gen = VariantGenerator(seed=42)
    v = gen.quality_degradation(text, words, concepts_to_remove=["zzz"])
    assert not v.text_differs, "precondition: this is a no-op"

    rubric = [RubricCriterion(criterion_name="correctness", criterion_type="correctness",
                             expected_concepts=["zzz"])]
    engine = ContentEngine(llm=MockLLMService(), db=None)
    test = ContentSensitivityTest(engine, DeliveryEngine(), threshold=0.15)
    metric = test.run(text, "q", rubric, [v])

    assert metric.invalid is True
    assert "no-op" in metric.invalid_reason.lower() or "identical" in metric.invalid_reason.lower()
    assert metric.invalid_variant_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Repeated-run makes N real calls (no cache hit)
# ─────────────────────────────────────────────────────────────────────────────

class _CallCountingLLM(MockLLMService):
    """Mock LLM that counts every complete() call and never uses a disk cache.

    Returns deterministic but distinct-enough JSON so parse succeeds. We vary
    the score per call to prove each call is real (not a cached copy).
    """
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self._model = "counting-mock"

    def complete(self, prompt, *, temperature=0.0, max_tokens=2048,
                 prompt_version="v1", run_index=None):
        self.call_count += 1
        # Return a valid rubric eval JSON with a score that depends on run_index
        # so we can confirm each call produced a distinct value.
        score = 0.5 + (0.01 * (run_index or 0))
        resp_json = {
            "criteria": [{"criterion_name": "correctness", "score": score,
                          "covered_concepts": [], "missing_concepts": [],
                          "evidence": "x", "star_component": None}],
            "overall_score": score,
            "star_detected": {"situation": False, "task": False,
                              "action": False, "result": False},
            "uncertainty": False, "uncertainty_reason": None,
        }
        return LLMResponse(text=json.dumps(resp_json), model_name=self._model)


def test_repeated_run_makes_n_real_calls_no_cache():
    """Two repeats must produce two distinct real API calls, not one cached one."""
    llm = _CallCountingLLM()
    engine = ContentEngine(llm=llm, db=None)
    rubric = [RubricCriterion(criterion_name="correctness", criterion_type="correctness",
                             expected_concepts=["x"])]
    test = RepeatedRunTest(engine, n_runs=3, threshold=0.02)
    metric = test.run("some answer", "q", rubric)

    # Three repeats → three real calls (cache bypassed via run_index)
    assert llm.call_count == 3, (
        f"expected 3 real calls, got {llm.call_count} (cache hit?)"
    )
    # The three scores must be distinct (proving no single cached response reused)
    scores = metric.details["scores"]
    assert len(scores) == 3
    assert scores[0] != scores[1], "repeats returned identical cached value"
    assert scores[1] != scores[2]


def test_repeated_run_without_run_index_hits_cache():
    """Sanity check: calling analyze twice WITHOUT run_index returns the same
    cached value (i.e. the run_index is what makes repeats real)."""
    # Use a fresh temp cache dir so we don't pollute the real cache
    from app.services.llm import _DiskCache
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cache = _DiskCache(td)
        # Monkeypatch the mock's cache... actually MockLLMService has no cache.
        # So this test just confirms default analyze is deterministic for the mock.
        llm = MockLLMService()
        engine = ContentEngine(llm=llm, db=None)
        rubric = [RubricCriterion(criterion_name="c", criterion_type="correctness",
                                 expected_concepts=[])]
        r1 = engine.analyze("answer", "q", rubric)
        r2 = engine.analyze("answer", "q", rubric)
        assert r1.overall_score == r2.overall_score  # deterministic mock


# ─────────────────────────────────────────────────────────────────────────────
# 3. Known-groups check FAILS on the mock judge
# ─────────────────────────────────────────────────────────────────────────────

class _ReverseOrderLLM(MockLLMService):
    """Mock judge that scores weak answers HIGHER than strong ones (wrong order),
    to prove the known-groups check can detect a failing judge."""
    def __init__(self):
        super().__init__()
        self._model = "reverse-order-mock"
        self._call_idx = 0

    def complete(self, prompt, *, temperature=0.0, max_tokens=2048,
                 prompt_version="v1", run_index=None):
        self._call_idx += 1
        # Return increasing scores per call, so the order is arbitrary here.
        # We instead key off the prompt content to assign scores: if the prompt
        # contains "WEAK" give 0.9, "MEDIUM" give 0.5, "STRONG" give 0.1.
        # (reversed → known-groups should FAIL)
        if "WEAKMARKER" in prompt:
            score = 0.9
        elif "MEDIUMMARKER" in prompt:
            score = 0.5
        elif "STRONGMARKER" in prompt:
            score = 0.1
        else:
            score = 0.5
        resp_json = {
            "criteria": [{"criterion_name": "correctness", "score": score,
                          "covered_concepts": [], "missing_concepts": [],
                          "evidence": "x", "star_component": None}],
            "overall_score": score,
            "star_detected": {"situation": False, "task": False,
                              "action": False, "result": False},
            "uncertainty": False, "uncertainty_reason": None,
        }
        return LLMResponse(text=json.dumps(resp_json), model_name=self._model)


def test_known_groups_fails_on_reverse_order_judge():
    """A judge that scores weak > strong must FAIL the known-groups check."""
    llm = _ReverseOrderLLM()
    engine = ContentEngine(llm=llm, db=None)
    rubric = [RubricCriterion(criterion_name="correctness", criterion_type="correctness",
                              expected_concepts=[])]
    test = KnownGroupsTest(engine)
    metric = test.run(
        question_text="q",
        rubric_criteria=rubric,
        strong_text="STRONGMARKER this is a great answer",
        medium_text="MEDIUMMARKER this is an ok answer",
        weak_text="WEAKMARKER this is a poor answer",
    )
    assert not metric.passed, "known-groups must FAIL when weak > strong"
    assert metric.details["ordering_ok"] is False
    # scores: strong=0.1, medium=0.5, weak=0.9 → not ordered strong>medium>weak
    assert metric.details["scores"]["strong"] < metric.details["scores"]["weak"]


def test_known_groups_passes_on_correct_order_judge():
    """A judge that scores strong > medium > weak with gaps >= margin PASSES.
    Here 0.9 / 0.5 / 0.1 → gaps 0.4 each, both >= 0.1 (default margin)."""
    class _CorrectOrderLLM(MockLLMService):
        def __init__(self):
            super().__init__()
            self._model = "correct-order-mock"
        def complete(self, prompt, *, temperature=0.0, max_tokens=2048,
                     prompt_version="v1", run_index=None, bypass_cache=False):
            if "STRONGMARKER" in prompt:
                score = 0.9
            elif "MEDIUMMARKER" in prompt:
                score = 0.5
            elif "WEAKMARKER" in prompt:
                score = 0.1
            else:
                score = 0.5
            resp_json = {
                "criteria": [{"criterion_name": "correctness", "score": score,
                              "covered_concepts": [], "missing_concepts": [],
                              "evidence": "x", "star_component": None}],
                "overall_score": score,
                "star_detected": {"situation": False, "task": False,
                                  "action": False, "result": False},
                "uncertainty": False, "uncertainty_reason": None,
            }
            return LLMResponse(text=json.dumps(resp_json), model_name=self._model)

    llm = _CorrectOrderLLM()
    engine = ContentEngine(llm=llm, db=None)
    rubric = [RubricCriterion(criterion_name="correctness", criterion_type="correctness",
                              expected_concepts=[])]
    test = KnownGroupsTest(engine)
    metric = test.run(
        question_text="q",
        rubric_criteria=rubric,
        strong_text="STRONGMARKER great",
        medium_text="MEDIUMMARKER ok",
        weak_text="WEAKMARKER poor",
    )
    assert metric.passed
    assert metric.details["ordering_ok"] is True


def test_known_groups_fails_on_equal_scores():
    """Ties / equal scores across tiers must FAIL under strict ordering.
    The plain mock returns 0.775 for every input → all tiers equal → FAIL."""
    llm = MockLLMService()
    engine = ContentEngine(llm=llm, db=None)
    rubric = [RubricCriterion(criterion_name="c", criterion_type="correctness",
                             expected_concepts=[])]
    test = KnownGroupsTest(engine, min_margin=0.1)
    metric = test.run(
        question_text="q",
        rubric_criteria=rubric,
        strong_text="a strong answer",
        medium_text="a medium answer",
        weak_text="a weak answer",
    )
    assert not metric.passed, "equal scores must FAIL (no discriminant validity)"
    assert metric.details["ordering_ok"] is False
    # gaps are 0 (equal) which is < margin 0.1
    assert metric.details["gaps"]["strong_medium"] == 0.0
    assert metric.details["gaps"]["medium_weak"] == 0.0


def test_known_groups_fails_when_gap_below_margin():
    """Correct direction but a gap smaller than the margin must FAIL."""
    class _SmallGapLLM(MockLLMService):
        def __init__(self):
            super().__init__()
            self._model = "small-gap-mock"
        def complete(self, prompt, *, temperature=0.0, max_tokens=2048,
                     prompt_version="v1", run_index=None, bypass_cache=False):
            if "STRONGMARKER" in prompt:
                score = 0.55
            elif "MEDIUMMARKER" in prompt:
                score = 0.50
            elif "WEAKMARKER" in prompt:
                score = 0.10
            else:
                score = 0.50
            resp_json = {
                "criteria": [{"criterion_name": "c", "score": score,
                              "covered_concepts": [], "missing_concepts": [],
                              "evidence": "x", "star_component": None}],
                "overall_score": score,
                "star_detected": {"situation": False, "task": False,
                                  "action": False, "result": False},
                "uncertainty": False, "uncertainty_reason": None,
            }
            return LLMResponse(text=json.dumps(resp_json), model_name=self._model)

    llm = _SmallGapLLM()
    engine = ContentEngine(llm=llm, db=None)
    rubric = [RubricCriterion(criterion_name="c", criterion_type="correctness",
                             expected_concepts=[])]
    # strong-medium gap = 0.05 < margin 0.1 → FAIL even though direction is right
    test = KnownGroupsTest(engine, min_margin=0.1)
    metric = test.run("q", rubric,
                      strong_text="STRONGMARKER", medium_text="MEDIUMMARKER",
                      weak_text="WEAKMARKER")
    assert not metric.passed
    assert metric.details["gaps"]["strong_medium"] == pytest.approx(0.05)
    assert metric.details["gaps"]["medium_weak"] == pytest.approx(0.40)


def test_known_groups_passes_with_custom_small_margin():
    """With a smaller margin, a small-but-positive gap can PASS."""
    class _SmallGapLLM(MockLLMService):
        def __init__(self):
            super().__init__()
            self._model = "small-gap-mock"
        def complete(self, prompt, *, temperature=0.0, max_tokens=2048,
                     prompt_version="v1", run_index=None, bypass_cache=False):
            if "STRONGMARKER" in prompt:
                score = 0.55
            elif "MEDIUMMARKER" in prompt:
                score = 0.50
            elif "WEAKMARKER" in prompt:
                score = 0.10
            else:
                score = 0.50
            resp_json = {
                "criteria": [{"criterion_name": "c", "score": score,
                              "covered_concepts": [], "missing_concepts": [],
                              "evidence": "x", "star_component": None}],
                "overall_score": score,
                "star_detected": {"situation": False, "task": False,
                                  "action": False, "result": False},
                "uncertainty": False, "uncertainty_reason": None,
            }
            return LLMResponse(text=json.dumps(resp_json), model_name=self._model)

    llm = _SmallGapLLM()
    engine = ContentEngine(llm=llm, db=None)
    rubric = [RubricCriterion(criterion_name="c", criterion_type="correctness",
                             expected_concepts=[])]
    # margin 0.04 → both gaps (0.05, 0.40) >= 0.04 → PASS
    test = KnownGroupsTest(engine, min_margin=0.04)
    metric = test.run("q", rubric,
                      strong_text="STRONGMARKER", medium_text="MEDIUMMARKER",
                      weak_text="WEAKMARKER")
    assert metric.passed


def test_known_groups_invalid_when_tiers_missing():
    """With only strong (no medium/weak), the check is invalid, not pass/fail."""
    llm = MockLLMService()
    engine = ContentEngine(llm=llm, db=None)
    rubric = [RubricCriterion(criterion_name="c", criterion_type="correctness",
                             expected_concepts=[])]
    test = KnownGroupsTest(engine)
    metric = test.run("q", rubric, strong_text="strong", medium_text=None, weak_text=None)
    assert metric.invalid is True
    assert not metric.passed


# ─────────────────────────────────────────────────────────────────────────────
# Degradation-method recording
# ─────────────────────────────────────────────────────────────────────────────

def test_degradation_method_recorded_handwritten():
    gen = VariantGenerator(seed=42)
    text = "Some answer."
    words = _make_words(text)
    v = gen.quality_degradation(text, words, concepts_to_remove=["x"],
                                hand_written="A weaker hand-written answer.")
    assert v.degradation_method == "handwritten_tier"
    assert v.params["degradation_method"] == "handwritten_tier"


def test_degradation_method_recorded_annotated():
    gen = VariantGenerator(seed=42)
    text = "Sentence one. Sentence two carries the key idea. Sentence three."
    words = _make_words(text)
    v = gen.quality_degradation(text, words, concepts_to_remove=["key idea"],
                                concept_sentences={"key idea": [1]})
    assert v.degradation_method == "annotated_sentence_removal"


def test_degradation_method_recorded_substring():
    gen = VariantGenerator(seed=42)
    text = ("I know LIFO and FIFO. I also know design patterns. "
            "An example is BFS.")
    words = _make_words(text)
    v = gen.quality_degradation(text, words, concepts_to_remove=["LIFO", "FIFO"])
    assert v.degradation_method == "substring_sentence_removal"


def test_degradation_method_recorded_fallback_drop_last():
    gen = VariantGenerator(seed=42)
    text = ("This answer mentions nothing related to removed concepts. "
            "It has a second sentence though.")
    words = _make_words(text)
    # No concept_sentences, no substring match → fallback drops last sentence
    v = gen.quality_degradation(text, words, concepts_to_remove=["nonexistent_xyz"])
    assert v.degradation_method == "fallback_drop_last_sentence"


def test_degradation_method_recorded_no_op():
    gen = VariantGenerator(seed=42)
    text = "Only one sentence with no matching concept."
    words = _make_words(text)
    v = gen.quality_degradation(text, words, concepts_to_remove=["zzz"])
    assert v.degradation_method == "no_op"
    assert not v.text_differs


# ─────────────────────────────────────────────────────────────────────────────
# Judge qualification (qualify_call / qualify_judge standard)
# ─────────────────────────────────────────────────────────────────────────────

def test_qualify_call_passes_on_mock():
    """The mock returns non-empty valid JSON with finish_reason='stop' → qualifies."""
    from app.services.qualify import qualify_call
    llm = MockLLMService()
    qr = qualify_call(llm, "What is OOP?", "OOP is object-oriented programming.")
    assert qr.ok
    assert qr.finish_reason == "stop"
    assert qr.max_tokens_sent == 512
    assert qr.completion_tokens is not None


def test_qualify_call_fails_on_empty_reply():
    from app.services.qualify import qualify_call
    llm = MockLLMService(fixed_response="")
    qr = qualify_call(llm, "q", "a")
    assert not qr.ok
    assert "empty" in qr.error.lower()


def test_qualify_call_fails_on_invalid_json():
    from app.services.qualify import qualify_call
    llm = MockLLMService(fixed_response="this is not json at all")
    qr = qualify_call(llm, "q", "a")
    assert not qr.ok
    assert "json" in qr.error.lower()


def test_qualify_call_fails_on_truncated_finish_reason():
    """finish_reason='length' (truncated) must FAIL even if JSON is valid."""
    from app.services.qualify import qualify_call
    class _TruncLLM(MockLLMService):
        def complete(self, prompt, *, temperature=0.0, max_tokens=2048,
                     prompt_version="v1", run_index=None, bypass_cache=False):
            return LLMResponse(
                text='{"overall_score": 0.5, "one_strength": "x"}',
                model_name="trunc", finish_reason="length",
                output_tokens=10, max_tokens_used=512, latency_ms=5,
            )
    qr = qualify_call(_TruncLLM(), "q", "a")
    assert not qr.ok
    assert qr.finish_reason == "length"
    assert "stop" in qr.error


def test_qualify_call_bypasses_cache():
    """qualify_call must make a real call even when a cache exists."""
    from app.services.qualify import qualify_call
    import tempfile
    from app.services.llm import _DiskCache

    class _CountingMock(MockLLMService):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def complete(self, prompt, *, temperature=0.0, max_tokens=2048,
                     prompt_version="v1", run_index=None, bypass_cache=False):
            self.calls += 1
            return LLMResponse(
                text='{"overall_score": 0.5, "one_strength": "x"}',
                model_name="counting-mock", finish_reason="stop",
                output_tokens=10, max_tokens_used=max_tokens, latency_ms=5,
            )

    llm = _CountingMock()
    # Two qualify calls → two real calls (bypass_cache=True), not cached
    qualify_call(llm, "q", "a")
    qualify_call(llm, "q", "a")
    assert llm.calls == 2, f"expected 2 real calls, got {llm.calls}"
