"""Unit tests for the Delivery Engine — fully deterministic, no LLM."""
from __future__ import annotations

import pytest

from app.engines.delivery import DeliveryEngine
from app.services.stt import WordTimestamp


# ─────────────────────────────────────────────────────────────────────────────
# WPM
# ─────────────────────────────────────────────────────────────────────────────

def test_speaking_rate_basic(sample_words, delivery_engine):
    result = delivery_engine.analyze(sample_words)
    duration = sample_words[-1].end - sample_words[0].start
    expected_wpm = len(sample_words) / duration * 60.0
    assert abs(result.speaking_rate_wpm - expected_wpm) < 0.1


def test_speaking_rate_empty():
    engine = DeliveryEngine()
    result = engine.analyze([])
    assert result.speaking_rate_wpm == 0.0
    assert result.duration_sec == 0.0


def test_speaking_rate_single_word():
    engine = DeliveryEngine()
    words = [WordTimestamp(word="hello", start=0.0, end=0.5, probability=1.0)]
    result = engine.analyze(words)
    # 1 word / 0.5s * 60 = 120 WPM
    assert abs(result.speaking_rate_wpm - 120.0) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Pauses
# ─────────────────────────────────────────────────────────────────────────────

def test_pause_detection(sample_words, delivery_engine):
    """The gap between 'um' (end=2.75) and 'and' (start=3.80) = 1.05s > threshold=1.0s."""
    result = delivery_engine.analyze(sample_words)
    assert result.pause_count >= 1
    long_pauses = [p for p in result.pauses if p.is_long]
    assert len(long_pauses) >= 1
    assert result.long_pause_count == len(long_pauses)


def test_long_pause_threshold_respected():
    """Pauses just below the threshold should not be counted as long."""
    engine = DeliveryEngine(long_pause_threshold_sec=2.0)
    words = [
        WordTimestamp("hello", 0.0, 0.5, 1.0),
        WordTimestamp("world", 1.6, 2.1, 1.0),  # gap = 1.1s, < 2.0s threshold
    ]
    result = engine.analyze(words)
    assert result.pause_count == 1
    assert result.long_pause_count == 0


def test_no_pauses_on_fast_speech():
    engine = DeliveryEngine(long_pause_threshold_sec=1.5)
    words = [
        WordTimestamp(f"word{i}", i * 0.3, i * 0.3 + 0.25, 1.0) for i in range(10)
    ]
    result = engine.analyze(words)
    assert result.long_pause_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Fillers
# ─────────────────────────────────────────────────────────────────────────────

def test_filler_detection(sample_words, delivery_engine):
    """'um' is in the sample words and should be detected."""
    result = delivery_engine.analyze(sample_words)
    assert result.filler_count >= 1
    filler_words_found = [f.word for f in result.fillers]
    assert "um" in filler_words_found


def test_multi_word_filler():
    engine = DeliveryEngine(filler_words=["you know", "um"])
    words = [
        WordTimestamp("I", 0.0, 0.2, 1.0),
        WordTimestamp("you", 0.25, 0.45, 1.0),
        WordTimestamp("know", 0.48, 0.68, 1.0),
        WordTimestamp("it", 0.70, 0.85, 1.0),
    ]
    result = engine.analyze(words)
    assert result.filler_count == 1
    assert result.fillers[0].word == "you know"


def test_filler_density_formula():
    """filler_density = fillers / word_count * 100."""
    engine = DeliveryEngine(filler_words=["um"])
    words = [WordTimestamp("um" if i % 5 == 0 else "word", i * 0.3, i * 0.3 + 0.25, 1.0)
             for i in range(10)]
    result = engine.analyze(words)
    expected_density = result.filler_count / len(words) * 100
    assert abs(result.filler_density - expected_density) < 0.01


def test_no_false_positive_fillers():
    """Words containing filler substrings but not matching exactly should not be flagged."""
    engine = DeliveryEngine(filler_words=["like"])
    words = [
        WordTimestamp("likelihood", 0.0, 0.5, 1.0),
        WordTimestamp("dislike", 0.6, 1.0, 1.0),
        WordTimestamp("like", 1.1, 1.3, 1.0),  # only this should match
    ]
    result = engine.analyze(words)
    assert result.filler_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Repetitions
# ─────────────────────────────────────────────────────────────────────────────

def test_repetition_detection(sample_words, delivery_engine):
    """'I have I have' in the sample should detect a repetition."""
    result = delivery_engine.analyze(sample_words)
    assert result.repetition_count >= 1


def test_bigram_repetition():
    engine = DeliveryEngine()
    words = [
        WordTimestamp("the", 0.0, 0.2, 1.0),
        WordTimestamp("cat", 0.25, 0.45, 1.0),
        WordTimestamp("the", 0.50, 0.70, 1.0),  # "the cat" repeated
        WordTimestamp("cat", 0.75, 0.95, 1.0),
    ]
    result = engine.analyze(words)
    phrases = [r.phrase for r in result.repetitions]
    assert "the cat" in phrases or "the" in phrases or "cat" in phrases


def test_no_false_positive_repetitions():
    """Non-consecutive occurrences of the same word should NOT be flagged.

    The word 'hello' appears at indices 0 and 4, separated by 3 other words.
    Only consecutive / immediate repetitions should be detected.
    """
    engine = DeliveryEngine()
    # "hello world foo bar hello world" — 'hello' at 0 and 4, 'world' at 1 and 5
    word_list = ["hello", "world", "foo", "bar", "hello", "world"]
    words = [
        WordTimestamp(w, i * 0.4, i * 0.4 + 0.3, 1.0)
        for i, w in enumerate(word_list)
    ]
    result = engine.analyze(words)
    # No consecutive pairs → 0 repetitions
    assert result.repetition_count == 0, (
        f"Expected 0 repetitions for non-adjacent reuse, got {result.repetition_count}: "
        f"{result.repetitions}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Duration
# ─────────────────────────────────────────────────────────────────────────────

def test_duration(sample_words, delivery_engine):
    result = delivery_engine.analyze(sample_words)
    expected = sample_words[-1].end - sample_words[0].start
    assert abs(result.duration_sec - expected) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# to_dict completeness
# ─────────────────────────────────────────────────────────────────────────────

def test_to_dict_keys(sample_words, delivery_engine):
    result = delivery_engine.analyze(sample_words)
    d = result.to_dict()
    assert "speaking_rate_wpm" in d
    assert "filler_density" in d
    assert "detail" in d
    assert "pauses" in d["detail"]
    assert "fillers" in d["detail"]
    assert "repetitions" in d["detail"]
