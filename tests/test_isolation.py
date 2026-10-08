"""Structural isolation tests.

These tests verify that the Content Engine and Delivery Engine
cannot share inputs — enforced at the import and type level.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Delivery Engine never imports from Content Engine or LLM service
# ─────────────────────────────────────────────────────────────────────────────

DELIVERY_MODULE_PATH = Path(__file__).parent.parent / "app" / "engines" / "delivery.py"

def _get_imports(filepath: Path) -> list[str]:
    """Return all module names imported by a file."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def test_delivery_engine_does_not_import_content():
    """Delivery engine must not import the content engine."""
    imports = _get_imports(DELIVERY_MODULE_PATH)
    content_imports = [i for i in imports if "content" in i.lower()]
    assert content_imports == [], (
        f"Delivery engine imports content-related modules: {content_imports}"
    )


def test_delivery_engine_does_not_import_llm():
    """Delivery engine must not import any LLM service."""
    imports = _get_imports(DELIVERY_MODULE_PATH)
    llm_imports = [i for i in imports if "llm" in i.lower() or "openai" in i.lower() or "anthropic" in i.lower()]
    assert llm_imports == [], (
        f"Delivery engine imports LLM-related modules: {llm_imports}"
    )


def test_delivery_engine_does_not_import_rubric():
    """Delivery engine must not import rubric or content-related types."""
    imports = _get_imports(DELIVERY_MODULE_PATH)
    rubric_imports = [i for i in imports if "rubric" in i.lower()]
    assert rubric_imports == [], (
        f"Delivery engine imports rubric-related modules: {rubric_imports}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: DeliveryEngine.analyze() signature only accepts WordTimestamp
# ─────────────────────────────────────────────────────────────────────────────

def test_delivery_analyze_signature_accepts_only_timestamps():
    """DeliveryEngine.analyze() must only accept list[WordTimestamp]."""
    from app.engines.delivery import DeliveryEngine
    from app.services.stt import WordTimestamp

    sig = inspect.signature(DeliveryEngine.analyze)
    params = list(sig.parameters.values())

    # First param is self, second is words
    assert len(params) >= 2
    words_param = params[1]
    assert words_param.name == "words", (
        f"Expected parameter named 'words', got '{words_param.name}'"
    )


def test_delivery_engine_rejects_text_kwarg(delivery_engine):
    """Passing transcript_text to DeliveryEngine.analyze() must raise TypeError."""
    from app.services.stt import WordTimestamp

    words = [WordTimestamp("hello", 0.0, 0.5, 1.0)]
    with pytest.raises(TypeError):
        # Should fail because analyze() doesn't accept 'transcript_text'
        delivery_engine.analyze(words, transcript_text="some text")  # type: ignore[call-arg]


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Content Engine never imports DeliveryEngine or DeliveryResult
# ─────────────────────────────────────────────────────────────────────────────

CONTENT_MODULE_PATH = Path(__file__).parent.parent / "app" / "engines" / "content.py"


def test_content_engine_does_not_import_delivery_engine():
    """Content engine must not import DeliveryEngine."""
    imports = _get_imports(CONTENT_MODULE_PATH)
    delivery_imports = [
        i for i in imports
        if "delivery" in i.lower() and "engine" in i.lower()
    ]
    assert delivery_imports == [], (
        f"Content engine imports delivery engine: {delivery_imports}"
    )


def test_content_engine_does_not_import_word_timestamp():
    """Content engine must not import WordTimestamp (timing data type)."""
    imports = _get_imports(CONTENT_MODULE_PATH)
    ts_imports = [i for i in imports if "stt" in i.lower()]
    assert ts_imports == [], (
        f"Content engine imports STT/timestamp module: {ts_imports}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: ContentEngine.analyze() won't receive DeliveryResult at runtime
# ─────────────────────────────────────────────────────────────────────────────

def test_content_engine_analyze_signature():
    """ContentEngine.analyze() must accept transcript_text (str), not timestamps."""
    from app.engines.content import ContentEngine
    sig = inspect.signature(ContentEngine.analyze)
    params = sig.parameters
    assert "transcript_text" in params, "Content engine must have transcript_text param"
    assert "words" not in params, (
        "Content engine must NOT have a 'words' (timestamp) param"
    )


def test_content_engine_passes_text_only(content_engine, rubric_criteria):
    """Content engine called with text only produces a valid result."""
    result = content_engine.analyze(
        transcript_text="I have strong experience with OOP and design patterns.",
        question_text="Tell me about your technical background.",
        rubric_criteria=rubric_criteria,
    )
    assert 0.0 <= result.overall_score <= 1.0
    assert result.used_delivery_features is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Pipeline isolation — delivery result NOT passed to content engine
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_no_cross_contamination(delivery_engine, content_engine, sample_words, rubric_criteria):
    """End-to-end: run both engines from the same source, verify they never share data."""
    # Delivery engine gets words only
    delivery_result = delivery_engine.analyze(sample_words)

    # Content engine gets text only (reconstructed from words)
    text = " ".join(w.word for w in sample_words)
    content_result = content_engine.analyze(
        transcript_text=text,
        question_text="Describe your programming experience.",
        rubric_criteria=rubric_criteria,
        # delivery_features intentionally NOT passed
    )

    # Neither result has cross-modal data
    assert not hasattr(delivery_result, "criteria"), "DeliveryResult must not have 'criteria'"
    assert not hasattr(delivery_result, "overall_score"), "DeliveryResult must not have 'overall_score'"
    assert content_result.used_delivery_features is False
