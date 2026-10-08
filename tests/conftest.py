"""Shared pytest fixtures."""
from __future__ import annotations

import os

import pytest

# Use mock provider for all tests — no API keys needed
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_interviewiq.db")


from app.services.stt import MockSTTService, WordTimestamp
from app.services.llm import MockLLMService
from app.engines.delivery import DeliveryEngine
from app.engines.content import ContentEngine, RubricCriterion
from app.audit.variants import VariantGenerator


@pytest.fixture
def sample_words() -> list[WordTimestamp]:
    """Realistic word-timestamp list for delivery engine tests."""
    raw = [
        ("I", 0.0, 0.15),
        ("have", 0.18, 0.35),
        ("strong", 0.38, 0.62),
        ("experience", 0.65, 1.10),
        ("with", 1.12, 1.25),
        ("object", 1.28, 1.55),
        ("oriented", 1.58, 1.90),
        ("programming", 1.93, 2.45),
        ("um", 2.55, 2.75),          # filler
        ("and", 3.80, 3.92),         # long pause before (gap = 1.05s)
        ("I", 3.95, 4.05),
        ("have", 4.08, 4.22),
        ("I", 4.25, 4.35),           # immediate repetition of "I have"
        ("have", 4.38, 4.52),
        ("worked", 4.55, 4.80),
        ("on", 4.82, 4.92),
        ("several", 4.95, 5.30),
        ("large", 5.33, 5.60),
        ("scale", 5.62, 5.90),
        ("projects", 5.93, 6.35),
    ]
    return [WordTimestamp(word=w, start=s, end=e, probability=0.95) for w, s, e in raw]


@pytest.fixture
def delivery_engine() -> DeliveryEngine:
    return DeliveryEngine(long_pause_threshold_sec=1.0, filler_words=["um", "uh", "like", "you know"])


@pytest.fixture
def mock_llm() -> MockLLMService:
    return MockLLMService()


@pytest.fixture
def content_engine(mock_llm) -> ContentEngine:
    return ContentEngine(llm=mock_llm, db=None)


@pytest.fixture
def rubric_criteria() -> list[RubricCriterion]:
    return [
        RubricCriterion(
            criterion_name="technical_correctness",
            criterion_type="correctness",
            weight=0.5,
            expected_concepts=["OOP", "design patterns"],
            star_component=False,
            scoring_guide="1.0 if OOP and design patterns mentioned.",
        ),
        RubricCriterion(
            criterion_name="relevance",
            criterion_type="relevance",
            weight=0.5,
            expected_concepts=["relevant experience"],
            star_component=False,
            scoring_guide="1.0 if experience is clearly relevant.",
        ),
    ]


@pytest.fixture
def variant_generator() -> VariantGenerator:
    return VariantGenerator(seed=42)
