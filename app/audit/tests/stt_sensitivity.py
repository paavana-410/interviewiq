"""Audit Test 5 — STT Sensitivity.

Hypothesis: Scoring drift between a gold/human transcript and the Whisper
transcript of the same audio should be minimal.

Metric: stt_sensitivity = |score_gold - score_whisper|
Pass condition: metric <= threshold (default 0.08)
"""
from __future__ import annotations

from app.audit.tests.base import AuditMetric, AuditTest
from app.config import get_settings
from app.engines.content import ContentEngine, RubricCriterion


class STTSensitivityTest(AuditTest):
    name = "stt_sensitivity"

    def __init__(self, content_engine: ContentEngine, threshold: float | None = None):
        self._engine = content_engine
        self._threshold = threshold or get_settings().audit_stt_max_drift

    def run(
        self,
        gold_transcript: str,
        whisper_transcript: str,
        question_text: str,
        rubric_criteria: list[RubricCriterion],
    ) -> AuditMetric:
        gold_result = self._engine.analyze(
            transcript_text=gold_transcript,
            question_text=question_text,
            rubric_criteria=rubric_criteria,
        )
        whisper_result = self._engine.analyze(
            transcript_text=whisper_transcript,
            question_text=question_text,
            rubric_criteria=rubric_criteria,
        )

        drift = abs(gold_result.overall_score - whisper_result.overall_score)
        passed = drift <= self._threshold

        return AuditMetric(
            test_name=self.name,
            metric_value=drift,
            passed=passed,
            threshold=self._threshold,
            details={
                "gold_score": gold_result.overall_score,
                "whisper_score": whisper_result.overall_score,
                "drift": drift,
                "gold_transcript_preview": gold_transcript[:80],
                "whisper_transcript_preview": whisper_transcript[:80],
            },
        )
