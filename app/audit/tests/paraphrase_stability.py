"""Audit Test 3 — Paraphrase Stability.

Hypothesis: Meaning-preserving paraphrases should receive similar content scores.

Metric: paraphrase_deviation = max |score_original - score_paraphrase_i|
Pass condition: deviation <= threshold (default 0.10)
"""
from __future__ import annotations

from app.audit.tests.base import AuditMetric, AuditTest
from app.audit.variants import Variant
from app.config import get_settings
from app.engines.content import ContentEngine, RubricCriterion


class ParaphraseStabilityTest(AuditTest):
    name = "paraphrase_stability"

    def __init__(self, content_engine: ContentEngine, threshold: float | None = None):
        self._engine = content_engine
        self._threshold = threshold or get_settings().audit_paraphrase_max_deviation

    def run(
        self,
        original_transcript: str,
        question_text: str,
        rubric_criteria: list[RubricCriterion],
        paraphrase_variants: list[Variant],
    ) -> AuditMetric:
        orig_result = self._engine.analyze(
            transcript_text=original_transcript,
            question_text=question_text,
            rubric_criteria=rubric_criteria,
        )
        orig_score = orig_result.overall_score

        deviations: list[float] = []
        variant_details: list[dict] = []

        for v in paraphrase_variants:
            para_result = self._engine.analyze(
                transcript_text=v.variant_text,
                question_text=question_text,
                rubric_criteria=rubric_criteria,
            )
            deviation = abs(orig_score - para_result.overall_score)
            deviations.append(deviation)
            variant_details.append({
                "variant_text_preview": v.variant_text[:80],
                "human_verified": v.human_verified,
                "original_score": orig_score,
                "paraphrase_score": para_result.overall_score,
                "deviation": deviation,
            })

        max_deviation = max(deviations) if deviations else 0.0
        passed = max_deviation <= self._threshold

        return AuditMetric(
            test_name=self.name,
            metric_value=max_deviation,
            passed=passed,
            threshold=self._threshold,
            details={
                "original_score": orig_score,
                "variants": variant_details,
                "max_deviation": max_deviation,
            },
        )
