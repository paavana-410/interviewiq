"""Audit Test 2 — Content Sensitivity.

Hypothesis: When answer quality degrades (key concepts removed), the content
score MUST drop. Delivery metrics must remain unchanged.

Metrics:
  content_sensitivity = original_score - degraded_score  (should be > min threshold)
  delivery_invariance_check = |delivery_metric_original - delivery_metric_degraded|
                               (should be ~0)
Pass condition: content drop >= threshold (default 0.15)
"""
from __future__ import annotations

from app.audit.tests.base import AuditMetric, AuditTest
from app.audit.variants import Variant
from app.config import get_settings
from app.engines.content import ContentEngine, RubricCriterion
from app.engines.delivery import DeliveryEngine


class ContentSensitivityTest(AuditTest):
    name = "content_sensitivity"

    def __init__(
        self,
        content_engine: ContentEngine,
        delivery_engine: DeliveryEngine,
        threshold: float | None = None,
    ):
        self._content = content_engine
        self._delivery = delivery_engine
        self._threshold = threshold or get_settings().audit_content_sensitivity_min

    def run(
        self,
        original_transcript: str,
        question_text: str,
        rubric_criteria: list[RubricCriterion],
        degraded_variants: list[Variant],
    ) -> AuditMetric:
        """Compare original vs quality-degraded variants.

        degraded_variants must be quality_degradation type.
        """
        orig_content = self._engine_analyze(original_transcript, question_text, rubric_criteria)
        orig_delivery = self._delivery.analyze(degraded_variants[0].original_words)
        orig_score = orig_content.overall_score

        sensitivity_values: list[float] = []
        variant_details: list[dict] = []

        for v in degraded_variants:
            deg_content = self._engine_analyze(v.variant_text, question_text, rubric_criteria)
            deg_delivery = self._delivery.analyze(v.variant_words)

            content_drop = orig_score - deg_content.overall_score
            delivery_delta = abs(orig_delivery.speaking_rate_wpm - deg_delivery.speaking_rate_wpm)

            sensitivity_values.append(content_drop)
            variant_details.append({
                "variant_params": v.params,
                "original_score": orig_score,
                "degraded_score": deg_content.overall_score,
                "content_drop": content_drop,
                "delivery_wpm_delta": delivery_delta,
            })

        min_sensitivity = min(sensitivity_values) if sensitivity_values else 0.0
        passed = min_sensitivity >= self._threshold

        return AuditMetric(
            test_name=self.name,
            metric_value=min_sensitivity,
            passed=passed,
            threshold=self._threshold,
            details={
                "original_score": orig_score,
                "variants": variant_details,
                "min_sensitivity": min_sensitivity,
            },
        )

    def _engine_analyze(self, text, question, rubric):
        return self._content.analyze(
            transcript_text=text,
            question_text=question,
            rubric_criteria=rubric,
        )
