"""Audit Test 2 — Content Sensitivity.

Hypothesis: When answer quality degrades (key concepts removed), the content
score MUST drop. Delivery metrics must remain unchanged.

Metric: content_sensitivity = original_score - degraded_score  (should be >= min threshold)
Pass condition: content drop >= threshold (default 0.15)

Variant validity: the degraded text MUST differ from the original. If a
variant is a no-op (degradation didn't change the text — e.g. the concept
string was never present verbatim), it is marked invalid and EXCLUDED from
the metric. It is never reported as content_drop=0.
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
        """Compare original vs quality-degraded variants."""
        orig_content = self._engine_analyze(original_transcript, question_text, rubric_criteria)
        orig_delivery = self._delivery.analyze(degraded_variants[0].original_words)
        orig_score = orig_content.overall_score

        sensitivity_values: list[float] = []
        variant_details: list[dict] = []
        invalid_count = 0

        for v in degraded_variants:
            # ── Variant validity check: degraded text MUST differ from original ──
            if not v.text_differs:
                invalid_count += 1
                variant_details.append({
                    "variant_params": v.params,
                    "original_score": orig_score,
                    "degraded_score": None,
                    "content_drop": None,
                    "INVALID": "degraded text is identical to original — "
                               "degradation was a no-op, this variant excluded",
                    "delivery_wpm_delta": None,
                })
                continue

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

        # If ALL variants were invalid, the test cannot be measured.
        if not sensitivity_values:
            return AuditMetric(
                test_name=self.name,
                metric_value=0.0,
                passed=False,
                threshold=self._threshold,
                details={
                    "original_score": orig_score,
                    "variants": variant_details,
                    "min_sensitivity": None,
                },
                invalid=True,
                invalid_reason=(
                    f"all {invalid_count} degraded variant(s) were no-ops "
                    f"(degraded text == original). content_sensitivity cannot "
                    f"be measured for this item."
                ),
                invalid_variant_count=invalid_count,
            )

        min_sensitivity = min(sensitivity_values)
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
                "invalid_variant_count": invalid_count,
            },
            invalid=False,
            invalid_variant_count=invalid_count,
        )

    def _engine_analyze(self, text, question, rubric):
        return self._content.analyze(
            transcript_text=text,
            question_text=question,
            rubric_criteria=rubric,
        )
