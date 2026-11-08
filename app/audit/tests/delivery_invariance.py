"""Audit Test 1 — Delivery Invariance.

Hypothesis: If only delivery changes (fillers/pauses/pace) but content stays
identical, the content engine score must NOT change significantly.

Metric: delivery_invariance_error = |content_score_original - content_score_variant|
Pass condition: metric <= threshold (default 0.08)
"""
from __future__ import annotations

from app.audit.tests.base import AuditMetric, AuditTest
from app.audit.variants import Variant
from app.config import get_settings
from app.engines.content import ContentEngine, ContentResult, RubricCriterion
from app.services.stt import WordTimestamp


class DeliveryInvarianceTest(AuditTest):
    name = "delivery_invariance"

    def __init__(self, content_engine: ContentEngine, threshold: float | None = None):
        self._engine = content_engine
        self._threshold = threshold or get_settings().audit_delivery_invariance_max

    def run(
        self,
        original_transcript: str,
        question_text: str,
        rubric_criteria: list[RubricCriterion],
        delivery_variants: list[Variant],
    ) -> AuditMetric:
        """Evaluate original + delivery-only variants, compute score delta.

        delivery_variants must all be filler_injection | pause_insertion | pace_change
        variants — i.e., same TEXT as original.
        """
        original_result = self._engine.analyze(
            transcript_text=original_transcript,
            question_text=question_text,
            rubric_criteria=rubric_criteria,
        )
        orig_score = original_result.overall_score

        deltas: list[float] = []
        variant_details: list[dict] = []
        invalid_count = 0

        for v in delivery_variants:
            # ── Variant validity: a delivery-only variant must actually change
            # the timing. If timing didn't change, the test is vacuous for it.
            if not v.timing_differs:
                invalid_count += 1
                variant_details.append({
                    "variant_type": v.variant_type,
                    "variant_params": v.params,
                    "original_score": orig_score,
                    "variant_score": None,
                    "delta": None,
                    "INVALID": "timing did not change — variant is a no-op",
                })
                continue

            variant_result = self._engine.analyze(
                transcript_text=v.variant_text,
                question_text=question_text,
                rubric_criteria=rubric_criteria,
            )
            delta = abs(orig_score - variant_result.overall_score)
            deltas.append(delta)
            variant_details.append({
                "variant_type": v.variant_type,
                "variant_params": v.params,
                "original_score": orig_score,
                "variant_score": variant_result.overall_score,
                "delta": delta,
            })

        if not deltas:
            return AuditMetric(
                test_name=self.name,
                metric_value=0.0,
                passed=False,
                threshold=self._threshold,
                details={
                    "original_score": orig_score,
                    "variants": variant_details,
                    "max_delta": None,
                },
                invalid=True,
                invalid_reason=(
                    f"all {invalid_count} delivery variant(s) did not change timing."
                ),
                invalid_variant_count=invalid_count,
            )

        max_delta = max(deltas)
        passed = max_delta <= self._threshold

        return AuditMetric(
            test_name=self.name,
            metric_value=max_delta,
            passed=passed,
            threshold=self._threshold,
            details={
                "original_score": orig_score,
                "variants": variant_details,
                "max_delta": max_delta,
                "invalid_variant_count": invalid_count,
            },
            invalid_variant_count=invalid_count,
        )
