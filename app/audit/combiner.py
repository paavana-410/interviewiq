"""Audit Combiner — aggregates all 5 test metrics into a single verdict.

All thresholds are read from settings and can be overridden per-run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.audit.tests.base import AuditMetric
from app.config import get_settings


@dataclass
class AuditReport:
    delivery_invariance_error: float | None
    content_sensitivity: float | None
    paraphrase_deviation: float | None
    repeatability_variance: float | None
    stt_sensitivity: float | None
    uncertainty_flag: bool
    all_tests_passed: bool
    details: dict = field(default_factory=dict)

    def to_db_dict(self) -> dict:
        return {
            "delivery_invariance_error": self.delivery_invariance_error,
            "content_sensitivity": self.content_sensitivity,
            "paraphrase_deviation": self.paraphrase_deviation,
            "repeatability_variance": self.repeatability_variance,
            "stt_sensitivity": self.stt_sensitivity,
            "uncertainty_flag": self.uncertainty_flag,
            "details": self.details,
        }


class AuditCombiner:
    """Combine individual AuditMetric results into an AuditReport."""

    def combine(self, metrics: list[AuditMetric]) -> AuditReport:
        s = get_settings()
        by_name = {m.test_name: m for m in metrics}

        def _val(name: str) -> float | None:
            return by_name[name].metric_value if name in by_name else None

        def _passed(name: str) -> bool:
            return by_name[name].passed if name in by_name else True

        delivery_invariance_error = _val("delivery_invariance")
        content_sensitivity = _val("content_sensitivity")
        paraphrase_deviation = _val("paraphrase_stability")
        repeatability_variance = _val("repeated_run")
        stt_sensitivity = _val("stt_sensitivity")

        all_passed = all(_passed(n) for n in by_name)
        uncertainty_flag = not all_passed

        details = {m.test_name: m.details for m in metrics}
        # Include pass/fail per test
        for m in metrics:
            details[m.test_name]["passed"] = m.passed
            details[m.test_name]["threshold"] = m.threshold

        return AuditReport(
            delivery_invariance_error=delivery_invariance_error,
            content_sensitivity=content_sensitivity,
            paraphrase_deviation=paraphrase_deviation,
            repeatability_variance=repeatability_variance,
            stt_sensitivity=stt_sensitivity,
            uncertainty_flag=uncertainty_flag,
            all_tests_passed=all_passed,
            details=details,
        )
