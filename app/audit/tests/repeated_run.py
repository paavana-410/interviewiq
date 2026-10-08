"""Audit Test 4 — Repeated-Run Consistency.

Hypothesis: Running the same input N times should produce near-identical scores.

Metric: repeatability_variance = variance of N score runs
Pass condition: variance <= threshold (default 0.02)
"""
from __future__ import annotations

import statistics

from app.audit.tests.base import AuditMetric, AuditTest
from app.config import get_settings
from app.engines.content import ContentEngine, RubricCriterion


class RepeatedRunTest(AuditTest):
    name = "repeated_run"

    def __init__(
        self,
        content_engine: ContentEngine,
        n_runs: int | None = None,
        threshold: float | None = None,
    ):
        self._engine = content_engine
        settings = get_settings()
        self._n_runs = n_runs or settings.audit_repeated_runs
        self._threshold = threshold or settings.audit_repeatability_max_variance

    def run(
        self,
        transcript_text: str,
        question_text: str,
        rubric_criteria: list[RubricCriterion],
    ) -> AuditMetric:
        scores: list[float] = []
        for i in range(self._n_runs):
            result = self._engine.analyze(
                transcript_text=transcript_text,
                question_text=question_text,
                rubric_criteria=rubric_criteria,
            )
            scores.append(result.overall_score)

        variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        passed = variance <= self._threshold

        return AuditMetric(
            test_name=self.name,
            metric_value=variance,
            passed=passed,
            threshold=self._threshold,
            details={
                "scores": scores,
                "mean": statistics.mean(scores),
                "variance": variance,
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "n_runs": self._n_runs,
            },
        )
