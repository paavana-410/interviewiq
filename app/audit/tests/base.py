"""Base class for audit tests."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class AuditMetric:
    test_name: str
    metric_value: float
    passed: bool
    threshold: float
    details: dict = field(default_factory=dict)
    # When a variant is a no-op (e.g. degradation didn't change the text),
    # the test cannot be measured. Mark it invalid and EXCLUDE from the metric.
    # Never report an invalid variant as 0 — that would be a false pass/fail.
    invalid: bool = False
    invalid_reason: str = ""
    # Count of variants that were no-ops for this test run
    invalid_variant_count: int = 0


class AuditTest(abc.ABC):
    name: str

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> AuditMetric:
        """Execute the test and return a metric."""
