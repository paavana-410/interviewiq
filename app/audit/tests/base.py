"""Base class for audit tests."""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class AuditMetric:
    test_name: str
    metric_value: float
    passed: bool
    threshold: float
    details: dict


class AuditTest(abc.ABC):
    name: str

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> AuditMetric:
        """Execute the test and return a metric."""
