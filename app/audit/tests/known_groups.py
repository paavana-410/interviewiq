"""Audit Test 6 — Known-Groups Validity (monotonicity).

Hypothesis: For items that have hand-written strong / medium / weak answers,
the judge's overall scores should order strong > medium > weak.

This is a known-groups validity check (a standard construct-validity method):
if the judge cannot distinguish a strong answer from a weak one, the rubric
scoring has no discriminant validity.

Inputs: the three answer texts (strong, medium, weak). Missing tiers are
skipped — the check only runs on tiers that are present.

Metric: 1.0 if the ordering is correct (strong > medium > weak, allowing
ties between adjacent tiers to be reported separately), else 0.0.
Pass condition: metric == 1.0 AND all three tiers present.
"""
from __future__ import annotations

from app.audit.tests.base import AuditMetric, AuditTest
from app.engines.content import ContentEngine, RubricCriterion


class KnownGroupsTest(AuditTest):
    name = "known_groups"

    def __init__(self, content_engine: ContentEngine):
        self._engine = content_engine

    def run(
        self,
        question_text: str,
        rubric_criteria: list[RubricCriterion],
        strong_text: str | None,
        medium_text: str | None,
        weak_text: str | None,
    ) -> AuditMetric:
        """Score each tier and check monotonicity."""
        present: list[tuple[str, str]] = []
        if strong_text and strong_text.strip():
            present.append(("strong", strong_text.strip()))
        if medium_text and medium_text.strip():
            present.append(("medium", medium_text.strip()))
        if weak_text and weak_text.strip():
            present.append(("weak", weak_text.strip()))

        scores: dict[str, float] = {}
        for tier, text in present:
            res = self._engine.analyze(
                transcript_text=text,
                question_text=question_text,
                rubric_criteria=rubric_criteria,
            )
            scores[tier] = res.overall_score

        tier_details = [
            {"tier": t, "score": scores.get(t), "present": t in scores}
            for t in ("strong", "medium", "weak")
        ]

        # Determine if the ordering is correct
        ordering_ok = True
        if "strong" in scores and "medium" in scores:
            if not (scores["strong"] >= scores["medium"]):
                ordering_ok = False
        if "medium" in scores and "weak" in scores:
            if not (scores["medium"] >= scores["weak"]):
                ordering_ok = False
        if "strong" in scores and "weak" in scores:
            if not (scores["strong"] >= scores["weak"]):
                ordering_ok = False

        # Pass requires all three tiers present AND correct ordering
        all_present = len(scores) == 3
        passed = all_present and ordering_ok

        return AuditMetric(
            test_name=self.name,
            metric_value=1.0 if ordering_ok else 0.0,
            passed=passed,
            threshold=1.0,
            details={
                "scores": scores,
                "tiers_present": list(scores.keys()),
                "all_tiers_present": all_present,
                "ordering_ok": ordering_ok,
                "tier_details": tier_details,
            },
            invalid=(not all_present),
            invalid_reason=(
                "only %d of 3 tiers present — known-groups check needs "
                "strong, medium AND weak answers" % len(scores)
            ) if not all_present else "",
        )
