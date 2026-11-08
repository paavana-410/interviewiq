"""Audit Test 6 — Known-Groups Validity (strict monotonicity).

Hypothesis: For items that have hand-written strong / medium / weak answers,
the judge's overall scores must STRICTLY order strong > medium > weak, with
each adjacent gap at least a configurable margin (default 0.1).

Ties or equal scores across tiers FAIL. This is a known-groups validity
check (a standard construct-validity method): if the judge cannot distinguish
a strong answer from a weak one with a clear margin, the rubric scoring has
no discriminant validity.

Pass condition: all three tiers present AND
  (strong - medium) >= margin AND (medium - weak) >= margin.
"""
from __future__ import annotations

from app.audit.tests.base import AuditMetric, AuditTest
from app.config import get_settings
from app.engines.content import ContentEngine, RubricCriterion


class KnownGroupsTest(AuditTest):
    name = "known_groups"

    def __init__(self, content_engine: ContentEngine, min_margin: float | None = None):
        self._engine = content_engine
        self._margin = (
            min_margin if min_margin is not None
            else get_settings().known_groups_min_margin
        )

    def run(
        self,
        question_text: str,
        rubric_criteria: list[RubricCriterion],
        strong_text: str | None,
        medium_text: str | None,
        weak_text: str | None,
    ) -> AuditMetric:
        """Score each tier and check strict monotonicity with margin."""
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

        all_present = len(scores) == 3

        # ── Strict monotonicity with margin ──────────────────────────────────
        # strong > medium >= margin gap, medium > weak >= margin gap.
        # Ties (equal scores) FAIL because the gap would be 0 < margin.
        gaps: dict[str, float | None] = {"strong_medium": None, "medium_weak": None}
        ordering_ok = True
        ordering_reason = ""

        if "strong" in scores and "medium" in scores:
            gap = scores["strong"] - scores["medium"]
            gaps["strong_medium"] = gap
            if gap < self._margin:
                ordering_ok = False
                ordering_reason = (
                    f"strong({scores['strong']:.3f}) - medium({scores['medium']:.3f}) "
                    f"= {gap:.3f} < margin {self._margin:.3f}"
                )
        if "medium" in scores and "weak" in scores:
            gap = scores["medium"] - scores["weak"]
            gaps["medium_weak"] = gap
            if gap < self._margin:
                ordering_ok = False
                reason = (
                    f"medium({scores['medium']:.3f}) - weak({scores['weak']:.3f}) "
                    f"= {gap:.3f} < margin {self._margin:.3f}"
                )
                ordering_reason = f"{ordering_reason}; {reason}" if ordering_reason else reason

        # Pass requires all three tiers present AND strict ordering with margin
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
                "ordering_reason": ordering_reason or None,
                "gaps": gaps,
                "min_margin": self._margin,
                "tier_details": tier_details,
            },
            invalid=(not all_present),
            invalid_reason=(
                "only %d of 3 tiers present — known-groups check needs "
                "strong, medium AND weak answers" % len(scores)
            ) if not all_present else "",
        )
