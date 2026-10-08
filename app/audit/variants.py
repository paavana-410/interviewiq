"""Variant Generator — creates controlled perturbations for audit tests.

Each procedure is seeded and documented so experiments are reproducible.

Procedures:
  1. filler_injection      — insert fillers at random word boundaries
  2. pause_insertion       — add artificial pause gaps in timestamps
  3. pace_change           — scale all timestamps by a factor (speed up / slow down)
  4. quality_degradation   — replace key concepts with vague/wrong text (text-level)
  5. paraphrase            — LLM-generated meaning-preserving rewrite (flagged for human review)
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any

from app.services.stt import WordTimestamp


# ─────────────────────────────────────────────────────────────────────────────
# Shared dataclass for a variant
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Variant:
    """A modified version of an original answer."""
    variant_type: str          # filler_injection | pause_insertion | pace_change |
                               # quality_degradation | paraphrase
    original_text: str
    variant_text: str
    original_words: list[WordTimestamp]
    variant_words: list[WordTimestamp]
    seed: int
    params: dict = field(default_factory=dict)
    # For paraphrase variants only
    needs_human_verification: bool = False
    human_verified: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class VariantGenerator:
    """Generates controlled perturbation variants.

    Procedures are seeded via `seed` for reproducibility.
    All parameters can be overridden for ablation studies.
    """

    DEFAULT_FILLERS = ["um", "uh", "like", "you know", "so", "basically"]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)

    def _fresh_rng(self, extra: int = 0) -> random.Random:
        """A sub-RNG seeded from master seed + offset for reproducibility."""
        return random.Random(self.seed + extra)

    # ── 1. Filler injection ───────────────────────────────────────────────────

    def filler_injection(
        self,
        words: list[WordTimestamp],
        filler_rate: float = 0.08,
        fillers: list[str] | None = None,
    ) -> Variant:
        """Insert filler words at random positions.

        filler_rate: probability of inserting a filler BEFORE each word.
        """
        rng = self._fresh_rng(1)
        filler_pool = fillers or self.DEFAULT_FILLERS
        new_words: list[WordTimestamp] = []

        for w in words:
            if rng.random() < filler_rate:
                filler_token = rng.choice(filler_pool)
                filler_dur = 0.25
                gap = 0.05
                filler_start = max(0.0, w.start - filler_dur - gap)
                filler_end = filler_start + filler_dur
                # Shift existing word end if needed (keep ordering)
                new_words.append(
                    WordTimestamp(
                        word=filler_token,
                        start=filler_start,
                        end=filler_end,
                        probability=1.0,
                    )
                )
            new_words.append(w)

        variant_text = " ".join(w.word for w in new_words)
        original_text = " ".join(w.word for w in words)

        return Variant(
            variant_type="filler_injection",
            original_text=original_text,
            variant_text=variant_text,
            original_words=words,
            variant_words=new_words,
            seed=self.seed,
            params={"filler_rate": filler_rate, "fillers": filler_pool},
        )

    # ── 2. Pause insertion ────────────────────────────────────────────────────

    def pause_insertion(
        self,
        words: list[WordTimestamp],
        num_pauses: int = 3,
        pause_duration_sec: float = 2.0,
    ) -> Variant:
        """Insert long artificial pauses at random word boundaries.

        Shifts all subsequent word timestamps to maintain consistency.
        """
        rng = self._fresh_rng(2)
        new_words = [copy.copy(w) for w in words]
        n = len(new_words)
        if n < 2:
            return self._identity_variant("pause_insertion", words)

        # Choose insertion points (sorted descending to shift correctly)
        positions = sorted(rng.sample(range(1, n), min(num_pauses, n - 1)), reverse=True)
        total_shift = 0.0

        for pos in positions:
            shift = pause_duration_sec
            for i in range(pos, len(new_words)):
                new_words[i] = WordTimestamp(
                    word=new_words[i].word,
                    start=new_words[i].start + shift,
                    end=new_words[i].end + shift,
                    probability=new_words[i].probability,
                )
            total_shift += shift

        original_text = " ".join(w.word for w in words)
        return Variant(
            variant_type="pause_insertion",
            original_text=original_text,
            variant_text=original_text,  # text unchanged
            original_words=words,
            variant_words=new_words,
            seed=self.seed,
            params={"num_pauses": num_pauses, "pause_duration_sec": pause_duration_sec},
        )

    # ── 3. Pace change (timestamp scaling) ───────────────────────────────────

    def pace_change(
        self,
        words: list[WordTimestamp],
        factor: float = 1.5,
    ) -> Variant:
        """Scale all timestamps by `factor` (>1 = slower, <1 = faster).

        Text is identical; only timing changes.
        """
        if not words:
            return self._identity_variant("pace_change", words)

        anchor = words[0].start
        new_words = [
            WordTimestamp(
                word=w.word,
                start=anchor + (w.start - anchor) * factor,
                end=anchor + (w.end - anchor) * factor,
                probability=w.probability,
            )
            for w in words
        ]
        original_text = " ".join(w.word for w in words)
        return Variant(
            variant_type="pace_change",
            original_text=original_text,
            variant_text=original_text,
            original_words=words,
            variant_words=new_words,
            seed=self.seed,
            params={"factor": factor},
        )

    # ── 4. Quality degradation ────────────────────────────────────────────────

    def quality_degradation(
        self,
        text: str,
        words: list[WordTimestamp],
        concepts_to_remove: list[str],
        replacement: str = "some stuff",
    ) -> Variant:
        """Replace key technical concepts with vague/incorrect text.

        Timestamps are rebuilt naively from the degraded text
        (uniform distribution, same total duration).
        """
        degraded = text
        for concept in concepts_to_remove:
            degraded = degraded.replace(concept, replacement)

        # Rebuild timestamps at same pace
        orig_duration = (words[-1].end - words[0].start) if words else 1.0
        degraded_tokens = degraded.split()
        n = len(degraded_tokens)
        dur_per_token = orig_duration / max(n, 1)
        start_t = words[0].start if words else 0.0
        new_words = [
            WordTimestamp(
                word=tok,
                start=start_t + i * dur_per_token,
                end=start_t + (i + 1) * dur_per_token - 0.05,
                probability=0.95,
            )
            for i, tok in enumerate(degraded_tokens)
        ]

        return Variant(
            variant_type="quality_degradation",
            original_text=text,
            variant_text=degraded,
            original_words=words,
            variant_words=new_words,
            seed=self.seed,
            params={"concepts_removed": concepts_to_remove, "replacement": replacement},
        )

    # ── 5. Paraphrase (LLM-generated) ─────────────────────────────────────────

    def paraphrase(
        self,
        text: str,
        words: list[WordTimestamp],
        llm_service: Any | None = None,
    ) -> Variant:
        """Generate a meaning-preserving paraphrase using an LLM.

        If no LLM is provided, returns a simple word-shuffle paraphrase
        (not meaning-preserving, for testing only).
        Sets needs_human_verification=True always.
        """
        if llm_service is not None:
            prompt = (
                "Rewrite the following answer in different wording but preserve "
                "all key technical content and meaning exactly. Return ONLY the "
                "rewritten text, no explanation.\n\n"
                f'"""\n{text}\n"""'
            )
            resp = llm_service.complete(prompt, temperature=0.3, max_tokens=512)
            paraphrased = resp.text.strip().strip('"').strip("'")
        else:
            # Naive fallback: move sentences around
            sentences = [s.strip() for s in text.replace(".", ".|").split("|") if s.strip()]
            rng = self._fresh_rng(5)
            rng.shuffle(sentences)
            paraphrased = " ".join(sentences)

        # Rebuild timestamps
        para_tokens = paraphrased.split()
        orig_duration = (words[-1].end - words[0].start) if words else 1.0
        dur_per_token = orig_duration / max(len(para_tokens), 1)
        start_t = words[0].start if words else 0.0
        new_words = [
            WordTimestamp(
                word=tok,
                start=start_t + i * dur_per_token,
                end=start_t + (i + 1) * dur_per_token - 0.05,
                probability=0.95,
            )
            for i, tok in enumerate(para_tokens)
        ]

        return Variant(
            variant_type="paraphrase",
            original_text=text,
            variant_text=paraphrased,
            original_words=words,
            variant_words=new_words,
            seed=self.seed,
            params={"llm_used": llm_service is not None},
            needs_human_verification=True,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _identity_variant(self, vtype: str, words: list[WordTimestamp]) -> Variant:
        text = " ".join(w.word for w in words)
        return Variant(
            variant_type=vtype,
            original_text=text,
            variant_text=text,
            original_words=words,
            variant_words=list(words),
            seed=self.seed,
        )
