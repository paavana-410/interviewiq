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
    # Source of the degraded text: "auto" (generated) or "handwritten" (from file)
    source: str = "auto"
    # Fine-grained degradation method (quality_degradation only):
    #   handwritten_tier | annotated_sentence_removal |
    #   substring_sentence_removal | fallback_drop_last_sentence | no_op
    degradation_method: str = ""

    @property
    def text_differs(self) -> bool:
        """True if the variant text differs from the original text."""
        return self.variant_text.strip() != self.original_text.strip()

    @property
    def timing_differs(self) -> bool:
        """True if the variant word timestamps differ from the original."""
        if len(self.variant_words) != len(self.original_words):
            return True
        for a, b in zip(self.variant_words, self.original_words):
            if abs(a.start - b.start) > 1e-6 or abs(a.end - b.end) > 1e-6:
                return True
        return False

    @property
    def differs_from_original(self) -> bool:
        """True if either text or timing differs."""
        return self.text_differs or self.timing_differs


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
        *,
        concept_sentences: dict[str, list[int]] | None = None,
        hand_written: str | None = None,
    ) -> Variant:
        """Degrade answer quality by removing content that carries target concepts.

        Two modes:
          - hand_written: if a hand-written degraded answer is provided (from
            answer_medium.txt / answer_weak.txt), use it directly. This is the
            preferred mode because it is a real, human-judged degradation.
          - auto: otherwise, remove whole SENTENCES that carry the target
            concepts. `concept_sentences` maps each concept -> list of sentence
            indices (0-based) that carry it. If provided, those sentences are
            deleted. If not provided, fall back to removing any sentence that
            contains the concept string (case-insensitive substring match).

        This replaces the old string-replace approach which was a no-op when
        the concept name did not appear verbatim in the text.
        """
        original_text = text
        method = ""

        if hand_written is not None and hand_written.strip():
            degraded = hand_written.strip()
            source = "handwritten"
            method = "handwritten_tier"
        else:
            degraded, method = self._sentence_based_degradation(
                text, concepts_to_remove, concept_sentences
            )
            source = "auto"

        # Rebuild timestamps at same pace from the degraded tokens
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
            original_text=original_text,
            variant_text=degraded,
            original_words=words,
            variant_words=new_words,
            seed=self.seed,
            params={
                "concepts_removed": concepts_to_remove,
                "replacement": replacement,
                "concept_sentences": concept_sentences,
                "source": source,
                "degradation_method": method,
            },
            source=source,
            degradation_method=method,
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on . ! ? while keeping the delimiter."""
        import re
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _sentence_based_degradation(
        self,
        text: str,
        concepts_to_remove: list[str],
        concept_sentences: dict[str, list[int]] | None,
    ) -> tuple[str, str]:
        """Remove sentences that carry the target concepts.

        Returns (degraded_text, method) where method is one of:
          annotated_sentence_removal — concept_sentences mapping was used
          substring_sentence_removal   — fallback substring match removed sentences
          fallback_drop_last_sentence  — no concept matched; dropped last sentence
          no_op                        — single sentence, nothing could be removed
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return text, "no_op"

        # Build the set of sentence indices to remove
        remove_indices: set[int] = set()
        method = "no_op"
        if concept_sentences:
            for concept in concepts_to_remove:
                for idx in concept_sentences.get(concept, []):
                    if 0 <= idx < len(sentences):
                        remove_indices.add(idx)
            if remove_indices:
                method = "annotated_sentence_removal"
        else:
            # Fallback: remove any sentence containing the concept (substring, ci)
            for i, sent in enumerate(sentences):
                sent_lower = sent.lower()
                for concept in concepts_to_remove:
                    if concept.lower() in sent_lower:
                        remove_indices.add(i)
                        break
            if remove_indices:
                method = "substring_sentence_removal"

        if not remove_indices:
            # No sentence matched any concept — degrade is a no-op.
            # As a last resort, drop the LAST sentence (removes some content).
            if len(sentences) > 1:
                remove_indices = {len(sentences) - 1}
                method = "fallback_drop_last_sentence"
            else:
                # Single sentence and no match → genuinely a no-op.
                return text, "no_op"

        kept = [s for i, s in enumerate(sentences) if i not in remove_indices]
        return (" ".join(kept) if kept else text), method

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
