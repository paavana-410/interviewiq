"""Delivery Engine — fully deterministic, no LLM.

INPUT:  list[WordTimestamp]  (timing data only)
OUTPUT: DeliveryResult

STRUCTURAL ISOLATION GUARANTEE:
  - This module has no import from app.engines.content
  - This module has no import from app.services.llm
  - This module does NOT receive rubric, question text, or transcript text
  - It receives ONLY word-level timestamps

The test_input_isolation test in tests/test_isolation.py enforces this.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from app.config import get_settings
from app.services.stt import WordTimestamp


# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PauseInfo:
    start: float
    end: float
    duration: float
    is_long: bool


@dataclass
class FillerOccurrence:
    word: str
    position_index: int   # index in the word list
    start: float


@dataclass
class RepetitionOccurrence:
    phrase: str
    first_index: int
    repeat_index: int


@dataclass
class DeliveryResult:
    """All delivery metrics derived purely from word timestamps."""
    speaking_rate_wpm: float             # words per minute of speech
    duration_sec: float
    pause_count: int
    long_pause_count: int
    pauses: list[PauseInfo] = field(default_factory=list)
    filler_count: int = 0
    filler_density: float = 0.0          # fillers per 100 words
    fillers: list[FillerOccurrence] = field(default_factory=list)
    repetition_count: int = 0
    repetitions: list[RepetitionOccurrence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "speaking_rate_wpm": self.speaking_rate_wpm,
            "duration_sec": self.duration_sec,
            "pause_count": self.pause_count,
            "long_pause_count": self.long_pause_count,
            "filler_count": self.filler_count,
            "filler_density": self.filler_density,
            "repetition_count": self.repetition_count,
            "detail": {
                "pauses": [
                    {"start": p.start, "end": p.end, "duration": p.duration, "is_long": p.is_long}
                    for p in self.pauses
                ],
                "fillers": [
                    {"word": f.word, "position_index": f.position_index, "start": f.start}
                    for f in self.fillers
                ],
                "repetitions": [
                    {
                        "phrase": r.phrase,
                        "first_index": r.first_index,
                        "repeat_index": r.repeat_index,
                    }
                    for r in self.repetitions
                ],
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class DeliveryEngine:
    """Compute delivery metrics from word-level timestamps.

    Parameters come from Settings but can be overridden for testing.
    """

    def __init__(
        self,
        long_pause_threshold_sec: float | None = None,
        filler_words: list[str] | None = None,
    ):
        settings = get_settings()
        self.long_pause_threshold = (
            long_pause_threshold_sec
            if long_pause_threshold_sec is not None
            else settings.long_pause_threshold_sec
        )
        self.filler_set: set[str] = set(
            filler_words if filler_words is not None else settings.filler_words_list
        )

    # ── public entry point ────────────────────────────────────────────────────

    def analyze(self, words: list[WordTimestamp]) -> DeliveryResult:
        """Compute all delivery metrics from a word timestamp list.

        This method intentionally accepts ONLY timing data.
        Any attempt to pass transcript text or rubric info will not compile
        because the type signature rejects it.
        """
        if not words:
            return DeliveryResult(
                speaking_rate_wpm=0.0,
                duration_sec=0.0,
                pause_count=0,
                long_pause_count=0,
            )

        duration = words[-1].end - words[0].start
        word_count = len(words)
        speaking_rate = (word_count / duration * 60.0) if duration > 0 else 0.0

        pauses = self._detect_pauses(words)
        short_pauses = [p for p in pauses if not p.is_long]
        long_pauses = [p for p in pauses if p.is_long]

        fillers = self._detect_fillers(words)
        filler_density = (len(fillers) / word_count * 100.0) if word_count > 0 else 0.0

        repetitions = self._detect_repetitions(words)

        return DeliveryResult(
            speaking_rate_wpm=round(speaking_rate, 2),
            duration_sec=round(duration, 2),
            pause_count=len(pauses),
            long_pause_count=len(long_pauses),
            pauses=pauses,
            filler_count=len(fillers),
            filler_density=round(filler_density, 2),
            fillers=fillers,
            repetition_count=len(repetitions),
            repetitions=repetitions,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _detect_pauses(self, words: list[WordTimestamp]) -> list[PauseInfo]:
        """Detect inter-word gaps as pauses."""
        pauses: list[PauseInfo] = []
        for i in range(1, len(words)):
            gap = words[i].start - words[i - 1].end
            if gap > 0.25:  # ignore gaps < 250ms (natural articulation)
                pauses.append(
                    PauseInfo(
                        start=words[i - 1].end,
                        end=words[i].start,
                        duration=round(gap, 3),
                        is_long=gap >= self.long_pause_threshold,
                    )
                )
        return pauses

    def _detect_fillers(self, words: list[WordTimestamp]) -> list[FillerOccurrence]:
        """Detect filler words by matching against the configured filler set.

        Multi-word fillers (e.g. "you know") are matched using a sliding window.
        """
        fillers: list[FillerOccurrence] = []
        # Sort filler phrases by length desc so longer matches win
        sorted_fillers = sorted(self.filler_set, key=lambda x: len(x.split()), reverse=True)

        lower_words = [w.word.lower().strip(r".,!?;:") for w in words]
        matched_indices: set[int] = set()

        for filler_phrase in sorted_fillers:
            filler_tokens = filler_phrase.split()
            n = len(filler_tokens)
            for i in range(len(lower_words) - n + 1):
                if i in matched_indices:
                    continue
                if lower_words[i : i + n] == filler_tokens:
                    fillers.append(
                        FillerOccurrence(
                            word=filler_phrase,
                            position_index=i,
                            start=words[i].start,
                        )
                    )
                    for j in range(n):
                        matched_indices.add(i + j)

        fillers.sort(key=lambda f: f.position_index)
        return fillers

    def _detect_repetitions(
        self, words: list[WordTimestamp], window: int = 2
    ) -> list[RepetitionOccurrence]:
        """Detect IMMEDIATE word/bigram repetitions only.

        A repetition is flagged only when the same word/bigram appears in
        consecutive positions (gap <= window words). This matches real-world
        disfluency ("I I think", "the the project") without flagging normal
        re-use of common words across a long answer.
        """
        repetitions: list[RepetitionOccurrence] = []
        lower_words = [w.word.lower().strip(".,!?;:") for w in words]

        # Unigrams: flag only if word[i] == word[i-1] (immediate stutter)
        for i in range(1, len(lower_words)):
            if lower_words[i] == lower_words[i - 1]:
                phrase = lower_words[i]
                # Don't double-count chains (report first pair only per run)
                if not repetitions or repetitions[-1].phrase != phrase or repetitions[-1].repeat_index != i - 1:
                    repetitions.append(
                        RepetitionOccurrence(phrase=phrase, first_index=i - 1, repeat_index=i)
                    )

        # Bigrams: flag only if bigram[i:i+2] == bigram[i-2:i] (immediate bigram repeat)
        for i in range(2, len(lower_words) - 1):
            bigram_cur = (lower_words[i], lower_words[i + 1])
            bigram_prev = (lower_words[i - 2], lower_words[i - 1])
            if bigram_cur == bigram_prev:
                phrase = " ".join(bigram_cur)
                if not any(r.phrase == phrase and r.repeat_index == i for r in repetitions):
                    repetitions.append(
                        RepetitionOccurrence(phrase=phrase, first_index=i - 2, repeat_index=i)
                    )

        return repetitions
