"""Speech-to-Text service interface + faster-whisper implementation.

The interface (STTService) is the only thing the rest of the codebase imports.
Swap implementations via config without touching callers.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WordTimestamp:
    word: str
    start: float   # seconds
    end: float     # seconds
    probability: float  # ASR confidence [0,1]


@dataclass
class TranscriptResult:
    text: str
    words: list[WordTimestamp] = field(default_factory=list)
    language: str = "en"
    avg_confidence: float = 1.0
    model_name: str = "unknown"

    @property
    def duration_sec(self) -> float:
        if not self.words:
            return 0.0
        return self.words[-1].end

    def to_word_timestamps_json(self) -> list[dict]:
        return [
            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
            for w in self.words
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class STTService(abc.ABC):
    """All STT implementations must satisfy this interface."""

    @abc.abstractmethod
    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        """Transcribe an audio file and return word-level timestamps."""


# ─────────────────────────────────────────────────────────────────────────────
# faster-whisper implementation
# ─────────────────────────────────────────────────────────────────────────────

class WhisperSTTService(STTService):
    """faster-whisper backed STT with word-level timestamps."""

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        # Lazy import so tests that mock STT don't need faster-whisper installed
        from faster_whisper import WhisperModel  # type: ignore

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._model_name = f"faster-whisper-{model_size}"

    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        segments, info = self._model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language="en",
        )

        words: list[WordTimestamp] = []
        full_text_parts: list[str] = []

        for segment in segments:
            full_text_parts.append(segment.text.strip())
            if segment.words:
                for w in segment.words:
                    words.append(
                        WordTimestamp(
                            word=w.word.strip(),
                            start=float(w.start),
                            end=float(w.end),
                            probability=float(w.probability),
                        )
                    )

        text = " ".join(full_text_parts)
        avg_conf = (
            sum(w.probability for w in words) / len(words) if words else 1.0
        )

        return TranscriptResult(
            text=text,
            words=words,
            language=info.language,
            avg_confidence=avg_conf,
            model_name=self._model_name,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mock implementation (for tests + CI without GPU/model download)
# ─────────────────────────────────────────────────────────────────────────────

class MockSTTService(STTService):
    """Returns a deterministic transcript for any audio file.
    Used in tests and when no Whisper model is available.
    """

    def __init__(self, fixed_transcript: str | None = None):
        self._fixed = fixed_transcript or (
            "I have strong experience with object-oriented programming "
            "and I have worked on several large-scale projects um you know "
            "I think I am a good fit for this role basically"
        )

    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        tokens = self._fixed.split()
        words: list[WordTimestamp] = []
        t = 0.0
        for token in tokens:
            dur = 0.35 + 0.02 * len(token)
            words.append(WordTimestamp(word=token, start=t, end=t + dur, probability=0.95))
            t += dur + 0.05  # small gap between words

        return TranscriptResult(
            text=self._fixed,
            words=words,
            language="en",
            avg_confidence=0.95,
            model_name="mock-stt",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_stt_service() -> STTService:
    from app.config import get_settings
    s = get_settings()
    if s.llm_provider == "mock":
        return MockSTTService()
    return WhisperSTTService(
        model_size=s.whisper_model_size,
        device=s.whisper_device,
        compute_type=s.whisper_compute_type,
    )
