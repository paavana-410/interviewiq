"""Acoustic proctoring service — multi-speaker & background voice detection.

Uses deterministic signal processing (autocorrelation pitch distribution &
inter-word energy analysis). Fast (<80ms), runs on CPU with zero heavy dependencies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AcousticProctorResult:
    multiple_speakers_detected: bool
    background_whisper_detected: bool
    primary_pitch_hz: float
    pitch_variance_hz: float
    inter_word_speech_detected: bool
    flags_summary: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "multiple_speakers": self.multiple_speakers_detected,
            "background_whisper": self.background_whisper_detected,
            "primary_pitch_hz": self.primary_pitch_hz,
            "pitch_variance_hz": self.pitch_variance_hz,
            "inter_word_speech": self.inter_word_speech_detected,
            "flags": self.flags_summary,
            "clean": len(self.flags_summary) == 0,
        }


def _load_audio_resampled(audio_path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Decode audio file to 16kHz mono float32 array using PyAV."""
    import av
    container = av.open(str(audio_path))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
    frames: list[np.ndarray] = []
    try:
        for frame in container.decode(audio=0):
            for r_frame in resampler.resample(frame):
                frames.append(r_frame.to_ndarray())
    finally:
        container.close()

    if not frames:
        return np.zeros(0, dtype=np.float32), target_sr

    arr = np.concatenate(frames, axis=-1).astype(np.float32) / 32768.0
    return arr.flatten(), target_sr


def analyze_acoustic_integrity(
    audio_path: str | Path | None,
    word_timestamps: list[dict[str, Any]] | None = None,
) -> AcousticProctorResult:
    """Analyze audio for multi-speaker pitch bimodality and pause-interval vocalization."""
    if not audio_path or not Path(audio_path).exists():
        return AcousticProctorResult(
            multiple_speakers_detected=False,
            background_whisper_detected=False,
            primary_pitch_hz=0.0,
            pitch_variance_hz=0.0,
            inter_word_speech_detected=False,
            flags_summary=[],
        )

    try:
        audio, sr = _load_audio_resampled(audio_path, target_sr=16000)
    except Exception as e:
        logger.warning("Could not decode audio for proctoring: %s", e)
        return AcousticProctorResult(
            multiple_speakers_detected=False,
            background_whisper_detected=False,
            primary_pitch_hz=0.0,
            pitch_variance_hz=0.0,
            inter_word_speech_detected=False,
            flags_summary=[],
        )

    if len(audio) < sr * 0.5:  # less than 0.5s of audio
        return AcousticProctorResult(
            multiple_speakers_detected=False,
            background_whisper_detected=False,
            primary_pitch_hz=0.0,
            pitch_variance_hz=0.0,
            inter_word_speech_detected=False,
            flags_summary=[],
        )

    # 1. Pitch Tracking via Normalized Autocorrelation
    frame_len = int(sr * 0.04)  # 40ms
    hop = int(sr * 0.02)        # 20ms
    min_lag = int(sr / 360)     # ~44 samples (360 Hz)
    max_lag = int(sr / 70)      # ~228 samples (70 Hz)

    pitches: list[float] = []
    for i in range(0, len(audio) - frame_len, hop):
        frame = audio[i : i + frame_len]
        energy = np.sum(frame ** 2)
        if energy < 0.003:
            continue
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(frame) - 1 :]
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            continue
        peak_idx = int(np.argmax(search)) + min_lag
        peak_val = corr[peak_idx]
        if corr[0] > 0 and (peak_val / corr[0]) > 0.38:
            f0 = float(sr / peak_idx)
            pitches.append(f0)

    pitches_arr = np.array(pitches)
    flags: list[str] = []
    multi_speaker = False
    background_whisper = False
    median_f0 = 0.0
    std_f0 = 0.0

    if len(pitches_arr) >= 15:
        median_f0 = float(np.median(pitches_arr))
        std_f0 = float(np.std(pitches_arr))
        p25, p75 = np.percentile(pitches_arr, [25, 75])
        iqr = float(p75 - p25)

        # 2-cluster separation test
        sorted_p = np.sort(pitches_arr)
        half = len(sorted_p) // 2
        c1, c2 = float(np.mean(sorted_p[:half])), float(np.mean(sorted_p[half:]))
        cluster_gap = abs(c2 - c1)

        # Multimodal pitch criteria indicating two distinct voices (e.g. candidate + helper)
        if (cluster_gap > 60.0 and std_f0 > 45.0) or (iqr > 55.0 and std_f0 > 48.0):
            multi_speaker = True
            flags.append(f"Multiple speakers detected (pitch cluster variance: {cluster_gap:.1f}Hz)")

    # 2. Inter-word pause interval vocalization check (Prompting / Whispering during candidate pauses)
    inter_word_speech = False
    if word_timestamps and len(word_timestamps) >= 2:
        noise_floor = float(np.percentile(np.abs(audio), 15)) + 1e-6
        for idx in range(len(word_timestamps) - 1):
            pause_start = float(word_timestamps[idx].get("end", 0.0))
            pause_end = float(word_timestamps[idx + 1].get("start", 0.0))
            pause_dur = pause_end - pause_start

            # If there is a natural pause > 0.8s, analyze acoustic activity inside this silent window
            if 0.8 <= pause_dur <= 5.0:
                s_idx = int(pause_start * sr)
                e_idx = int(pause_end * sr)
                if e_idx <= len(audio):
                    pause_slice = audio[s_idx:e_idx]
                    pause_rms = float(np.sqrt(np.mean(pause_slice ** 2)))
                    # If energy in supposed pause is > 4.5x the ambient noise floor
                    if pause_rms > noise_floor * 4.5 and pause_rms > 0.015:
                        background_whisper = True
                        inter_word_speech = True
                        break

        if background_whisper:
            flags.append("Secondary vocal energy detected during candidate pauses")

    return AcousticProctorResult(
        multiple_speakers_detected=multi_speaker,
        background_whisper_detected=background_whisper,
        primary_pitch_hz=round(median_f0, 1),
        pitch_variance_hz=round(std_f0, 1),
        inter_word_speech_detected=inter_word_speech,
        flags_summary=flags,
    )
