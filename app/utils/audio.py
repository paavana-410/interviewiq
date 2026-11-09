"""Audio file handling for the UI response pipeline.

MediaRecorder in Chrome/Firefox typically produces webm/opus. faster-whisper
accepts webm natively when its bundled ctranslate2 is linked against ffmpeg.

When faster-whisper is not installed (mock/dev mode) the raw bytes are stored
and the MockSTT service transcribes synthetically — format doesn't matter.

If ffmpeg IS on PATH, we convert to wav. If not, we store the original file
and let Whisper deal with it. This keeps the dependency footprint minimal.

Required for production audio: install faster-whisper and ffmpeg.
  pip install faster-whisper
  # Windows: download ffmpeg from https://ffmpeg.org/download.html and add to PATH
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def save_upload(raw_bytes: bytes, dest_dir: Path, original_name: str) -> Path:
    """Save uploaded audio bytes to dest_dir.

    If the file is webm/ogg/mp4 AND ffmpeg is available, convert to wav first.
    Otherwise save as-is (faster-whisper can handle webm natively when installed).

    Returns the path to the saved file.
    """
    import uuid
    stem = uuid.uuid4().hex
    original_suffix = Path(original_name).suffix.lower() or ".webm"
    raw_path = dest_dir / f"{stem}{original_suffix}"
    raw_path.write_bytes(raw_bytes)

    # Formats that need conversion for reliable Whisper support
    needs_conversion = original_suffix in {".webm", ".ogg", ".opus", ".mp4", ".m4a"}

    if needs_conversion and _ffmpeg_available():
        wav_path = dest_dir / f"{stem}.wav"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(raw_path),
                    "-ac", "1", "-ar", "16000",  # mono 16 kHz — ideal for Whisper
                    str(wav_path),
                ],
                check=True,
                capture_output=True,
            )
            raw_path.unlink(missing_ok=True)
            return wav_path
        except subprocess.CalledProcessError:
            # ffmpeg failed — return the raw file and hope Whisper manages
            return raw_path

    return raw_path


def audio_dep_info() -> dict:
    """Return a dict with audio dependency status for the /health endpoint."""
    try:
        from faster_whisper import WhisperModel  # noqa
        whisper_ok = True
    except ImportError:
        whisper_ok = False

    return {
        "ffmpeg_available": _ffmpeg_available(),
        "faster_whisper_installed": whisper_ok,
        "note": (
            "Production audio requires faster-whisper and ffmpeg. "
            "In mock/dev mode audio is transcribed synthetically."
        ) if not (whisper_ok and _ffmpeg_available()) else "OK",
    }
