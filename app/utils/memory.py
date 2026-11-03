"""Memory monitoring utilities for RAM-aware stage execution.

Uses psutil to read process RSS and system available RAM.
Logs to experiments/<run_id>/memory.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _psutil():
    import psutil  # type: ignore
    return psutil


def rss_mb() -> float:
    """Current process RSS in MB."""
    p = _psutil().Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)


def available_ram_gb() -> float:
    """System available RAM in GB."""
    return _psutil().virtual_memory().available / (1024 ** 3)


def warn_if_low(stage: str, threshold_gb: float = 1.5) -> None:
    """Print a warning if available RAM is below threshold."""
    avail = available_ram_gb()
    if avail < threshold_gb:
        print(
            f"\n  ⚠️  WARNING: Only {avail:.1f} GB available RAM before stage '{stage}'.\n"
            f"     Threshold is {threshold_gb:.1f} GB. The stage may fail or swap.\n"
            f"     Close other applications before continuing.\n"
        )


class MemoryTracker:
    """Track RSS at start, end, and peak for a stage."""

    def __init__(self, stage: str, run_dir: Path):
        self.stage = stage
        self.run_dir = run_dir
        self._start_rss: float = 0.0
        self._peak_rss: float = 0.0
        self._end_rss: float = 0.0
        self._start_avail: float = 0.0
        self._end_avail: float = 0.0
        self._start_time: float = 0.0

    def start(self) -> None:
        self._start_rss = rss_mb()
        self._peak_rss = self._start_rss
        self._start_avail = available_ram_gb()
        self._start_time = time.monotonic()

    def update_peak(self) -> None:
        cur = rss_mb()
        if cur > self._peak_rss:
            self._peak_rss = cur

    def finish(self) -> dict:
        self._end_rss = rss_mb()
        self._end_avail = available_ram_gb()
        elapsed = time.monotonic() - self._start_time
        record = {
            "stage": self.stage,
            "elapsed_sec": round(elapsed, 1),
            "rss_start_mb": round(self._start_rss, 1),
            "rss_peak_mb": round(self._peak_rss, 1),
            "rss_end_mb": round(self._end_rss, 1),
            "rss_delta_mb": round(self._end_rss - self._start_rss, 1),
            "available_ram_start_gb": round(self._start_avail, 2),
            "available_ram_end_gb": round(self._end_avail, 2),
        }
        self._save(record)
        return record

    def _save(self, record: dict) -> None:
        path = self.run_dir / "memory.json"
        existing: list[dict] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = [existing]
            except Exception:
                existing = []
        existing.append(record)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
