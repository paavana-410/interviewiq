#!/usr/bin/env python
"""run_audit_benchmark.py — Research benchmark CLI for InterviewIQ.

Stage-based design (each stage loads only what it needs):

    text        Hosted LLM judges over HTTP only.
                No Whisper, no Ollama, no TTS loaded.
                RAM: ~300 MB process + network calls.

    local_judge Ollama local judge only.
                OLLAMA_NUM_PARALLEL=1, short context (num_ctx ≤ 4096).
                RAM: ~2–3 GB (model weights stay in Ollama process).
                Never run together with the audio stage.

    audio       TTS synthesis + faster-whisper transcription only.
                Loads and unloads model after each file.
                RAM: ~500 MB–1 GB peak (base Whisper).
                Ollama must NOT be loaded — enforced at startup.

    analysis    Reads saved results.csv, computes plots and summary.
                RAM: ~200 MB (pandas + matplotlib only).

Typical run order:
    python run_audit_benchmark.py --stage text   --tag pilot --runs 1
    python run_audit_benchmark.py --stage analysis --run-id <id>

Usage:
    python run_audit_benchmark.py \\
        --answers-dir data/benchmark_answers \\
        --stage text \\
        --judges judge_gemini \\
        --runs 1 \\
        --tag pilot

    # Resume analysis on a previous run
    python run_audit_benchmark.py --stage analysis --run-id 20261103T...

Tags: debug | mock | pilot | final
    debug  — not logged to experiments.md; for local testing
    mock   — mock LLM; always safe; results not citable
    pilot  — real calls, exploratory; logged; may cite as preliminary
    final  — real calls; working tree must be clean

Security: API keys are never printed, logged, or written to any file.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv  # type: ignore
load_dotenv(override=False)

import click

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# Git / pip helpers
# ─────────────────────────────────────────────────────────────────────────────

def _git_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        r = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True, check=True)
        return bool(r.stdout.strip())
    except Exception:
        return False


def _pip_freeze() -> list[str]:
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip().splitlines()
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# experiments.md logger
# ─────────────────────────────────────────────────────────────────────────────

def _append_to_experiments_log(
    run_id: str,
    tag: str,
    judge_names: list[str],
    n_items: int,
    prompt_version: str,
    git_hash: str,
    summary: dict,
    run_status: str,
    failed_judges: list[str],
    stage: str,
) -> None:
    log_path = Path("docs/experiments.md")
    if not log_path.exists():
        return

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _pct(test: str) -> str:
        t = summary.get("per_test", {}).get(test, {})
        return f"{t.get('pass_rate', 0)*100:.0f}%" if t else "—"

    tests = summary.get("per_test", {})
    overall = (
        f"{sum(v.get('pass_rate', 0) for v in tests.values()) / len(tests) * 100:.0f}%"
        if tests else "—"
    )

    status_str = f"**{run_status.upper()}**"
    if failed_judges:
        status_str += f" (failed: {', '.join(failed_judges)})"

    row = (
        f"| {run_id} | {date_str} | **{tag}** | [{stage}] {'/'.join(judge_names)} "
        f"| {n_items} | v{prompt_version} | `{git_hash[:8]}` "
        f"| {_pct('delivery_invariance')} | {_pct('content_sensitivity')} "
        f"| {_pct('paraphrase_stability')} | {_pct('repeated_run')} "
        f"| {overall} | {status_str} |"
    )

    content = log_path.read_text(encoding="utf-8")
    marker = "<!-- EXPERIMENT_LOG_END"
    if marker in content:
        content = content.replace(marker, row + "\n" + marker)
        log_path.write_text(content, encoding="utf-8")
    else:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + row + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnswerItem:
    item_id: str
    question_text: str
    answer_text: str
    rubric_criteria: list[dict]
    gold_transcript: str | None
    audio_path: str | None
    # Hand-written answer tiers for known-groups / content-sensitivity
    answer_medium: str | None = None
    answer_weak: str | None = None
    # concept -> list of sentence indices that carry it (for sentence-based degradation)
    concept_sentences: dict | None = None


@dataclass
class BenchmarkRow:
    run_id: str
    item_id: str
    judge_name: str
    judge_model: str
    judge_provider: str
    test_name: str
    metric_name: str
    metric_value: float
    passed: bool
    threshold: float
    prompt_version: str = "v2"
    details: str = ""


@dataclass
class JudgeRunResult:
    judge_name: str
    judge_model: str
    status: Literal["ok", "failed"]
    failure_reason: str = ""
    rows: list[BenchmarkRow] = field(default_factory=list)
    call_stats: dict = field(default_factory=dict)  # from ContentEngine.stats.summary()


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_answer_items(answers_dir: Path) -> list[AnswerItem]:
    items: list[AnswerItem] = []
    for sub in sorted(answers_dir.iterdir()):
        if not sub.is_dir():
            continue
        answer_file = sub / "answer.txt"
        question_file = sub / "question.txt"
        rubric_file = sub / "rubric.json"
        if not answer_file.exists() or not question_file.exists():
            click.echo(f"  [skip] {sub.name}: missing answer.txt or question.txt")
            continue
        answer_text = answer_file.read_text(encoding="utf-8").strip()
        question_text = question_file.read_text(encoding="utf-8").strip()
        rubric_criteria = (
            json.loads(rubric_file.read_text(encoding="utf-8"))
            if rubric_file.exists() else []
        )
        # concept_sentences (optional): first rubric entry may carry a
        # {concept: [sentence_indices]} map for sentence-based degradation.
        concept_sentences = None
        if rubric_criteria and isinstance(rubric_criteria[0], dict):
            concept_sentences = rubric_criteria[0].get("concept_sentences")

        items.append(AnswerItem(
            item_id=sub.name,
            question_text=question_text,
            answer_text=answer_text,
            rubric_criteria=rubric_criteria,
            gold_transcript=(
                (sub / "gold_transcript.txt").read_text(encoding="utf-8").strip()
                if (sub / "gold_transcript.txt").exists() else None
            ),
            audio_path=str(sub / "audio.wav") if (sub / "audio.wav").exists() else None,
            answer_medium=(
                (sub / "answer_medium.txt").read_text(encoding="utf-8").strip()
                if (sub / "answer_medium.txt").exists() else None
            ),
            answer_weak=(
                (sub / "answer_weak.txt").read_text(encoding="utf-8").strip()
                if (sub / "answer_weak.txt").exists() else None
            ),
            concept_sentences=concept_sentences,
        ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Judge resolver
# ─────────────────────────────────────────────────────────────────────────────

def resolve_judges(judges_flag: str) -> list[tuple[str, object]]:
    from app.services.llm import MockLLMService, get_all_judge_services

    if judges_flag.strip().lower() == "mock":
        return [("mock", MockLLMService())]

    all_services = dict(get_all_judge_services())
    if judges_flag.strip().lower() == "all":
        return list(all_services.items())

    names = [n.strip() for n in judges_flag.split(",") if n.strip()]
    result = []
    for name in names:
        if name not in all_services:
            raise click.ClickException(
                f"Judge {name!r} not found. Available: {list(all_services.keys())}"
            )
        result.append((name, all_services[name]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────────────────────

_MAX_CONSECUTIVE_FAILURES = 3


def _probe_judges(judge_services: list[tuple[str, object]], secrets: list[str]) -> dict[str, bool]:
    """Pre-flight: qualify each judge with ONE real rubric call.

    Uses the same standard as qualify_judge.py (non-empty, valid JSON,
    finish_reason == 'stop') via app.services.qualify.qualify_call, which
    bypasses the disk cache so the call is real. Keys are never printed.
    """
    from app.services.qualify import qualify_call
    from app.services.judge_config import _mask_key, mask_secrets_in_text

    _Q = "Explain the difference between a stack and a queue."
    _A = ("A stack is LIFO; a queue is FIFO. Stacks are used for call stacks "
          "and queues for scheduling. Both have O(1) core operations.")

    results: dict[str, bool] = {}
    for judge_name, svc in judge_services:
        cfg = getattr(svc, "_config", None)
        masked_key = _mask_key(cfg.api_key) if cfg is not None else "***"
        try:
            qr = qualify_call(svc, _Q, _A, max_tokens=512)
        except Exception as exc:
            safe = mask_secrets_in_text(str(exc), secrets)
            click.secho(
                f"  pre-flight [{judge_name}]  key={masked_key}  "
                f"FAILED — {safe[:120]}",
                fg="red",
            )
            results[judge_name] = False
            continue

        results[judge_name] = qr.ok
        label = f"  pre-flight [{judge_name}]  key={masked_key}"
        if qr.ok:
            click.secho(
                f"{label}  OK  finish_reason={qr.finish_reason!r} "
                f"completion_tokens={qr.completion_tokens} "
                f"max_tokens_sent={qr.max_tokens_sent} latency={qr.latency_ms}ms",
                fg="green",
            )
        else:
            safe_err = mask_secrets_in_text(qr.error, secrets) if qr.error else ""
            click.secho(
                f"{label}  FAILED  finish_reason={qr.finish_reason!r} "
                f"completion_tokens={qr.completion_tokens} "
                f"max_tokens_sent={qr.max_tokens_sent} latency={qr.latency_ms}ms"
                + (f" — {safe_err}" if safe_err else ""),
                fg="red",
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-judge benchmark runner (shared by text + local_judge stages)
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark_for_judge(
    items: list[AnswerItem],
    judge_name: str,
    llm_service: object,
    run_id: str,
    settings_override: dict,
    seed: int,
    secrets: list[str],
) -> JudgeRunResult:
    from app.audit.tests.content_sensitivity import ContentSensitivityTest
    from app.audit.tests.delivery_invariance import DeliveryInvarianceTest
    from app.audit.tests.paraphrase_stability import ParaphraseStabilityTest
    from app.audit.tests.repeated_run import RepeatedRunTest
    from app.audit.tests.stt_sensitivity import STTSensitivityTest
    from app.audit.tests.known_groups import KnownGroupsTest
    from app.audit.variants import VariantGenerator
    from app.config import get_settings
    from app.engines.content import ContentEngine, RubricCriterion
    from app.engines.delivery import DeliveryEngine
    from app.engines.holistic import HolisticEvaluator
    from app.services.llm import get_paraphrase_service
    from app.services.judge_config import mask_secrets_in_text
    from app.services.stt import MockSTTService
    from app.utils.memory import MemoryTracker

    settings = get_settings()
    content_engine = ContentEngine(llm=llm_service, db=None)
    delivery_engine = DeliveryEngine()
    generator = VariantGenerator(seed=seed)
    pv = settings.content_engine_prompt_version

    try:
        para_service = get_paraphrase_service()
    except Exception:
        para_service = llm_service

    rows: list[BenchmarkRow] = []
    consecutive_failures = 0

    def _is_fatal(exc: Exception) -> bool:
        return isinstance(exc, (PermissionError, LookupError)) and not isinstance(exc, json.JSONDecodeError)

    def _safe_msg(exc: Exception) -> str:
        return mask_secrets_in_text(str(exc), secrets)[:200]

    def _add(test_name, metric_name, value, passed, threshold, details=None,
             invalid=False, invalid_reason=""):
        d = details or {}
        if invalid:
            d = {**d, "invalid": True, "invalid_reason": invalid_reason}
        rows.append(BenchmarkRow(
            run_id=run_id,
            item_id=item.item_id,
            judge_name=judge_name,
            judge_model=llm_service.model_name,
            judge_provider=llm_service.provider,
            test_name=test_name,
            metric_name=metric_name,
            metric_value=value,
            passed=passed,
            threshold=threshold,
            prompt_version=pv,
            details=json.dumps(d),
        ))

    def _call(fn, label: str):
        nonlocal consecutive_failures
        try:
            result = fn()
            consecutive_failures = 0
            return result
        except Exception as exc:
            consecutive_failures += 1
            safe = _safe_msg(exc)
            if _is_fatal(exc):
                raise RuntimeError(
                    f"Fatal error on {label} ({type(exc).__name__}): {safe}"
                )
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"Judge aborted after {_MAX_CONSECUTIVE_FAILURES} "
                    f"consecutive failures on {label}: {safe}"
                )
            click.echo(f"      [warn] {label}: {safe}")
            return None

    try:
        for item in items:
            click.echo(
                f"    {item.item_id:<32} judge={judge_name}  model={llm_service.model_name}"
            )

            rubric_criteria = [
                RubricCriterion(
                    criterion_name=c.get("criterion_name", "criterion"),
                    criterion_type=c.get("criterion_type", "correctness"),
                    weight=c.get("weight", 1.0),
                    expected_concepts=c.get("expected_concepts", []),
                    star_component=c.get("star_component", False),
                    scoring_guide=c.get("scoring_guide", ""),
                )
                for c in item.rubric_criteria
            ]

            stt_mock = MockSTTService(fixed_transcript=item.answer_text)
            stt_result = stt_mock.transcribe("synthetic")
            words = stt_result.words

            def _holistic():
                h = HolisticEvaluator(llm=llm_service).evaluate(
                    item.answer_text, item.question_text
                )
                _add("baseline_holistic", "holistic_score", h.score, True, 0.0,
                     {"reasoning": h.reasoning})
            _call(_holistic, "baseline_holistic")

            def _rubric():
                c_result = content_engine.analyze(
                    transcript_text=item.answer_text,
                    question_text=item.question_text,
                    rubric_criteria=rubric_criteria,
                )
                _add("baseline_rubric", "rubric_score", c_result.overall_score, True, 0.0)
            _call(_rubric, "baseline_rubric")

            def _di():
                variants = [
                    generator.filler_injection(words, filler_rate=0.10),
                    generator.pause_insertion(words, num_pauses=4),
                    generator.pace_change(words, factor=1.6),
                ]
                di = DeliveryInvarianceTest(
                    content_engine,
                    threshold=settings_override.get(
                        "delivery_invariance_max", settings.audit_delivery_invariance_max
                    ),
                ).run(item.answer_text, item.question_text, rubric_criteria, variants)
                _add(di.test_name, "delivery_invariance_error", di.metric_value,
                     di.passed, di.threshold, di.details,
                     invalid=di.invalid, invalid_reason=di.invalid_reason)
            _call(_di, "delivery_invariance")

            def _cs():
                concepts = [c.expected_concepts[0]
                            for c in rubric_criteria if c.expected_concepts][:3]
                if not concepts:
                    return
                # Prefer a hand-written degraded answer when present (stronger
                # evidence than auto-degradation). answer_weak first, else medium.
                hand_written = item.answer_weak or item.answer_medium
                deg_v = generator.quality_degradation(
                    item.answer_text, words, concepts_to_remove=concepts,
                    concept_sentences=item.concept_sentences,
                    hand_written=hand_written,
                )
                cs = ContentSensitivityTest(
                    content_engine, delivery_engine,
                    threshold=settings_override.get(
                        "content_sensitivity_min", settings.audit_content_sensitivity_min
                    ),
                ).run(item.answer_text, item.question_text, rubric_criteria, [deg_v])
                _add(cs.test_name, "content_sensitivity", cs.metric_value,
                     cs.passed, cs.threshold, cs.details,
                     invalid=cs.invalid, invalid_reason=cs.invalid_reason)
            _call(_cs, "content_sensitivity")

            def _ps():
                para_v = generator.paraphrase(
                    item.answer_text, words, llm_service=para_service
                )
                ps = ParaphraseStabilityTest(
                    content_engine,
                    threshold=settings_override.get(
                        "paraphrase_max_deviation", settings.audit_paraphrase_max_deviation
                    ),
                ).run(item.answer_text, item.question_text, rubric_criteria, [para_v])
                _add(ps.test_name, "paraphrase_deviation", ps.metric_value,
                     ps.passed, ps.threshold, ps.details,
                     invalid=ps.invalid, invalid_reason=ps.invalid_reason)
            _call(_ps, "paraphrase_stability")

            def _rr():
                rr = RepeatedRunTest(
                    content_engine,
                    n_runs=settings_override.get("repeated_runs", settings.audit_repeated_runs),
                    threshold=settings_override.get(
                        "repeatability_max_variance", settings.audit_repeatability_max_variance
                    ),
                ).run(item.answer_text, item.question_text, rubric_criteria)
                _add(rr.test_name, "repeatability_variance", rr.metric_value,
                     rr.passed, rr.threshold, rr.details)
            _call(_rr, "repeated_run")

            if item.gold_transcript and item.audio_path:
                def _stt():
                    w_result = MockSTTService(
                        fixed_transcript=item.answer_text
                    ).transcribe(item.audio_path)
                    st = STTSensitivityTest(
                        content_engine,
                        threshold=settings_override.get(
                            "stt_max_drift", settings.audit_stt_max_drift
                        ),
                    ).run(item.gold_transcript, w_result.text,
                          item.question_text, rubric_criteria)
                    _add(st.test_name, "stt_sensitivity", st.metric_value,
                         st.passed, st.threshold, st.details)
                _call(_stt, "stt_sensitivity")

            def _ablation():
                dr = delivery_engine.analyze(words)
                delivery_features = {
                    "speaking_rate_wpm": dr.speaking_rate_wpm,
                    "filler_density": dr.filler_density,
                    "pause_count": dr.pause_count,
                }
                import app.config as _cfg
                _orig = _cfg._settings.content_engine_include_delivery
                _cfg._settings.content_engine_include_delivery = True
                c_abl = content_engine.analyze(
                    transcript_text=item.answer_text,
                    question_text=item.question_text,
                    rubric_criteria=rubric_criteria,
                    delivery_features=delivery_features,
                )
                _cfg._settings.content_engine_include_delivery = _orig
                _add("ablation_with_delivery", "rubric_score_with_delivery",
                     c_abl.overall_score, True, 0.0)
            _call(_ablation, "ablation")

            # ── Known-groups validity (needs strong/medium/weak answers) ────────
            def _kg():
                if not (item.answer_medium or item.answer_weak):
                    return  # no tiered answers for this item — skip silently
                kg = KnownGroupsTest(content_engine).run(
                    item.question_text, rubric_criteria,
                    item.answer_text, item.answer_medium, item.answer_weak,
                )
                _add(kg.test_name, "ordering_score", kg.metric_value,
                     kg.passed, kg.threshold, kg.details,
                     invalid=kg.invalid, invalid_reason=kg.invalid_reason)
            _call(_kg, "known_groups")

    except RuntimeError as abort_exc:
        safe = mask_secrets_in_text(str(abort_exc), secrets)
        click.secho(f"  ✗ Judge {judge_name} ABORTED: {safe}", fg="red")
        return JudgeRunResult(
            judge_name=judge_name,
            judge_model=llm_service.model_name,
            status="failed",
            failure_reason=safe,
            rows=[],
            call_stats=content_engine.stats.summary(),
        )

    return JudgeRunResult(
        judge_name=judge_name,
        judge_model=llm_service.model_name,
        status="ok",
        rows=rows,
        call_stats=content_engine.stats.summary(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(
    rows: list[BenchmarkRow],
    out_dir: Path,
    config: dict,
    judge_results: list[JudgeRunResult],
) -> dict:
    """Write CSV, summary.json, plots. Returns summary dict."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    summary: dict = {"per_test": {}, "per_judge": {}}

    if not rows:
        click.echo("  (no rows to write — all judges failed)")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    fieldnames = list(BenchmarkRow.__dataclass_fields__.keys())
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    df = pd.DataFrame([asdict(r) for r in rows])

    # Parse the "invalid" flag out of each row's details JSON so we can
    # exclude no-op variants from pass rates (never reported as 0).
    def _is_invalid(details_str):
        try:
            return bool(json.loads(details_str).get("invalid", False))
        except Exception:
            return False

    df["invalid"] = df["details"].apply(_is_invalid) if "details" in df.columns else False
    valid_df = df[~df["invalid"]]

    for test_name, grp in df.groupby("test_name"):
        invalid_n = int(grp["invalid"].sum()) if "invalid" in grp.columns else 0
        valid_grp = grp[~grp["invalid"]]
        summary["per_test"][test_name] = {
            "pass_rate": float(valid_grp["passed"].mean()) if len(valid_grp) else 0.0,
            "n_valid": int(len(valid_grp)),
            "n_invalid": invalid_n,
            "mean_metric": float(valid_grp["metric_value"].mean()) if len(valid_grp) else None,
            "std_metric": float(valid_grp["metric_value"].std()) if len(valid_grp) > 1 else None,
            "n": int(len(grp)),
        }
    for judge, grp in df.groupby("judge_name"):
        valid_grp = grp[~grp["invalid"]] if "invalid" in grp.columns else grp
        summary["per_judge"][judge] = {
            "model": grp["judge_model"].iloc[0],
            "provider": grp["judge_provider"].iloc[0],
            "overall_pass_rate": float(valid_grp["passed"].mean()) if len(valid_grp) else 0.0,
            "mean_metric": float(valid_grp["metric_value"].mean()) if len(valid_grp) else None,
            "n_invalid": int(grp["invalid"].sum()) if "invalid" in grp.columns else 0,
        }

    # Total invalid variants across the run
    summary["invalid_variants_total"] = int(df["invalid"].sum()) if "invalid" in df.columns else 0

    # ── Degradation-method counts (content_sensitivity variants only) ────────
    # Each content_sensitivity row's details carries variants[].variant_params.degradation_method.
    method_counts: dict[str, int] = {}
    cs_rows = df[df["test_name"] == "content_sensitivity"] if "test_name" in df.columns else df.iloc[0:0]
    for _, row in cs_rows.iterrows():
        try:
            det = json.loads(row["details"])
        except Exception:
            continue
        for v in det.get("variants", []):
            # A variant may be an invalid no-op (no variant_params) — skip those
            vp = v.get("variant_params")
            if not vp:
                continue
            m = vp.get("degradation_method", "unknown")
            method_counts[m] = method_counts.get(m, 0) + 1
    summary["degradation_methods"] = method_counts

    failed = [jr for jr in judge_results if jr.status == "failed"]
    if failed:
        summary["failed_judges"] = [
            {"judge_name": jr.judge_name, "reason": jr.failure_reason}
            for jr in failed
        ]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plots — best-effort, skip if matplotlib not available
    try:
        from app.analysis.plots import (
            plot_audit_summary, plot_model_comparison,
            plot_repeatability, plot_score_deltas,
        )
        _make_plots(df, out_dir, summary)
    except Exception as e:
        click.echo(f"  [warn] plots skipped: {e}")

    click.echo(f"  ✓ Results  : {out_dir}/results.csv")
    click.echo(f"  ✓ Summary  : {out_dir}/summary.json")
    click.echo(f"  ✓ Config   : {out_dir}/config.json")

    return summary


def _make_plots(df, out_dir, summary):
    from app.analysis.plots import (
        plot_audit_summary, plot_model_comparison,
        plot_repeatability, plot_score_deltas,
    )
    import pandas as pd

    try:
        ddf = df[df["test_name"] == "delivery_invariance"][["judge_name", "metric_value"]].copy()
        ddf.columns = ["variant_type", "delta"]
        if not ddf.empty:
            plot_score_deltas(ddf, out_dir)
    except Exception:
        pass
    try:
        rdf = df[df["test_name"] == "repeated_run"]
        if not rdf.empty:
            scores = json.loads(rdf.iloc[0]["details"]).get("scores", [])
            if scores:
                plot_repeatability(scores, out_dir)
    except Exception:
        pass
    try:
        bdf = df[df["test_name"] == "baseline_rubric"][["judge_name", "metric_value"]].copy()
        bdf.columns = ["model", "overall_score"]
        if not bdf.empty:
            plot_model_comparison(bdf, out_dir)
    except Exception:
        pass
    try:
        audit_tests = [
            "delivery_invariance", "content_sensitivity",
            "paraphrase_stability", "repeated_run", "stt_sensitivity",
        ]
        av, at = {}, {}
        for t in audit_tests:
            sub = df[df["test_name"] == t]
            if not sub.empty:
                av[t] = float(sub["metric_value"].mean())
                at[t] = float(sub["threshold"].iloc[0])
        if av:
            plot_audit_summary(av, at, out_dir)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Stage implementations
# ─────────────────────────────────────────────────────────────────────────────

def stage_text(
    items: list[AnswerItem],
    judge_services: list[tuple[str, object]],
    run_id: str,
    out_dir: Path,
    settings_override: dict,
    seed: int,
    skip_preflight: bool,
    secrets: list[str],
    config_snapshot: dict,
) -> tuple[list[JudgeRunResult], dict]:
    """Hosted judges over HTTP. No Whisper/Ollama loaded."""
    from app.utils.memory import MemoryTracker, warn_if_low
    from app.config import get_settings

    settings = get_settings()
    warn_if_low("text", settings.memory_warn_below_gb)

    tracker = MemoryTracker("text", out_dir)
    tracker.start()

    # Pre-flight
    if not skip_preflight:
        click.echo("\nPre-flight connectivity check:")
        probe_results = _probe_judges(judge_services, secrets)
        failed_preflight = [n for n, ok in probe_results.items() if not ok]
        if failed_preflight:
            click.secho(
                f"\n  Pre-flight FAILED for: {failed_preflight}\n"
                "  Fix these judges or use --skip-preflight.",
                fg="red", bold=True,
            )
            tracker.finish()
            raise SystemExit(1)
        click.echo("  All judges reachable.\n")

    all_rows: list[BenchmarkRow] = []
    judge_results: list[JudgeRunResult] = []

    for judge_name, svc in judge_services:
        click.echo(f"\n  ── Judge: {judge_name}  ({svc.model_name})")
        result = run_benchmark_for_judge(
            items=items,
            judge_name=judge_name,
            llm_service=svc,
            run_id=run_id,
            settings_override=settings_override,
            seed=seed,
            secrets=secrets,
        )
        judge_results.append(result)
        if result.status == "ok":
            all_rows.extend(result.rows)
        tracker.update_peak()

    mem = tracker.finish()
    click.echo(
        f"\n  Memory: RSS {mem['rss_start_mb']:.0f}→{mem['rss_end_mb']:.0f} MB "
        f"(peak {mem['rss_peak_mb']:.0f} MB)"
    )

    summary = write_outputs(all_rows, out_dir, config_snapshot, judge_results)
    return judge_results, summary


def stage_local_judge(
    items: list[AnswerItem],
    run_id: str,
    out_dir: Path,
    settings_override: dict,
    seed: int,
    secrets: list[str],
    config_snapshot: dict,
) -> tuple[list[JudgeRunResult], dict]:
    """Ollama local judge only. Enforces OLLAMA_NUM_PARALLEL=1."""
    from app.utils.memory import MemoryTracker, warn_if_low
    from app.services.llm import get_ollama_service
    from app.config import get_settings

    settings = get_settings()
    warn_if_low("local_judge", settings.memory_warn_below_gb)

    os.environ["OLLAMA_NUM_PARALLEL"] = "1"
    click.echo("  OLLAMA_NUM_PARALLEL=1 enforced.")

    tracker = MemoryTracker("local_judge", out_dir)
    tracker.start()

    svc = get_ollama_service()
    click.echo(f"\n  ── Ollama: {svc.model_name}  (ctx={settings.ollama_num_ctx})")

    result = run_benchmark_for_judge(
        items=items,
        judge_name="judge_ollama",
        llm_service=svc,
        run_id=run_id,
        settings_override=settings_override,
        seed=seed,
        secrets=secrets,
    )
    tracker.update_peak()

    mem = tracker.finish()
    click.echo(
        f"\n  Memory: RSS {mem['rss_start_mb']:.0f}→{mem['rss_end_mb']:.0f} MB "
        f"(peak {mem['rss_peak_mb']:.0f} MB)"
    )

    all_rows = result.rows if result.status == "ok" else []
    summary = write_outputs(all_rows, out_dir, config_snapshot, [result])
    return [result], summary


def stage_audio(
    items: list[AnswerItem],
    run_id: str,
    out_dir: Path,
    config_snapshot: dict,
) -> None:
    """TTS + Whisper only. Refuses if Ollama is running."""
    import httpx
    from app.utils.memory import MemoryTracker, warn_if_low
    from app.config import get_settings

    settings = get_settings()
    warn_if_low("audio", settings.memory_warn_below_gb)

    # Guard: refuse if Ollama is serving (would double RAM)
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        if r.status_code == 200:
            models = r.json().get("models", [])
            if models:
                click.secho(
                    f"\n  ERROR: Ollama is running with {len(models)} model(s) loaded.\n"
                    "  The audio stage loads Whisper and should not run alongside Ollama.\n"
                    "  Stop Ollama first: close the Ollama app or kill the process.\n",
                    fg="red", bold=True,
                )
                raise SystemExit(1)
    except (httpx.ConnectError, httpx.TimeoutException):
        pass  # Ollama not running — safe to proceed

    tracker = MemoryTracker("audio", out_dir)
    tracker.start()

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    results = []

    for item in items:
        if item.audio_path and Path(item.audio_path).exists():
            click.echo(f"  Transcribing existing audio: {item.audio_path}")
        else:
            click.echo(f"  [skip] {item.item_id}: no audio.wav — skipping audio stage")
            continue

        # Lazy-load Whisper
        try:
            from faster_whisper import WhisperModel  # type: ignore
            model = WhisperModel(
                settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            segments, info = model.transcribe(item.audio_path, word_timestamps=True)
            text = " ".join(seg.text.strip() for seg in segments)
            result_rec = {"item_id": item.item_id, "transcript": text, "language": info.language}
            results.append(result_rec)
            click.echo(f"    → {text[:80]}")
        except ImportError:
            click.secho("  faster-whisper not installed. Run: pip install faster-whisper", fg="red")
            break
        except Exception as exc:
            click.echo(f"  [warn] {item.item_id}: {exc}")
        finally:
            # Unload model and free RAM
            try:
                del model
            except NameError:
                pass
            gc.collect()
            tracker.update_peak()

    tracker.update_peak()
    mem = tracker.finish()

    (out_dir / "audio_transcripts.json").write_text(json.dumps(results, indent=2))
    click.echo(
        f"\n  ✓ Audio transcripts: {out_dir}/audio_transcripts.json\n"
        f"  Memory: RSS {mem['rss_start_mb']:.0f}→{mem['rss_end_mb']:.0f} MB "
        f"(peak {mem['rss_peak_mb']:.0f} MB)"
    )


def stage_analysis(run_id: str | None, output_dir: Path) -> None:
    """Read saved results and regenerate plots + summary. Pandas only."""
    import pandas as pd
    from app.utils.memory import MemoryTracker, warn_if_low
    from app.config import get_settings
    from app.analysis.plots import (
        plot_audit_summary, plot_model_comparison,
        plot_repeatability, plot_score_deltas,
    )

    settings = get_settings()
    warn_if_low("analysis", settings.memory_warn_below_gb)

    # Find run dir
    if run_id:
        run_dir = output_dir / run_id
    else:
        # Latest run
        dirs = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not dirs:
            click.secho("No experiment runs found.", fg="red")
            raise SystemExit(1)
        run_dir = dirs[0]
        click.echo(f"  Using latest run: {run_dir.name}")

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        click.secho(f"  No results.csv in {run_dir}", fg="red")
        raise SystemExit(1)

    tracker = MemoryTracker("analysis", run_dir)
    tracker.start()

    df = pd.read_csv(results_csv)
    click.echo(f"  Loaded {len(df)} rows from {results_csv}")

    _make_plots(df, run_dir, {})

    mem = tracker.finish()
    click.echo(
        f"  ✓ Plots regenerated in {run_dir}/\n"
        f"  Memory: RSS {mem['rss_start_mb']:.0f}→{mem['rss_end_mb']:.0f} MB "
        f"(peak {mem['rss_peak_mb']:.0f} MB)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--answers-dir", default="data/benchmark_answers", show_default=True,
              type=click.Path())
@click.option(
    "--stage",
    type=click.Choice(["text", "local_judge", "audio", "analysis"], case_sensitive=False),
    default="text", show_default=True,
    help="text=hosted judges; local_judge=Ollama; audio=Whisper; analysis=plots only",
)
@click.option("--judges", default="all", show_default=True,
              help="For --stage text: 'all', 'mock', or comma-separated judge names")
@click.option("--runs", default=3, show_default=True,
              help="Repeated runs for consistency test")
@click.option("--output-dir", default="experiments", show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option(
    "--tag", default="pilot",
    type=click.Choice(["debug", "mock", "pilot", "final"], case_sensitive=False),
    show_default=True,
)
@click.option("--run-id", default=None,
              help="For --stage analysis: run_id to re-analyse (default: latest)")
@click.option("--delivery-invariance-max", default=None, type=float)
@click.option("--content-sensitivity-min", default=None, type=float)
@click.option("--paraphrase-max-deviation", default=None, type=float)
@click.option("--skip-preflight", is_flag=True)
def main(
    answers_dir: str,
    stage: str,
    judges: str,
    runs: int,
    output_dir: str,
    seed: int,
    tag: str,
    run_id: str | None,
    delivery_invariance_max: float | None,
    content_sensitivity_min: float | None,
    paraphrase_max_deviation: float | None,
    skip_preflight: bool,
) -> None:
    """InterviewIQ research benchmark. Keys are never printed or logged."""
    from app.config import get_settings
    from app.services.judge_config import get_judge_registry, mask_secrets_in_text
    from app.utils.memory import warn_if_low

    settings = get_settings()

    # ── analysis stage doesn't need items or judges ───────────────────────────
    if stage == "analysis":
        stage_analysis(run_id=run_id, output_dir=Path(output_dir))
        return

    # ── guards ────────────────────────────────────────────────────────────────
    if tag == "final" and _git_dirty():
        click.secho(
            "\nERROR: Working tree has uncommitted changes.\n"
            "Final runs require a clean git working tree.",
            fg="red", bold=True,
        )
        raise SystemExit(1)

    if tag in ("mock", "debug") or stage == "local_judge":
        if tag == "mock":
            os.environ["LLM_PROVIDER"] = "mock"
    if tag == "debug":
        click.secho("  [debug] This run will NOT be logged to experiments.md.", fg="yellow")

    # ── build run_id ──────────────────────────────────────────────────────────
    computed_run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        f"_{tag}_{stage}_{uuid.uuid4().hex[:8]}"
    )

    click.echo(f"\nInterviewIQ Audit Benchmark")
    click.echo(f"  run_id   : {computed_run_id}")
    click.echo(f"  stage    : {stage}")
    click.echo(f"  tag      : {tag}")
    click.echo(f"  seed     : {seed}")

    # ── load items ────────────────────────────────────────────────────────────
    answers_path = Path(answers_dir)
    if not answers_path.exists():
        click.secho(f"answers-dir not found: {answers_dir}", fg="red")
        raise SystemExit(1)

    items = load_answer_items(answers_path)
    if not items:
        click.secho("No valid answer items found.", fg="red")
        raise SystemExit(1)
    click.echo(f"  items    : {len(items)}")

    # ── secrets for masking ───────────────────────────────────────────────────
    try:
        registry = get_judge_registry()
        secrets = registry.all_secrets()
    except Exception:
        secrets = []

    # ── settings overrides ────────────────────────────────────────────────────
    settings_override: dict = {"repeated_runs": runs}
    if delivery_invariance_max is not None:
        settings_override["delivery_invariance_max"] = delivery_invariance_max
    if content_sensitivity_min is not None:
        settings_override["content_sensitivity_min"] = content_sensitivity_min
    if paraphrase_max_deviation is not None:
        settings_override["paraphrase_max_deviation"] = paraphrase_max_deviation

    # ── config snapshot ───────────────────────────────────────────────────────
    git_hash = _git_hash()

    try:
        selected_judge_names = (
            [n for n, _ in resolve_judges(judges)]
            if stage != "local_judge" else [f"ollama/{settings.ollama_model}"]
        )
        judges_safe = [
            j.safe_info() for j in registry.judges
            if j.name in set(selected_judge_names)
        ] if stage != "local_judge" else [
            {"name": "judge_ollama", "model": settings.ollama_model,
             "provider": "ollama", "base_url": settings.ollama_base_url}
        ]
        paraphrase_safe = registry.paraphrase.safe_info() if registry.paraphrase else None
    except Exception:
        judges_safe = []
        paraphrase_safe = None

    config_snapshot: dict = {
        "run_id": computed_run_id,
        "tag": tag,
        "stage": stage,
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "answers_dir": str(answers_path.resolve()),
        "judges": judges_safe,
        "paraphrase": paraphrase_safe,
        "seed": seed,
        "n_items": len(items),
        "settings_override": settings_override,
        "base_settings": {
            "whisper_model_size": settings.whisper_model_size,
            "audit_delivery_invariance_max": settings.audit_delivery_invariance_max,
            "audit_content_sensitivity_min": settings.audit_content_sensitivity_min,
            "audit_paraphrase_max_deviation": settings.audit_paraphrase_max_deviation,
            "audit_repeatability_max_variance": settings.audit_repeatability_max_variance,
            "audit_stt_max_drift": settings.audit_stt_max_drift,
            "content_engine_prompt_version": settings.content_engine_prompt_version,
            "ollama_model": settings.ollama_model,
            "ollama_num_ctx": settings.ollama_num_ctx,
        },
        "pip_freeze": _pip_freeze(),
        # API keys are NEVER included
    }

    out_dir = Path(output_dir) / computed_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2))

    # ── dispatch to stage ─────────────────────────────────────────────────────
    judge_results: list[JudgeRunResult] = []
    summary: dict = {}

    if stage == "text":
        try:
            judge_services = resolve_judges(judges)
        except Exception as exc:
            click.secho(str(exc), fg="red")
            raise SystemExit(1)
        click.echo(f"  judges   : {[n for n, _ in judge_services]}\n")

        judge_results, summary = stage_text(
            items=items,
            judge_services=judge_services,
            run_id=computed_run_id,
            out_dir=out_dir,
            settings_override=settings_override,
            seed=seed,
            skip_preflight=skip_preflight,
            secrets=secrets,
            config_snapshot=config_snapshot,
        )

    elif stage == "local_judge":
        judge_results, summary = stage_local_judge(
            items=items,
            run_id=computed_run_id,
            out_dir=out_dir,
            settings_override=settings_override,
            seed=seed,
            secrets=secrets,
            config_snapshot=config_snapshot,
        )

    elif stage == "audio":
        stage_audio(
            items=items,
            run_id=computed_run_id,
            out_dir=out_dir,
            config_snapshot=config_snapshot,
        )
        # audio stage doesn't produce BenchmarkRows
        config_snapshot["status"] = "ok"
        (out_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2))
        click.echo(f"\nRun ID : {computed_run_id}")
        return

    # ── finalise status ───────────────────────────────────────────────────────
    failed_judges = [jr for jr in judge_results if jr.status == "failed"]
    ok_judges     = [jr for jr in judge_results if jr.status == "ok"]

    run_status = "failed" if not ok_judges else ("partial" if failed_judges else "ok")
    config_snapshot["status"] = run_status
    if failed_judges:
        config_snapshot["failed_judges"] = [
            {"judge_name": jr.judge_name, "model": jr.judge_model,
             "reason": jr.failure_reason}
            for jr in failed_judges
        ]
    (out_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2))

    # ── log to experiments.md (never for debug) ───────────────────────────────
    if tag != "debug":
        _append_to_experiments_log(
            run_id=computed_run_id,
            tag=tag,
            judge_names=[jr.judge_name for jr in judge_results],
            n_items=len(items),
            prompt_version=settings.content_engine_prompt_version,
            git_hash=git_hash,
            summary=summary,
            run_status=run_status,
            failed_judges=[jr.judge_name for jr in failed_judges],
            stage=stage,
        )
        click.echo(f"  ✓ Logged to : docs/experiments.md  (status={run_status})")
    else:
        click.echo("  [debug] Not logged to experiments.md.")

    # ── summary table ─────────────────────────────────────────────────────────
    import pandas as pd
    all_rows = [r for jr in judge_results if jr.status == "ok" for r in jr.rows]

    click.echo("\n── Summary ────────────────────────────────────────────────")
    if failed_judges:
        click.secho(
            f"  Aborted: {', '.join(jr.judge_name for jr in failed_judges)}", fg="red"
        )
    if all_rows:
        df = pd.DataFrame([asdict(r) for r in all_rows])
        def _is_invalid(s):
            try: return bool(json.loads(s).get("invalid", False))
            except Exception: return False
        df["invalid"] = df["details"].apply(_is_invalid) if "details" in df.columns else False
        for tn, grp in df.groupby("test_name"):
            invalid_n = int(grp["invalid"].sum()) if "invalid" in grp.columns else 0
            valid_grp = grp[~grp["invalid"]]
            pr = (valid_grp["passed"].mean() * 100) if len(valid_grp) else 0.0
            mv = valid_grp["metric_value"].mean() if len(valid_grp) else float("nan")
            icon = "✓" if pr >= 80 else "✗"
            inv = f"  invalid={invalid_n}" if invalid_n else ""
            click.echo(f"  {icon} {tn:<35}  pass={pr:.0f}%  mean={mv:.4f}{inv}")

    # ── Invalid-variant count (total) ────────────────────────────────────────
    if all_rows:
        df_all = pd.DataFrame([asdict(r) for r in all_rows])
        df_all["invalid"] = df_all["details"].apply(
            lambda s: bool(json.loads(s).get("invalid", False)) if s else False
        )
        total_invalid = int(df_all["invalid"].sum())
        if total_invalid:
            click.secho(
                f"\n  ⚠️  {total_invalid} invalid variant(s) excluded from pass rates "
                f"(degradation/paraphrase was a no-op for those items).",
                fg="yellow",
            )

    # ── Degradation-method counts ────────────────────────────────────────────
    if all_rows:
        cs = df_all[df_all["test_name"] == "content_sensitivity"]
        m_counts: dict[str, int] = {}
        for _, row in cs.iterrows():
            try:
                det = json.loads(row["details"])
            except Exception:
                continue
            for v in det.get("variants", []):
                vp = v.get("variant_params")
                if vp and vp.get("degradation_method"):
                    m = vp["degradation_method"]
                    m_counts[m] = m_counts.get(m, 0) + 1
        if m_counts:
            click.echo("\n── Degradation methods ─────────────────────────────────────")
            for m, c in sorted(m_counts.items()):
                click.echo(f"  {m:<35}  {c}")

    # ── Call stats per judge ─────────────────────────────────────────────────
    click.echo("\n── Call stats ─────────────────────────────────────────────")
    totals = {"total_calls": 0, "truncated": 0, "retried": 0, "failed": 0}
    for jr in judge_results:
        cs = jr.call_stats
        if cs:
            click.echo(
                f"  {jr.judge_name:<30}  total={cs.get('total_calls',0)}  "
                f"truncated={cs.get('truncated',0)}  "
                f"retried={cs.get('retried',0)}  "
                f"failed={cs.get('failed',0)}"
            )
            for k in totals:
                totals[k] += cs.get(k, 0)
    if len(judge_results) > 1:
        click.echo(
            f"  {'TOTAL':<30}  total={totals['total_calls']}  "
            f"truncated={totals['truncated']}  "
            f"retried={totals['retried']}  "
            f"failed={totals['failed']}"
        )
    # Save to summary.json
    try:
        run_summary_path = out_dir / "summary.json"
        if run_summary_path.exists():
            existing = json.loads(run_summary_path.read_text(encoding="utf-8"))
            existing["call_stats"] = {jr.judge_name: jr.call_stats for jr in judge_results}
            run_summary_path.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass

    status_color = {"ok": "green", "partial": "yellow", "failed": "red"}[run_status]
    click.secho(f"\nRun status : {run_status.upper()}", fg=status_color, bold=True)
    click.echo(f"Run ID     : {computed_run_id}")
    click.echo(f"Output     : {out_dir}/\n")

    if run_status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
