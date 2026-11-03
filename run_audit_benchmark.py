#!/usr/bin/env python
"""run_audit_benchmark.py — Research benchmark CLI for InterviewIQ.

Usage:
    python run_audit_benchmark.py --answers-dir data/benchmark_answers --tag pilot
    python run_audit_benchmark.py --answers-dir data/benchmark_answers \\
        --judges judge_nemotron,judge_gemini --tag pilot
    python run_audit_benchmark.py --answers-dir data/benchmark_answers --tag final
    python run_audit_benchmark.py --answers-dir data/benchmark_answers --tag debug

Tags:
    debug  — real or mock calls; NOT logged to experiments.md; for local testing
    mock   — mock LLM only; logged as mock (never cite)
    pilot  — real calls, exploratory; logged; may cite as preliminary
    final  — real calls; fails if working tree is dirty; logged as final

Per-judge failure policy:
    If a judge returns 401/403/404, or fails the same call 3 times in a row,
    that judge is aborted. Its status is recorded as "failed" in config.json
    and experiments.md. Results from other judges are still written normally.
    The run as a whole is marked "partial" if ≥1 judge failed, "ok" if all passed.

Security:
    API keys are never printed, logged, or included in any output file.
    All error messages are masked before display.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# Load .env before any app imports
from dotenv import load_dotenv  # type: ignore
load_dotenv(override=False)

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# Research record-keeping helpers
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


def _append_to_experiments_log(
    run_id: str,
    tag: str,
    judge_names: list[str],
    n_items: int,
    prompt_version: str,
    git_hash: str,
    summary: dict,
    run_status: str,         # "ok" | "partial" | "failed"
    failed_judges: list[str],
) -> None:
    """Append one row to docs/experiments.md. Never called for debug runs."""
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

    judges_str = "/".join(judge_names)
    row = (
        f"| {run_id} | {date_str} | **{tag}** | {judges_str} | {n_items} "
        f"| v{prompt_version} | `{git_hash[:8]}` "
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
    details: str = ""


@dataclass
class JudgeRunResult:
    judge_name: str
    judge_model: str
    status: Literal["ok", "failed"]
    failure_reason: str = ""          # safe — no key values
    rows: list[BenchmarkRow] = field(default_factory=list)


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
        gold = (
            (sub / "gold_transcript.txt").read_text(encoding="utf-8").strip()
            if (sub / "gold_transcript.txt").exists() else None
        )
        audio = str(sub / "audio.wav") if (sub / "audio.wav").exists() else None
        items.append(AnswerItem(
            item_id=sub.name,
            question_text=question_text,
            answer_text=answer_text,
            rubric_criteria=rubric_criteria,
            gold_transcript=gold,
            audio_path=audio,
        ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Judge resolution
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
# Pre-flight check — probe each judge before starting
# ─────────────────────────────────────────────────────────────────────────────

# Non-retryable HTTP status codes — abort the judge immediately
_FATAL_STATUS_CODES = {401, 403, 404}
# How many consecutive failures before we give up on a judge
_MAX_CONSECUTIVE_FAILURES = 3


def _probe_judges(
    judge_services: list[tuple[str, object]],
    secrets: list[str],
) -> dict[str, bool]:
    """Pre-flight: probe every judge. Returns {judge_name: ok}."""
    from check_config import probe_judge
    from app.services.judge_config import JudgeConfig

    results: dict[str, bool] = {}
    for judge_name, svc in judge_services:
        if svc.provider == "mock":
            results[judge_name] = True
            continue

        # Build a minimal JudgeConfig from the service for probing
        cfg = svc._config  # OpenAICompatibleService stores _config
        result = probe_judge(cfg)
        results[judge_name] = result.ok

        from app.services.judge_config import _mask_key
        masked = _mask_key(cfg.api_key)
        label = f"  pre-flight [{judge_name}]"
        if result.ok:
            click.secho(f"{label:<40} OK  ({result.message})", fg="green")
        else:
            click.secho(f"{label:<40} FAILED — {result.message}", fg="red")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-judge benchmark runner
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
    """Run all audit tests for one judge. Returns a JudgeRunResult.

    Aborts the judge (returns status=failed) if:
    - A non-retryable error occurs (401/403/404 / ValueError / PermissionError / LookupError)
    - The same call fails _MAX_CONSECUTIVE_FAILURES times in a row
    """
    from app.audit.tests.content_sensitivity import ContentSensitivityTest
    from app.audit.tests.delivery_invariance import DeliveryInvarianceTest
    from app.audit.tests.paraphrase_stability import ParaphraseStabilityTest
    from app.audit.tests.repeated_run import RepeatedRunTest
    from app.audit.tests.stt_sensitivity import STTSensitivityTest
    from app.audit.variants import VariantGenerator
    from app.config import get_settings
    from app.engines.content import ContentEngine, RubricCriterion
    from app.engines.delivery import DeliveryEngine
    from app.engines.holistic import HolisticEvaluator
    from app.services.llm import get_paraphrase_service
    from app.services.judge_config import mask_secrets_in_text
    from app.services.stt import MockSTTService

    settings = get_settings()
    content_engine = ContentEngine(llm=llm_service, db=None)
    delivery_engine = DeliveryEngine()
    generator = VariantGenerator(seed=seed)

    try:
        para_service = get_paraphrase_service()
    except Exception:
        para_service = llm_service  # mock fallback only

    rows: list[BenchmarkRow] = []
    consecutive_failures = 0

    def _is_fatal(exc: Exception) -> bool:
        """Non-retryable exceptions — abort immediately."""
        return isinstance(exc, (PermissionError, LookupError, ValueError))

    def _safe_msg(exc: Exception) -> str:
        return mask_secrets_in_text(str(exc), secrets)[:200]

    def _add(test_name, metric_name, value, passed, threshold, details=None):
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
            details=json.dumps(details or {}),
        ))

    def _call(fn, label: str):
        """Execute fn(), track failures, abort judge on fatal error or too many retries."""
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
                    f"Judge aborted after {_MAX_CONSECUTIVE_FAILURES} consecutive "
                    f"failures on {label}: {safe}"
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

            # ── Baseline: holistic ────────────────────────────────────────────
            def _holistic():
                h = HolisticEvaluator(llm=llm_service).evaluate(
                    item.answer_text, item.question_text
                )
                _add("baseline_holistic", "holistic_score", h.score, True, 0.0,
                     {"reasoning": h.reasoning})
            _call(_holistic, "baseline_holistic")

            # ── Baseline: rubric ──────────────────────────────────────────────
            def _rubric():
                c_result = content_engine.analyze(
                    transcript_text=item.answer_text,
                    question_text=item.question_text,
                    rubric_criteria=rubric_criteria,
                )
                _add("baseline_rubric", "rubric_score", c_result.overall_score, True, 0.0)
            res = _call(_rubric, "baseline_rubric")
            if res is None and consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError("Aborted after baseline_rubric failures")

            # ── Test 1: Delivery Invariance ───────────────────────────────────
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
                     di.passed, di.threshold, di.details)
            _call(_di, "delivery_invariance")

            # ── Test 2: Content Sensitivity ───────────────────────────────────
            def _cs():
                concepts = [c.expected_concepts[0]
                            for c in rubric_criteria if c.expected_concepts][:3]
                if not concepts:
                    return
                deg_v = generator.quality_degradation(
                    item.answer_text, words, concepts_to_remove=concepts
                )
                cs = ContentSensitivityTest(
                    content_engine, delivery_engine,
                    threshold=settings_override.get(
                        "content_sensitivity_min", settings.audit_content_sensitivity_min
                    ),
                ).run(item.answer_text, item.question_text, rubric_criteria, [deg_v])
                _add(cs.test_name, "content_sensitivity", cs.metric_value,
                     cs.passed, cs.threshold, cs.details)
            _call(_cs, "content_sensitivity")

            # ── Test 3: Paraphrase Stability ──────────────────────────────────
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
                     ps.passed, ps.threshold, ps.details)
            _call(_ps, "paraphrase_stability")

            # ── Test 4: Repeated Run ──────────────────────────────────────────
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

            # ── Test 5: STT Sensitivity ───────────────────────────────────────
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

            # ── Ablation: content + delivery features ─────────────────────────
            def _ablation():
                delivery_result = delivery_engine.analyze(words)
                delivery_features = {
                    "speaking_rate_wpm": delivery_result.speaking_rate_wpm,
                    "filler_density": delivery_result.filler_density,
                    "pause_count": delivery_result.pause_count,
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

    except RuntimeError as abort_exc:
        safe = mask_secrets_in_text(str(abort_exc), secrets)
        click.secho(f"  ✗ Judge {judge_name} ABORTED: {safe}", fg="red")
        return JudgeRunResult(
            judge_name=judge_name,
            judge_model=llm_service.model_name,
            status="failed",
            failure_reason=safe,
            rows=[],
        )

    return JudgeRunResult(
        judge_name=judge_name,
        judge_model=llm_service.model_name,
        status="ok",
        rows=rows,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(
    rows: list[BenchmarkRow],
    out_dir: Path,
    config: dict,
    judge_results: list[JudgeRunResult],
) -> None:
    import pandas as pd
    from app.analysis.plots import (
        plot_audit_summary, plot_model_comparison,
        plot_repeatability, plot_score_deltas,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    if not rows:
        click.echo("  (no rows to write — all judges failed)")
        return

    fieldnames = list(BenchmarkRow.__dataclass_fields__.keys())
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    df = pd.DataFrame([asdict(r) for r in rows])

    summary: dict = {"per_test": {}, "per_judge": {}}
    for test_name, grp in df.groupby("test_name"):
        summary["per_test"][test_name] = {
            "pass_rate": float(grp["passed"].mean()),
            "mean_metric": float(grp["metric_value"].mean()),
            "std_metric": float(grp["metric_value"].std()),
            "n": int(len(grp)),
        }
    for judge, grp in df.groupby("judge_name"):
        summary["per_judge"][judge] = {
            "model": grp["judge_model"].iloc[0],
            "provider": grp["judge_provider"].iloc[0],
            "overall_pass_rate": float(grp["passed"].mean()),
            "mean_metric": float(grp["metric_value"].mean()),
        }

    # Add failed judges to summary
    failed = [jr for jr in judge_results if jr.status == "failed"]
    if failed:
        summary["failed_judges"] = [
            {"judge_name": jr.judge_name, "reason": jr.failure_reason}
            for jr in failed
        ]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plots (best-effort)
    try:
        delta_df = df[df["test_name"] == "delivery_invariance"][["judge_name", "metric_value"]].copy()
        delta_df.columns = ["variant_type", "delta"]
        if not delta_df.empty:
            plot_score_deltas(delta_df, out_dir)
    except Exception:
        pass
    try:
        rr_df = df[df["test_name"] == "repeated_run"]
        if not rr_df.empty:
            scores_list = json.loads(rr_df.iloc[0]["details"]).get("scores", [])
            if scores_list:
                plot_repeatability(scores_list, out_dir)
    except Exception:
        pass
    try:
        baseline_df = df[df["test_name"] == "baseline_rubric"][["judge_name", "metric_value"]].copy()
        baseline_df.columns = ["model", "overall_score"]
        if not baseline_df.empty:
            plot_model_comparison(baseline_df, out_dir)
    except Exception:
        pass
    try:
        audit_tests = [
            "delivery_invariance", "content_sensitivity",
            "paraphrase_stability", "repeated_run", "stt_sensitivity",
        ]
        audit_vals, audit_thresh = {}, {}
        for t in audit_tests:
            sub = df[df["test_name"] == t]
            if not sub.empty:
                audit_vals[t] = float(sub["metric_value"].mean())
                audit_thresh[t] = float(sub["threshold"].iloc[0])
        if audit_vals:
            plot_audit_summary(audit_vals, audit_thresh, out_dir)
    except Exception:
        pass

    click.echo(f"\n  ✓ Results : {out_dir}/results.csv")
    click.echo(f"  ✓ Summary : {out_dir}/summary.json")
    click.echo(f"  ✓ Config  : {out_dir}/config.json")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--answers-dir", required=True, type=click.Path(exists=True))
@click.option("--judges", default="all", show_default=True,
              help="'all', 'mock', 'debug', or comma-separated judge names")
@click.option("--runs", default=5, show_default=True)
@click.option("--output-dir", default="experiments", show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option(
    "--tag", default="pilot",
    type=click.Choice(["debug", "mock", "pilot", "final"], case_sensitive=False),
    show_default=True,
    help="debug=not logged; mock=mock only; pilot=real/exploratory; final=clean tree required",
)
@click.option("--delivery-invariance-max", default=None, type=float)
@click.option("--content-sensitivity-min", default=None, type=float)
@click.option("--paraphrase-max-deviation", default=None, type=float)
@click.option("--skip-preflight", is_flag=True,
              help="Skip the pre-flight judge connectivity check (not recommended)")
def main(
    answers_dir: str,
    judges: str,
    runs: int,
    output_dir: str,
    seed: int,
    tag: str,
    delivery_invariance_max: float | None,
    content_sensitivity_min: float | None,
    paraphrase_max_deviation: float | None,
    skip_preflight: bool,
) -> None:
    """Run counterfactual audit benchmark. Keys are never printed or logged."""
    from app.config import get_settings
    from app.services.judge_config import get_judge_registry, mask_secrets_in_text

    settings = get_settings()

    # ── Tag: final requires clean tree ────────────────────────────────────────
    if tag == "final" and _git_dirty():
        click.secho(
            "\nERROR: Working tree has uncommitted changes.\n"
            "Final runs require a clean git working tree.\n"
            "  git status          → see what is uncommitted\n"
            "  git add . && git commit -m 'pre-final-run state'",
            fg="red", bold=True,
        )
        raise SystemExit(1)

    # ── Tag: mock forces mock provider ───────────────────────────────────────
    if tag == "mock":
        os.environ["LLM_PROVIDER"] = "mock"

    # ── Debug tag note ────────────────────────────────────────────────────────
    if tag == "debug":
        click.secho("  [debug] This run will NOT be logged to experiments.md.", fg="yellow")

    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        f"_{tag}"
        f"_{uuid.uuid4().hex[:8]}"
    )

    click.echo(f"\nInterviewIQ Audit Benchmark")
    click.echo(f"  run_id   : {run_id}")
    click.echo(f"  tag      : {tag}")
    click.echo(f"  answers  : {answers_dir}")
    click.echo(f"  judges   : {judges}")
    click.echo(f"  runs     : {runs}")
    click.echo(f"  seed     : {seed}")

    # ── Resolve judges ────────────────────────────────────────────────────────
    try:
        judge_services = resolve_judges(judges)
    except Exception as exc:
        click.secho(str(exc), fg="red")
        raise SystemExit(1)

    click.echo(f"  resolved : {[n for n, _ in judge_services]}\n")

    # ── Load items ────────────────────────────────────────────────────────────
    answers_path = Path(answers_dir)
    items = load_answer_items(answers_path)
    if not items:
        click.secho("No valid answer items found.", fg="red")
        raise SystemExit(1)
    click.echo(f"  items    : {len(items)}\n")

    # ── Collect secrets for masking ───────────────────────────────────────────
    try:
        registry = get_judge_registry()
        secrets = registry.all_secrets()
    except Exception:
        secrets = []

    # ── Pre-flight check ──────────────────────────────────────────────────────
    if not skip_preflight and tag not in ("mock", "debug"):
        click.echo("Pre-flight connectivity check:")
        probe_results = _probe_judges(judge_services, secrets)
        failed_preflight = [n for n, ok in probe_results.items() if not ok]
        if failed_preflight:
            click.secho(
                f"\n  Pre-flight FAILED for: {failed_preflight}\n"
                "  Fix these judges before running the benchmark.\n"
                "  Run `python check_config.py` for details.\n"
                "  Use --skip-preflight to bypass (not recommended).",
                fg="red", bold=True,
            )
            raise SystemExit(1)
        click.echo("  All judges reachable.\n")

    # ── Settings overrides ────────────────────────────────────────────────────
    settings_override: dict = {"repeated_runs": runs}
    if delivery_invariance_max is not None:
        settings_override["delivery_invariance_max"] = delivery_invariance_max
    if content_sensitivity_min is not None:
        settings_override["content_sensitivity_min"] = content_sensitivity_min
    if paraphrase_max_deviation is not None:
        settings_override["paraphrase_max_deviation"] = paraphrase_max_deviation

    # ── Build config snapshot (NO keys) ──────────────────────────────────────
    try:
        judges_safe = [j.safe_info() for j in registry.judges
                       if j.name in {n for n, _ in judge_services}]
        paraphrase_safe = registry.paraphrase.safe_info() if registry.paraphrase else None
    except Exception:
        judges_safe = [{"name": n, "model": svc.model_name, "provider": svc.provider}
                       for n, svc in judge_services]
        paraphrase_safe = None

    git_hash = _git_hash()
    pip_packages = _pip_freeze()

    config_snapshot: dict = {
        "run_id": run_id,
        "tag": tag,
        "status": "running",        # updated to ok/partial/failed at end
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
        },
        "pip_freeze": pip_packages,
        # API keys are NEVER included here
    }

    # Write config early so it exists even if the run crashes
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2))

    # ── Run benchmark per judge ───────────────────────────────────────────────
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

    # ── Determine run status ──────────────────────────────────────────────────
    failed_judges = [jr for jr in judge_results if jr.status == "failed"]
    ok_judges = [jr for jr in judge_results if jr.status == "ok"]

    if not ok_judges:
        run_status = "failed"
    elif failed_judges:
        run_status = "partial"
    else:
        run_status = "ok"

    # Update config with final status and failure details
    config_snapshot["status"] = run_status
    if failed_judges:
        config_snapshot["failed_judges"] = [
            {"judge_name": jr.judge_name, "model": jr.judge_model, "reason": jr.failure_reason}
            for jr in failed_judges
        ]
    (out_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2))

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_outputs(all_rows, out_dir, config_snapshot, judge_results)

    # ── Log to experiments.md (skip for debug runs) ───────────────────────────
    if tag != "debug":
        import pandas as pd
        summary_for_log: dict = {"per_test": {}}
        if all_rows:
            df_s = pd.DataFrame([asdict(r) for r in all_rows])
            for test_name, grp in df_s.groupby("test_name"):
                summary_for_log["per_test"][test_name] = {
                    "pass_rate": float(grp["passed"].mean()),
                }
        _append_to_experiments_log(
            run_id=run_id,
            tag=tag,
            judge_names=[n for n, _ in judge_services],
            n_items=len(items),
            prompt_version=settings.content_engine_prompt_version,
            git_hash=git_hash,
            summary=summary_for_log,
            run_status=run_status,
            failed_judges=[jr.judge_name for jr in failed_judges],
        )
        click.echo(f"  ✓ Logged to : docs/experiments.md  (status={run_status})")
    else:
        click.echo("  [debug] Not logged to experiments.md.")

    # ── Summary table ─────────────────────────────────────────────────────────
    import pandas as pd
    click.echo("\n── Summary ────────────────────────────────────────────────")

    if failed_judges:
        click.secho(
            f"  Aborted judges ({len(failed_judges)}): "
            + ", ".join(jr.judge_name for jr in failed_judges),
            fg="red",
        )

    if all_rows:
        df = pd.DataFrame([asdict(r) for r in all_rows])
        for test_name, grp in df.groupby("test_name"):
            pass_rate = grp["passed"].mean() * 100
            mean_val = grp["metric_value"].mean()
            status_icon = "✓" if pass_rate >= 80 else "✗"
            click.echo(
                f"  {status_icon} {test_name:<35}  pass={pass_rate:.0f}%  mean={mean_val:.4f}"
            )

    status_color = {"ok": "green", "partial": "yellow", "failed": "red"}[run_status]
    click.secho(f"\nRun status : {run_status.upper()}", fg=status_color, bold=True)
    click.echo(f"Run ID     : {run_id}")
    click.echo(f"Output     : {out_dir}/\n")

    # Exit non-zero if the run fully failed (all judges aborted)
    if run_status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
