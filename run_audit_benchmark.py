#!/usr/bin/env python
"""run_audit_benchmark.py — Research benchmark CLI for InterviewIQ.

Usage:
    # Run all configured judges (default)
    python run_audit_benchmark.py --answers-dir data/benchmark_answers

    # Run specific judges by name
    python run_audit_benchmark.py --answers-dir data/benchmark_answers \\
        --judges judge_gpt4o,judge_claude

    # Legacy --models flag still works for mock / CI
    python run_audit_benchmark.py --answers-dir data/benchmark_answers \\
        --judges mock --runs 3

Input folder layout (one sub-folder per answer item):
    data/benchmark_answers/
        q01_stack_queue/
            answer.txt          # plain text answer
            question.txt        # interview question
            rubric.json         # list of rubric criteria
            gold_transcript.txt # (optional) gold transcript for STT test
            audio.wav           # (optional) audio for STT test

Outputs (per run):
    experiments/<run_id>/
        config.json      — full config snapshot (judge provider + model, no keys)
        results.csv      — per-answer per-test per-judge metrics
        summary.json     — aggregated stats
        *.png            — plots
"""
from __future__ import annotations

import csv
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Load .env before any app imports
from dotenv import load_dotenv  # type: ignore
load_dotenv(override=False)

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


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
            if rubric_file.exists()
            else []
        )
        gold = (
            (sub / "gold_transcript.txt").read_text(encoding="utf-8").strip()
            if (sub / "gold_transcript.txt").exists()
            else None
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
# Resolve judges from --judges flag
# ─────────────────────────────────────────────────────────────────────────────

def resolve_judges(judges_flag: str) -> list[tuple[str, "LLMService"]]:  # type: ignore[name-defined]
    """Return [(judge_name, service), ...] based on --judges flag.

    'all'  → all configured judges from the registry
    'mock' → single MockLLMService
    'name1,name2' → named subset from registry
    """
    from app.services.llm import MockLLMService, get_all_judge_services
    from app.services.judge_config import get_judge_registry

    if judges_flag.strip().lower() == "mock":
        return [("mock", MockLLMService())]

    all_services = dict(get_all_judge_services())

    if judges_flag.strip().lower() == "all":
        return list(all_services.items())

    names = [n.strip() for n in judges_flag.split(",") if n.strip()]
    result = []
    for name in names:
        if name not in all_services:
            available = list(all_services.keys())
            raise click.ClickException(
                f"Judge {name!r} not found. Available: {available}"
            )
        result.append((name, all_services[name]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-judge benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark_for_judge(
    items: list[AnswerItem],
    judge_name: str,
    llm_service,
    run_id: str,
    settings_override: dict,
    seed: int = 42,
) -> list[BenchmarkRow]:
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
    from app.services.stt import MockSTTService

    settings = get_settings()
    content_engine = ContentEngine(llm=llm_service, db=None)
    delivery_engine = DeliveryEngine()
    generator = VariantGenerator(seed=seed)

    # Paraphrase service (separate model, validated ≠ judge)
    try:
        para_service = get_paraphrase_service()
    except Exception:
        para_service = llm_service  # fallback to same for mock runs only

    rows: list[BenchmarkRow] = []

    def _add(test_name, metric_name, value, passed, threshold, details={}):
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
            details=json.dumps(details),
        ))

    for item in items:
        click.echo(
            f"    item={item.item_id:<30} judge={judge_name}  model={llm_service.model_name}"
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

        # Synthetic timestamps from answer text
        stt_mock = MockSTTService(fixed_transcript=item.answer_text)
        stt_result = stt_mock.transcribe("synthetic")
        words = stt_result.words

        # ── Baseline: holistic score ──────────────────────────────────────────
        try:
            holistic = HolisticEvaluator(llm=llm_service)
            h = holistic.evaluate(item.answer_text, item.question_text)
            _add("baseline_holistic", "holistic_score", h.score, True, 0.0,
                 {"reasoning": h.reasoning})
        except Exception as e:
            click.echo(f"      [warn] holistic: {e}")

        # ── Baseline: rubric without audit ────────────────────────────────────
        try:
            c_result = content_engine.analyze(
                transcript_text=item.answer_text,
                question_text=item.question_text,
                rubric_criteria=rubric_criteria,
            )
            _add("baseline_rubric", "rubric_score", c_result.overall_score, True, 0.0)
        except Exception as e:
            click.echo(f"      [warn] rubric baseline: {e}")
            continue

        # ── Test 1: Delivery Invariance ────────────────────────────────────────
        try:
            variants = [
                generator.filler_injection(words, filler_rate=0.10),
                generator.pause_insertion(words, num_pauses=4),
                generator.pace_change(words, factor=1.6),
            ]
            di_test = DeliveryInvarianceTest(
                content_engine,
                threshold=settings_override.get(
                    "delivery_invariance_max", settings.audit_delivery_invariance_max
                ),
            )
            di = di_test.run(item.answer_text, item.question_text, rubric_criteria, variants)
            _add(di.test_name, "delivery_invariance_error", di.metric_value, di.passed,
                 di.threshold, di.details)
        except Exception as e:
            click.echo(f"      [warn] delivery_invariance: {e}")

        # ── Test 2: Content Sensitivity ────────────────────────────────────────
        try:
            concepts = [
                c.expected_concepts[0]
                for c in rubric_criteria
                if c.expected_concepts
            ][:3]
            if concepts:
                deg_v = generator.quality_degradation(
                    item.answer_text, words, concepts_to_remove=concepts
                )
                cs_test = ContentSensitivityTest(
                    content_engine,
                    delivery_engine,
                    threshold=settings_override.get(
                        "content_sensitivity_min", settings.audit_content_sensitivity_min
                    ),
                )
                cs = cs_test.run(
                    item.answer_text, item.question_text, rubric_criteria, [deg_v]
                )
                _add(cs.test_name, "content_sensitivity", cs.metric_value, cs.passed,
                     cs.threshold, cs.details)
        except Exception as e:
            click.echo(f"      [warn] content_sensitivity: {e}")

        # ── Test 3: Paraphrase Stability ───────────────────────────────────────
        try:
            para_v = generator.paraphrase(
                item.answer_text, words, llm_service=para_service
            )
            ps_test = ParaphraseStabilityTest(
                content_engine,
                threshold=settings_override.get(
                    "paraphrase_max_deviation", settings.audit_paraphrase_max_deviation
                ),
            )
            ps = ps_test.run(
                item.answer_text, item.question_text, rubric_criteria, [para_v]
            )
            _add(ps.test_name, "paraphrase_deviation", ps.metric_value, ps.passed,
                 ps.threshold, ps.details)
        except Exception as e:
            click.echo(f"      [warn] paraphrase_stability: {e}")

        # ── Test 4: Repeated Run ───────────────────────────────────────────────
        try:
            n_runs = settings_override.get("repeated_runs", settings.audit_repeated_runs)
            rr_test = RepeatedRunTest(
                content_engine,
                n_runs=n_runs,
                threshold=settings_override.get(
                    "repeatability_max_variance", settings.audit_repeatability_max_variance
                ),
            )
            rr = rr_test.run(item.answer_text, item.question_text, rubric_criteria)
            _add(rr.test_name, "repeatability_variance", rr.metric_value, rr.passed,
                 rr.threshold, rr.details)
        except Exception as e:
            click.echo(f"      [warn] repeated_run: {e}")

        # ── Test 5: STT Sensitivity ────────────────────────────────────────────
        if item.gold_transcript and item.audio_path:
            try:
                whisper_stt = MockSTTService(fixed_transcript=item.answer_text)
                w_result = whisper_stt.transcribe(item.audio_path)
                stt_test = STTSensitivityTest(
                    content_engine,
                    threshold=settings_override.get(
                        "stt_max_drift", settings.audit_stt_max_drift
                    ),
                )
                st = stt_test.run(
                    item.gold_transcript, w_result.text,
                    item.question_text, rubric_criteria,
                )
                _add(st.test_name, "stt_sensitivity", st.metric_value, st.passed,
                     st.threshold, st.details)
            except Exception as e:
                click.echo(f"      [warn] stt_sensitivity: {e}")

        # ── Ablation: content + delivery features ──────────────────────────────
        try:
            delivery_result = delivery_engine.analyze(words)
            delivery_features = {
                "speaking_rate_wpm": delivery_result.speaking_rate_wpm,
                "filler_density": delivery_result.filler_density,
                "pause_count": delivery_result.pause_count,
            }
            import app.config as _cfg
            _original = _cfg._settings.content_engine_include_delivery
            _cfg._settings.content_engine_include_delivery = True
            c_ablation = content_engine.analyze(
                transcript_text=item.answer_text,
                question_text=item.question_text,
                rubric_criteria=rubric_criteria,
                delivery_features=delivery_features,
            )
            _cfg._settings.content_engine_include_delivery = _original
            _add("ablation_with_delivery", "rubric_score_with_delivery",
                 c_ablation.overall_score, True, 0.0)
        except Exception as e:
            click.echo(f"      [warn] ablation: {e}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(
    rows: list[BenchmarkRow],
    out_dir: Path,
    config: dict,
) -> None:
    import pandas as pd
    from app.analysis.plots import (
        plot_audit_summary,
        plot_model_comparison,
        plot_repeatability,
        plot_score_deltas,
    )
    from app.config import get_settings

    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

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

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    settings = get_settings()

    # Plots (best-effort)
    try:
        delta_df = df[df["test_name"] == "delivery_invariance"][
            ["judge_name", "metric_value"]
        ].copy()
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
        baseline_df = df[df["test_name"] == "baseline_rubric"][
            ["judge_name", "metric_value"]
        ].copy()
        baseline_df.columns = ["model", "overall_score"]
        if not baseline_df.empty:
            plot_model_comparison(baseline_df, out_dir)
    except Exception:
        pass

    try:
        audit_tests = [
            "delivery_invariance", "content_sensitivity", "paraphrase_stability",
            "repeated_run", "stt_sensitivity",
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
@click.option(
    "--answers-dir", required=True, type=click.Path(exists=True),
    help="Folder of benchmark answer items",
)
@click.option(
    "--judges", default="all", show_default=True,
    help=(
        "'all' = every configured judge, "
        "'mock' = no-key test mode, "
        "or comma-separated judge names e.g. 'judge_gpt4o,judge_claude'"
    ),
)
@click.option("--runs", default=5, show_default=True, help="Repeated runs for consistency test")
@click.option("--output-dir", default="experiments", show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--delivery-invariance-max", default=None, type=float)
@click.option("--content-sensitivity-min", default=None, type=float)
@click.option("--paraphrase-max-deviation", default=None, type=float)
def main(
    answers_dir: str,
    judges: str,
    runs: int,
    output_dir: str,
    seed: int,
    delivery_invariance_max: float | None,
    content_sensitivity_min: float | None,
    paraphrase_max_deviation: float | None,
) -> None:
    """Run all 5 counterfactual audit tests across configured LLM judges."""
    from app.config import get_settings
    from app.services.judge_config import get_judge_registry, mask_secrets_in_text

    settings = get_settings()
    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        f"_{uuid.uuid4().hex[:8]}"
    )

    click.echo(f"\nInterviewIQ Audit Benchmark  run_id={run_id}")
    click.echo(f"  answers_dir : {answers_dir}")
    click.echo(f"  judges flag : {judges}")
    click.echo(f"  runs        : {runs}")
    click.echo(f"  seed        : {seed}")

    # Resolve judge services
    try:
        judge_services = resolve_judges(judges)
    except Exception as exc:
        click.secho(str(exc), fg="red")
        raise SystemExit(1)

    click.echo(f"  judges resolved: {[n for n, _ in judge_services]}\n")

    # Load answer items
    answers_path = Path(answers_dir)
    items = load_answer_items(answers_path)
    if not items:
        click.secho("No valid answer items found.", fg="red")
        raise SystemExit(1)
    click.echo(f"  items loaded: {len(items)}\n")

    settings_override: dict = {"repeated_runs": runs}
    if delivery_invariance_max is not None:
        settings_override["delivery_invariance_max"] = delivery_invariance_max
    if content_sensitivity_min is not None:
        settings_override["content_sensitivity_min"] = content_sensitivity_min
    if paraphrase_max_deviation is not None:
        settings_override["paraphrase_max_deviation"] = paraphrase_max_deviation

    # Build config snapshot — NO key values
    try:
        registry = get_judge_registry()
        judges_safe = [j.safe_info() for j in registry.judges]
        paraphrase_safe = registry.paraphrase.safe_info() if registry.paraphrase else None
    except Exception:
        judges_safe = [{"name": n, "model": svc.model_name, "provider": svc.provider}
                       for n, svc in judge_services]
        paraphrase_safe = None

    config_snapshot = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    }

    # Run benchmark
    all_rows: list[BenchmarkRow] = []
    for judge_name, svc in judge_services:
        click.echo(f"  ── Judge: {judge_name}  ({svc.model_name})")
        try:
            rows = run_benchmark_for_judge(
                items=items,
                judge_name=judge_name,
                llm_service=svc,
                run_id=run_id,
                settings_override=settings_override,
                seed=seed,
            )
            all_rows.extend(rows)
        except Exception as exc:
            try:
                registry = get_judge_registry()
                safe = mask_secrets_in_text(str(exc), registry.all_secrets())
            except Exception:
                safe = str(exc)
            click.secho(f"  Judge {judge_name} failed: {safe}", fg="red")

    out_dir = Path(output_dir) / run_id
    write_outputs(all_rows, out_dir, config_snapshot)

    # Summary table
    import pandas as pd
    df = pd.DataFrame([asdict(r) for r in all_rows])
    click.echo("\n── Summary ────────────────────────────────────────────────")
    for test_name, grp in df.groupby("test_name"):
        pass_rate = grp["passed"].mean() * 100
        mean_val = grp["metric_value"].mean()
        status = "✓" if pass_rate >= 80 else "✗"
        click.echo(
            f"  {status} {test_name:<35}  pass={pass_rate:.0f}%  mean={mean_val:.4f}"
        )
    click.echo(f"\nRun ID : {run_id}")
    click.echo(f"Output : {out_dir}/\n")


if __name__ == "__main__":
    main()
