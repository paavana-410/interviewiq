#!/usr/bin/env python
"""diagnose_content_sensitivity.py — explain WHY content_sensitivity is what it is.

Modes:
  default         : failed count first, then per-item content_sensitivity table.
  --deep [--item] : for one item (default q01): exact original text, exact
                    degraded text with word-level diff, cache keys for both
                    calls, and the raw cached judge output for each.
                    Reads ONLY from cache + results files. No API calls.

Usage:
    python diagnose_content_sensitivity.py
    python diagnose_content_sensitivity.py --run-id 20261103T...
    python diagnose_content_sensitivity.py --deep
    python diagnose_content_sensitivity.py --deep --item q01_stack_queue
    python diagnose_content_sensitivity.py --deep --run-id 20261103T...
"""
from __future__ import annotations

import difflib
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import click
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _latest_run_with_data(experiments_dir: Path) -> str | None:
    dirs = [
        d for d in experiments_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and (d / "results.csv").exists()
    ]
    if not dirs:
        return None
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs[0].name


def _print_call_stats(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        click.secho("  (no summary.json — call stats unavailable)", fg="yellow")
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    call_stats = summary.get("call_stats", {})
    if not call_stats:
        click.secho("  (no call_stats in summary.json — run predates the counter)", fg="yellow")
        return {}
    click.echo("\n── Call stats (failed count FIRST) ─────────────────────────")
    totals = {"total_calls": 0, "truncated": 0, "retried": 0, "failed": 0}
    for judge_name, stats in call_stats.items():
        click.echo(
            f"  {judge_name:<30}  total={stats.get('total_calls',0):<5}  "
            f"truncated={stats.get('truncated',0):<3}  "
            f"retried={stats.get('retried',0):<3}  "
            f"FAILED={stats.get('failed',0)}"
        )
        for k in totals:
            totals[k] += stats.get(k, 0)
    if len(call_stats) > 1:
        click.secho(
            f"  {'TOTAL':<30}  total={totals['total_calls']:<5}  "
            f"truncated={totals['truncated']:<3}  "
            f"retried={totals['retried']:<3}  "
            f"FAILED={totals['failed']}",
            bold=True,
        )
    if totals["failed"] > 0:
        click.secho(
            f"\n  ⚠️  {totals['failed']} call(s) FAILED (returned None, never guessed).\n"
            f"     The table below may be built on PARTIAL DATA.",
            fg="yellow", bold=True,
        )
    else:
        click.secho("  ✓ 0 failed calls — table is on complete data.", fg="green")
    return totals


def _print_table(run_dir: Path, show_details: bool) -> None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        click.secho("  No results.csv.", fg="red")
        return
    df = pd.read_csv(csv_path)
    all_items = sorted(df["item_id"].unique()) if "item_id" in df.columns else []
    cs_rows = df[df["test_name"] == "content_sensitivity"].copy()
    click.echo("\n── content_sensitivity diagnostic ──────────────────────────")
    if cs_rows.empty:
        click.secho("  No content_sensitivity rows — test FAILED for every item.", fg="red", bold=True)
        click.echo(f"\n  Items in run: {len(all_items)}\n  With row: 0")
        return
    click.echo(
        f"\n  Items in run:                          {len(all_items)}\n"
        f"  Items with content_sensitivity row:    {len(cs_rows)}\n"
        f"  Items where test FAILED (no row):       {len(all_items) - len(cs_rows)}",
    )
    if len(all_items) - len(cs_rows) > 0:
        missing = set(all_items) - set(cs_rows["item_id"])
        click.secho(f"  Missing items: {sorted(missing)}", fg="yellow")
    click.echo(
        f"\n  {'item_id':<28} {'orig':>6} {'degraded':>9} {'drop':>7} "
        f"{'thresh':>7} {'pass':>5}  concepts_removed"
    )
    click.echo("  " + "-" * 95)
    for _, row in cs_rows.iterrows():
        item_id = row["item_id"]
        try:
            details = json.loads(row["details"]) if row["details"] else {}
        except Exception:
            details = {}
        orig_score = details.get("original_score")
        for v in details.get("variants", []):
            deg_score = v.get("degraded_score")
            content_drop = v.get("content_drop")
            concepts = v.get("variant_params", {}).get("concepts_removed", [])
            concepts_str = ", ".join(concepts[:4]) if concepts else "(none)"
            def _fmt(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "  ?  "
            pass_icon = "✓" if row["passed"] else "✗"
            click.echo(
                f"  {item_id:<28} {_fmt(orig_score):>6} {_fmt(deg_score):>9} "
                f"{_fmt(content_drop):>7} {row['threshold']:>7.2f} {pass_icon:>5}  {concepts_str}"
            )
            if show_details:
                for c in concepts:
                    click.echo(f"        removed: {c}")
    click.echo(
        "\n  Interpretation:\n"
        "    content_drop = orig - degraded. Should be ≥ threshold (0.15) for PASS.\n"
        "    drop ≈ 0 means the judge gave the SAME score after concepts were removed."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deep mode — exact texts, diff, cache keys, raw cached output
# ─────────────────────────────────────────────────────────────────────────────

_V1_TEMPLATE_FILE = Path("_v1_template.txt")


def _load_v1_template() -> str:
    """Load the byte-exact v1 prompt template extracted from git history."""
    if not _V1_TEMPLATE_FILE.exists():
        raise FileNotFoundError(
            f"{_V1_TEMPLATE_FILE} not found. Run the extraction step first "
            f"(see repo notes) to recover the v1 prompt template from git."
        )
    return _V1_TEMPLATE_FILE.read_text(encoding="utf-8")


def _v1_build_prompt(
    template: str,
    question_text: str,
    transcript_text: str,
    rubric_criteria: list[dict],
    delivery_features: dict | None = None,
) -> str:
    """Reconstruct the v1 prompt byte-for-byte (matches the cache key)."""
    rubric_json = json.dumps(
        [
            {
                "criterion_name": c["criterion_name"],
                "criterion_type": c["criterion_type"],
                "weight": c["weight"],
                "expected_concepts": c["expected_concepts"],
                "star_component": c.get("star_component", False),
                "scoring_guide": c.get("scoring_guide", ""),
            }
            for c in rubric_criteria
        ],
        indent=2,
    )
    delivery_section = ""
    if delivery_features:
        ablation = (
            "## Delivery Features (ablation mode — for research comparison only)\n"
            "{delivery_json}\n"
            "NOTE: You MAY use these features to adjust scoring. This is the ABLATION condition.\n"
        )
        delivery_section = ablation.format(
            delivery_json=json.dumps(delivery_features, indent=2)
        )
    return template.format(
        question_text=question_text,
        transcript_text=transcript_text,
        rubric_json=rubric_json,
        delivery_section=delivery_section,
    )


def _cache_key(model: str, prompt_version: str, prompt: str) -> str:
    raw = f"{model}:{prompt_version}:{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.json"


def _read_cache(cache_dir: Path, key: str) -> dict | None:
    p = _cache_path(cache_dir, key)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _word_diff(orig: str, degr: str) -> list[str]:
    """Return a unified word-level diff."""
    orig_words = orig.split()
    degr_words = degr.split()
    diff = difflib.unified_diff(
        orig_words, degr_words,
        fromfile="original", tofile="degraded",
        lineterm="",
    )
    return list(diff)


def _deep(run_dir: Path, item_id: str) -> None:
    """For one item: exact texts, diff, cache keys, raw cached outputs."""
    from app.audit.variants import VariantGenerator
    from app.services.stt import MockSTTService

    # Config for this run: model + prompt version
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        click.secho("  No config.json — cannot determine model/prompt version.", fg="red")
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    prompt_version = cfg.get("base_settings", {}).get(
        "content_engine_prompt_version", "v1"
    )
    judges = cfg.get("judges", [])
    if not judges:
        click.secho("  No judge in config.json.", fg="red")
        return
    model = judges[0]["model"]

    click.echo("\n── Deep diagnostic ─────────────────────────────────────────")
    click.echo(f"  run_id        : {run_dir.name}")
    click.echo(f"  item          : {item_id}")
    click.echo(f"  judge model   : {model}")
    click.echo(f"  prompt_version: {prompt_version}")

    # Load the answer item
    item_dir = Path("data/benchmark_answers") / item_id
    if not item_dir.exists():
        click.secho(f"  Item dir not found: {item_dir}", fg="red")
        return
    answer_text = (item_dir / "answer.txt").read_text(encoding="utf-8").strip()
    question_text = (item_dir / "question.txt").read_text(encoding="utf-8").strip()
    rubric_criteria = json.loads(
        (item_dir / "rubric.json").read_text(encoding="utf-8")
    )

    # Determine concepts to remove (same logic as the benchmark runner)
    concepts = [
        c["expected_concepts"][0]
        for c in rubric_criteria
        if c.get("expected_concepts")
    ][:3]
    click.echo(f"  concepts_to_remove: {concepts}")

    # Build the degraded variant (deterministic, no API)
    gen = VariantGenerator(seed=42)
    stt_mock = MockSTTService(fixed_transcript=answer_text)
    words = stt_mock.transcribe("synthetic").words
    variant = gen.quality_degradation(
        answer_text, words, concepts_to_remove=concepts
    )
    degraded_text = variant.variant_text

    # (1) exact original text
    click.echo("\n── (1) Original text sent to the judge ─────────────────────")
    click.echo(f'  """\n  {answer_text}\n  """')

    # (2) exact degraded text + word diff
    click.echo("\n── (2) Degraded text sent to the judge ────────────────────")
    click.echo(f'  """\n  {degraded_text}\n  """')

    click.echo("\n  Word-level diff (original → degraded):")
    diff = _word_diff(answer_text, degraded_text)
    if not diff:
        click.secho("  (no differences — texts are IDENTICAL)", fg="yellow", bold=True)
    else:
        for line in diff:
            click.echo(f"  {line}")
    # Explicit check: did any concept actually appear in the original?
    click.echo("\n  Concept presence in original text (case-sensitive):")
    for c in concepts:
        present = c in answer_text
        icon = "✓ present" if present else "✗ ABSENT"
        click.echo(f"    {c:<20} {icon}")

    # (3) cache keys
    click.echo("\n── (3) Cache keys ─────────────────────────────────────────")
    template = _load_v1_template()
    orig_prompt = _v1_build_prompt(
        template, question_text, answer_text, rubric_criteria, None
    )
    degr_prompt = _v1_build_prompt(
        template, question_text, degraded_text, rubric_criteria, None
    )
    orig_key = _cache_key(model, prompt_version, orig_prompt)
    degr_key = _cache_key(model, prompt_version, degr_prompt)
    click.echo(f"  original  key: {orig_key}")
    click.echo(f"  degraded  key: {degr_key}")
    click.echo(f"  keys differ : {orig_key != degr_key}")
    if orig_key == degr_key:
        click.secho(
            "  → prompts are byte-identical, so the cache returns ONE response for both.",
            fg="yellow",
        )

    # (4) raw cached outputs
    cache_dir = Path("experiments/.llm_cache")
    click.echo("\n── (4) Raw cached judge output ────────────────────────────")

    for label, key in [("original", orig_key), ("degraded", degr_key)]:
        click.echo(f"\n  [{label}] cache file: {_cache_path(cache_dir, key)}")
        cached = _read_cache(cache_dir, key)
        if cached is None:
            click.secho(f"    NOT FOUND in cache (no file at this key).", fg="red")
            continue
        click.echo(f"    model_name    : {cached.get('model_name')}")
        click.echo(f"    input_tokens  : {cached.get('input_tokens')}")
        click.echo(f"    output_tokens : {cached.get('output_tokens')}")
        click.echo(f"    finish_reason : {cached.get('finish_reason')}")
        click.echo(f"    latency_ms    : {cached.get('latency_ms')}")
        raw_text = cached.get("text", "")
        click.echo(f"    raw text ({len(raw_text)} chars):")
        # Pretty-print if it's JSON
        try:
            parsed = json.loads(raw_text)
            click.echo("    parsed JSON:")
            click.echo(json.dumps(parsed, indent=2))
        except Exception:
            click.echo(f'    """\n    {raw_text[:800]}\n    """')

    click.echo("\n── end deep diagnostic ─────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--run-id", default=None, help="Run to diagnose (default: latest with data)")
@click.option("--details", is_flag=True, help="Show each removed concept on its own line")
@click.option("--deep", is_flag=True, help="Deep diagnostic for one item: texts, diff, cache keys, raw output")
@click.option("--item", default="q01_stack_queue", show_default=True,
              help="Item id for --deep mode")
def main(run_id: str | None, details: bool, deep: bool, item: str) -> None:
    experiments_dir = Path("experiments")
    if run_id is None:
        run_id = _latest_run_with_data(experiments_dir)
        if run_id is None:
            click.secho("No runs with results.csv found.", fg="red")
            raise SystemExit(1)
        click.echo(f"Using latest run with data: {run_id}")
    else:
        click.echo(f"Run: {run_id}")

    run_dir = experiments_dir / run_id
    if not run_dir.exists():
        click.secho(f"Run directory not found: {run_dir}", fg="red")
        raise SystemExit(1)

    config_path = run_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        click.echo(
            f"\n  prompt_version : {cfg.get('base_settings',{}).get('content_engine_prompt_version','?')}"
        )
        click.echo(f"  tag            : {cfg.get('tag','?')}")
        click.echo(f"  status         : {cfg.get('status','?')}")
        judges = cfg.get("judges", [])
        if judges:
            click.echo(f"  judge          : {judges[0].get('name','?')} / {judges[0].get('model','?')}")

    if deep:
        _deep(run_dir, item)
        return

    _print_call_stats(run_dir)
    _print_table(run_dir, show_details=details)


if __name__ == "__main__":
    main()
