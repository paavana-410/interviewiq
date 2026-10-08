#!/usr/bin/env python
"""check_config.py — verify all configured judges and the paraphrase model are reachable.

Makes one minimal test call per configured judge/paraphrase and reports OK / FAILED.
API key values are never printed; only masked versions appear in output.

Usage:
    python check_config.py
    python check_config.py --judge judge_gpt4o
    python check_config.py --no-paraphrase
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env before any app imports
from dotenv import load_dotenv  # type: ignore
load_dotenv(override=False)

import click

from app.services.judge_config import get_judge_registry, _mask_key, mask_secrets_in_text
from app.services.llm import OpenAICompatibleService, LLMResponse
from app.services.judge_config import JudgeConfig, ParaphraseConfig


_TEST_PROMPT = (
    'Reply with exactly this JSON and nothing else: {"status": "ok", "model": "self"}'
)


def _test_one(name: str, config: JudgeConfig | ParaphraseConfig, label: str) -> bool:
    """Make a minimal call and return True on success."""
    masked_key = _mask_key(config.api_key)
    click.echo(f"  {label:<30} model={config.model}  key={masked_key}  ... ", nl=False)
    try:
        # Build a JudgeConfig for the OpenAICompatibleService
        if isinstance(config, ParaphraseConfig):
            judge_cfg = JudgeConfig(
                name=config.name,
                provider=config.provider,
                base_url=config.base_url,
                api_key=config.api_key,
                model=config.model,
                rpm=config.rpm,
            )
        else:
            judge_cfg = config

        svc = OpenAICompatibleService(judge_cfg, cache_dir=None)
        resp: LLMResponse = svc._complete_with_retry(
            _TEST_PROMPT, temperature=0.0, max_tokens=64
        )
        # Sanity: response should have some text
        if not resp.text.strip():
            click.secho("FAILED (empty response)", fg="red")
            return False
        click.secho(
            f"OK  ({resp.latency_ms}ms, {resp.input_tokens}→{resp.output_tokens} tokens)",
            fg="green",
        )
        return True
    except Exception as exc:
        # Mask any leaked key value from the error message
        registry = get_judge_registry()
        safe_msg = mask_secrets_in_text(str(exc), registry.all_secrets())
        click.secho(f"FAILED — {safe_msg}", fg="red")
        return False


@click.command()
@click.option("--judge", default=None, help="Test only this judge name (default: all)")
@click.option("--no-paraphrase", is_flag=True, help="Skip paraphrase model check")
def main(judge: str | None, no_paraphrase: bool) -> None:
    """Test connectivity to all configured LLM judges."""
    try:
        registry = get_judge_registry(reload=True)
    except Exception as exc:
        click.secho(f"Failed to load judge registry: {exc}", fg="red")
        raise SystemExit(1)

    if not registry.judges and registry.paraphrase is None:
        click.secho(
            "No judges configured. Set JUDGE_1_NAME, JUDGE_1_PROVIDER, "
            "JUDGE_1_API_KEY, JUDGE_1_MODEL in your .env file.",
            fg="yellow",
        )
        raise SystemExit(1)

    click.echo(f"\nInterviewIQ — Config Check")
    click.echo(f"  Judges configured : {len(registry.judges)}")
    click.echo(
        f"  Paraphrase        : {'configured' if registry.paraphrase else 'not configured'}"
    )
    click.echo()

    results: list[bool] = []

    # ── Judges ────────────────────────────────────────────────────────────────
    judges_to_test = registry.judges
    if judge is not None:
        try:
            judges_to_test = [registry.get_judge(judge)]
        except KeyError as e:
            click.secho(str(e), fg="red")
            raise SystemExit(1)

    click.echo("Judges:")
    for j in judges_to_test:
        ok = _test_one(j.name, j, f"[judge] {j.name}")
        results.append(ok)

    # ── Paraphrase ─────────────────────────────────────────────────────────────
    if not no_paraphrase:
        click.echo("\nParaphrase model:")
        if registry.paraphrase is None:
            click.secho("  Not configured (PARAPHRASE_* missing from .env)", fg="yellow")
        else:
            ok = _test_one(
                registry.paraphrase.name,
                registry.paraphrase,
                f"[paraphrase] {registry.paraphrase.name}",
            )
            results.append(ok)

    # ── Summary ───────────────────────────────────────────────────────────────
    click.echo()
    total = len(results)
    passed = sum(results)
    if passed == total:
        click.secho(f"All {total} check(s) passed.", fg="green", bold=True)
    else:
        click.secho(
            f"{total - passed}/{total} check(s) FAILED. "
            "Fix the failing entries in your .env before running the benchmark.",
            fg="red",
            bold=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
