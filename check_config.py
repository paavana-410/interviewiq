#!/usr/bin/env python
"""check_config.py — verify all configured judges and the paraphrase model are reachable.

Uses raw httpx calls (not the OpenAI SDK) so timeouts are reliable.
API key values are never printed; only masked versions appear in output.

Usage:
    python check_config.py
    python check_config.py --judge judge_gemini
    python check_config.py --no-paraphrase
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv  # type: ignore
load_dotenv(override=False)

import click
import httpx

from app.services.judge_config import (
    JudgeConfig, ParaphraseConfig,
    get_judge_registry, _mask_key, mask_secrets_in_text,
)

_TEST_PROMPT = 'Reply with exactly this JSON and nothing else: {"status": "ok"}'
_TIMEOUT = 20  # seconds per request


def _call_openai_compat(
    base_url: str,
    api_key: str,
    model: str,
    is_gemini_aq: bool,
) -> tuple[int, dict]:
    """Make a raw chat completions POST. Returns (status_code, response_dict)."""
    url = base_url.rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    params = {}

    if is_gemini_aq:
        # AQ. keys need x-goog-api-key header + Authorization: Bearer with the same key
        params["key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-goog-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": _TEST_PROMPT}],
        "max_tokens": 32,
        "temperature": 0.0,
    }

    t0 = time.monotonic()
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.post(url, headers=headers, params=params, json=body)
    latency_ms = int((time.monotonic() - t0) * 1000)
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text[:300]}
    return r.status_code, data, latency_ms


def _test_one(config: JudgeConfig | ParaphraseConfig, label: str) -> bool:
    masked_key = _mask_key(config.api_key)
    click.echo(f"  {label:<35} model={config.model:<40} key={masked_key}  ... ", nl=False)

    is_gemini_aq = (
        config.api_key.startswith("AQ.")
        and "googleapis.com" in (config.base_url or "")
    )

    try:
        status, data, latency_ms = _call_openai_compat(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            is_gemini_aq=is_gemini_aq,
        )
    except httpx.TimeoutException:
        click.secho(f"FAILED — Timed out after {_TIMEOUT}s (network issue or wrong BASE_URL)", fg="red")
        return False
    except httpx.ConnectError as e:
        click.secho(f"FAILED — Cannot connect: {e}", fg="red")
        return False
    except Exception as exc:
        try:
            registry = get_judge_registry()
            safe = mask_secrets_in_text(str(exc), registry.all_secrets())
        except Exception:
            safe = str(exc)
        click.secho(f"FAILED — {safe[:120]}", fg="red")
        return False

    # Mask key from response just in case
    try:
        registry = get_judge_registry()
        safe_data = mask_secrets_in_text(json.dumps(data), registry.all_secrets())
    except Exception:
        safe_data = str(data)[:200]

    if status == 200:
        # Extract reply text
        try:
            reply = data["choices"][0]["message"]["content"]
            tokens_in = data.get("usage", {}).get("prompt_tokens", "?")
            tokens_out = data.get("usage", {}).get("completion_tokens", "?")
            click.secho(f"OK  ({latency_ms}ms, {tokens_in}in → {tokens_out}out)  reply={reply[:40]!r}", fg="green")
        except Exception:
            click.secho(f"OK  ({latency_ms}ms)", fg="green")
        return True
    elif status == 401:
        click.secho(f"FAILED — 401 Invalid API key. Check your .env.", fg="red")
        return False
    elif status == 400:
        err_msg = data.get("error", {}).get("message", safe_data[:100]) if isinstance(data, dict) else safe_data[:100]
        click.secho(f"FAILED — 400 Bad request: {err_msg}", fg="red")
        return False
    elif status == 404:
        click.secho(f"FAILED — 404 Model not found: {config.model}", fg="red")
        return False
    elif status == 429:
        err_msg = data.get("error", {}).get("message", "Rate limit") if isinstance(data, dict) else "Rate limit"
        click.secho(f"FAILED — 429 {err_msg[:100]}", fg="red")
        return False
    else:
        click.secho(f"FAILED — HTTP {status}: {safe_data[:120]}", fg="red")
        return False


@click.command()
@click.option("--judge", default=None, help="Test only this judge by name (default: all)")
@click.option("--no-paraphrase", is_flag=True, help="Skip paraphrase model check")
def main(judge: str | None, no_paraphrase: bool) -> None:
    """Test connectivity to all configured LLM judges."""
    try:
        registry = get_judge_registry(reload=True)
    except Exception as exc:
        click.secho(f"Failed to load judge registry: {exc}", fg="red")
        raise SystemExit(1)

    if not registry.judges and registry.paraphrase is None:
        click.secho("No judges configured in .env.", fg="yellow")
        raise SystemExit(1)

    click.echo(f"\nInterviewIQ — Config Check")
    click.echo(f"  Judges configured : {len(registry.judges)}")
    click.echo(f"  Paraphrase        : {'configured' if registry.paraphrase else 'not configured'}")
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
        ok = _test_one(j, f"[judge] {j.name}")
        results.append(ok)

    # ── Paraphrase ─────────────────────────────────────────────────────────────
    if not no_paraphrase:
        click.echo("\nParaphrase model:")
        if registry.paraphrase is None:
            click.secho("  Not configured", fg="yellow")
        else:
            ok = _test_one(registry.paraphrase, f"[paraphrase] {registry.paraphrase.name}")
            results.append(ok)

    # ── Summary ───────────────────────────────────────────────────────────────
    click.echo()
    total = len(results)
    passed = sum(results)
    if passed == total and total > 0:
        click.secho(f"All {total} check(s) passed.", fg="green", bold=True)
    else:
        click.secho(f"{total - passed}/{total} check(s) FAILED.", fg="red", bold=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
