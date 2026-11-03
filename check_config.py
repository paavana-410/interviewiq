#!/usr/bin/env python
"""check_config.py — verify all configured judges and the paraphrase model.

Makes one minimal test call per judge using raw httpx (not the OpenAI SDK,
which hangs on certain providers).

API key values are NEVER printed, logged, or included in any output.
Only masked versions (first-4…last-4) appear.

Usage:
    python check_config.py                    # all judges + paraphrase
    python check_config.py --judge judge_gemini
    python check_config.py --no-paraphrase
    python check_config.py --quiet            # exit 0/1 only, no output

Programmatic use (called by run_audit_benchmark.py):
    from check_config import probe_judge
    ok, status_code, message = probe_judge(judge_config)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv  # type: ignore
load_dotenv(override=False)

import click
import httpx

from app.services.judge_config import (
    JudgeConfig,
    ParaphraseConfig,
    _mask_key,
    get_judge_registry,
    mask_secrets_in_text,
)

_TEST_PROMPT = 'Reply with exactly this JSON and nothing else: {"status": "ok"}'
_TIMEOUT_SEC = 20


# ─────────────────────────────────────────────────────────────────────────────
# Core probe — returns structured result, never raises, never prints
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    ok: bool
    status_code: int | None   # HTTP status, or None on network error
    message: str              # safe to print — no key values
    latency_ms: int | None
    tokens_in: int | None
    tokens_out: int | None


def probe_judge(config: JudgeConfig | ParaphraseConfig) -> ProbeResult:
    """Make one test call to a judge endpoint.

    Returns a ProbeResult. Never raises. Never includes key values in output.
    """
    secrets = [config.api_key]

    is_gemini_aq = (
        config.api_key.startswith("AQ.")
        and "googleapis.com" in (config.base_url or "")
    )

    url = (config.base_url or "").rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    params: dict[str, str] = {}

    if is_gemini_aq:
        headers["Authorization"] = f"Bearer {config.api_key}"
        headers["x-goog-api-key"] = config.api_key
    else:
        headers["Authorization"] = f"Bearer {config.api_key}"

    body = {
        "model": config.model,
        "messages": [{"role": "user", "content": _TEST_PROMPT}],
        "max_tokens": 32,
        "temperature": 0.0,
    }

    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=_TIMEOUT_SEC) as client:
            r = client.post(url, headers=headers, params=params, json=body)
        latency_ms = int((time.monotonic() - t0) * 1000)
    except httpx.TimeoutException:
        return ProbeResult(
            ok=False,
            status_code=None,
            message=f"Timed out after {_TIMEOUT_SEC}s — check BASE_URL or network",
            latency_ms=None,
            tokens_in=None,
            tokens_out=None,
        )
    except httpx.ConnectError as exc:
        safe = mask_secrets_in_text(str(exc), secrets)
        return ProbeResult(
            ok=False,
            status_code=None,
            message=f"Connection error: {safe[:120]}",
            latency_ms=None,
            tokens_in=None,
            tokens_out=None,
        )
    except Exception as exc:
        safe = mask_secrets_in_text(str(exc), secrets)
        return ProbeResult(
            ok=False,
            status_code=None,
            message=f"Unexpected error: {safe[:120]}",
            latency_ms=None,
            tokens_in=None,
            tokens_out=None,
        )

    # Parse response — mask secrets from body before any logging
    try:
        data = r.json()
    except Exception:
        data = {}
    safe_body = mask_secrets_in_text(json.dumps(data), secrets)

    status = r.status_code

    if status == 200:
        try:
            reply = data["choices"][0]["message"]["content"] or ""
            tokens_in = data.get("usage", {}).get("prompt_tokens")
            tokens_out = data.get("usage", {}).get("completion_tokens")
            # Mask just in case a model echoes the prompt back
            safe_reply = mask_secrets_in_text(reply[:50], secrets)
            msg = f"OK ({latency_ms}ms, {tokens_in}in→{tokens_out}out) reply={safe_reply!r}"
            return ProbeResult(
                ok=True, status_code=200, message=msg,
                latency_ms=latency_ms, tokens_in=tokens_in, tokens_out=tokens_out,
            )
        except (KeyError, IndexError):
            return ProbeResult(
                ok=True, status_code=200,
                message=f"OK ({latency_ms}ms) — unexpected response shape",
                latency_ms=latency_ms, tokens_in=None, tokens_out=None,
            )

    # Error responses — extract message safely
    err_detail = ""
    if isinstance(data, dict):
        err = data.get("error", {})
        if isinstance(err, list) and err:
            err = err[0].get("error", {})
        if isinstance(err, dict):
            err_detail = err.get("message", "")[:120]
    if not err_detail:
        err_detail = safe_body[:100]
    # Final mask pass
    err_detail = mask_secrets_in_text(err_detail, secrets)

    messages = {
        400: f"HTTP 400 Bad request — {err_detail}",
        401: "HTTP 401 Invalid API key — check .env",
        403: f"HTTP 403 Forbidden — {err_detail}",
        404: f"HTTP 404 Model not found: {config.model}",
        429: f"HTTP 429 Rate limit / quota — {err_detail}",
        500: f"HTTP 500 Server error — {err_detail}",
        502: f"HTTP 502 Bad gateway — {err_detail}",
        503: f"HTTP 503 Service unavailable (high demand, retry) — {err_detail}",
    }
    msg = messages.get(status, f"HTTP {status} — {err_detail}")
    return ProbeResult(
        ok=False, status_code=status, message=msg,
        latency_ms=latency_ms, tokens_in=None, tokens_out=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(label: str, config: JudgeConfig | ParaphraseConfig, result: ProbeResult) -> None:
    masked = _mask_key(config.api_key)
    prefix = f"  {label:<35} model={config.model:<38} key={masked}  "
    if result.ok:
        click.secho(prefix + result.message, fg="green")
    else:
        click.secho(prefix + "FAILED — " + result.message, fg="red")


@click.command()
@click.option("--judge", default=None, help="Test only this judge by name (default: all)")
@click.option("--no-paraphrase", is_flag=True, help="Skip paraphrase check")
@click.option("--quiet", is_flag=True, help="No output; exit 0=all pass, 1=any fail")
def main(judge: str | None, no_paraphrase: bool, quiet: bool) -> None:
    """Verify all configured LLM judges are reachable. Keys are never printed."""
    try:
        registry = get_judge_registry(reload=True)
    except Exception as exc:
        if not quiet:
            click.secho(f"Failed to load judge registry: {exc}", fg="red")
        raise SystemExit(1)

    if not registry.judges and registry.paraphrase is None:
        if not quiet:
            click.secho("No judges configured. Set JUDGE_1_* in .env.", fg="yellow")
        raise SystemExit(1)

    if not quiet:
        click.echo(f"\nInterviewIQ — Config Check")
        click.echo(f"  Judges    : {len(registry.judges)}")
        click.echo(f"  Paraphrase: {'configured' if registry.paraphrase else 'not configured'}")
        click.echo()

    results: list[ProbeResult] = []

    # Judges
    judges_to_test = registry.judges
    if judge is not None:
        try:
            judges_to_test = [registry.get_judge(judge)]
        except KeyError as exc:
            if not quiet:
                click.secho(str(exc), fg="red")
            raise SystemExit(1)

    if not quiet:
        click.echo("Judges:")
    for j in judges_to_test:
        result = probe_judge(j)
        results.append(result)
        if not quiet:
            _print_result(f"[judge] {j.name}", j, result)

    # Paraphrase
    if not no_paraphrase and registry.paraphrase is not None:
        if not quiet:
            click.echo("\nParaphrase model:")
        result = probe_judge(registry.paraphrase)
        results.append(result)
        if not quiet:
            _print_result(f"[paraphrase] {registry.paraphrase.name}", registry.paraphrase, result)

    total = len(results)
    passed = sum(1 for r in results if r.ok)
    failed = total - passed

    if not quiet:
        click.echo()
        if failed == 0:
            click.secho(f"All {total} check(s) passed.", fg="green", bold=True)
        else:
            click.secho(f"{failed}/{total} check(s) FAILED.", fg="red", bold=True)

    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
