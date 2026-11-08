#!/usr/bin/env python
"""qualify_judge.py — qualify a judge with 3 real rubric calls on fixed answers.

Standard (a judge PASSES only if all 3 calls succeed):
  1. reply is non-empty,
  2. reply is valid JSON,
  3. finish_reason == "stop" (model finished naturally, not truncated).

Each call bypasses the disk cache so it is a real call to the live judge.

Prints per call: finish_reason, completion_tokens, max_tokens sent, latency.
Prints an overall PASS/FAIL and exits 0 only on PASS.

Usage:
    python qualify_judge.py --judge judge_nemotron_ultra
    python qualify_judge.py --judge mock          # uses MockLLMService
    python qualify_judge.py --all                  # qualify every configured judge

API keys are never printed. Provider error text is masked before display.
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv  # type: ignore
load_dotenv(override=False)

import click

sys.path.insert(0, str(Path(__file__).parent))

from app.services.qualify import qualify_call
from app.services.judge_config import mask_secrets_in_text, get_judge_registry


# 3 fixed answers of varying quality on one fixed question.
_QUALIFY_QUESTION = "Explain the difference between a stack and a queue."
_QUALIFY_ANSWERS = [
    ("strong", (
        "A stack is Last In First Out (LIFO): you push and pop from the top. "
        "A queue is First In First Out (FIFO): you enqueue at the back and "
        "dequeue from the front. Stacks are used for call stacks and undo; "
        "queues for scheduling and BFS. Both have O(1) for their core ops."
    )),
    ("medium", (
        "A stack is LIFO and a queue is FIFO. You add and remove from a stack "
        "at one end, and from a queue at opposite ends. They are used in "
        "various algorithms."
    )),
    ("weak", (
        "A stack and a queue are both ways to store data. A stack is faster "
        "because it uses less memory. I used them once in a project."
    )),
]


def _get_service(judge_name: str):
    """Return an LLMService for the named judge, or the mock."""
    if judge_name == "mock":
        from app.services.llm import MockLLMService
        return MockLLMService()
    registry = get_judge_registry()
    cfg = registry.get_judge(judge_name)
    from app.services.llm import OpenAICompatibleService, _cache_dir
    return OpenAICompatibleService(cfg, cache_dir=_cache_dir())


def _qualify_one(judge_name: str, secrets: list[str]) -> bool:
    """Run 3 qualification calls on the judge. Returns True if all pass."""
    svc = _get_service(judge_name)
    click.echo(f"\n  Judge: {judge_name}  ({svc.model_name})")

    all_ok = True
    for tier, answer in _QUALIFY_ANSWERS:
        qr = qualify_call(svc, _QUALIFY_QUESTION, answer, max_tokens=512)
        # Mask any secret that a provider might echo into error text
        safe_error = mask_secrets_in_text(qr.error, secrets) if qr.error else ""
        status = "OK " if qr.ok else "FAIL"
        click.echo(
            f"    [{tier:<7}] {status}  finish_reason={qr.finish_reason!r:<10} "
            f"completion_tokens={qr.completion_tokens}  "
            f"max_tokens_sent={qr.max_tokens_sent}  latency={qr.latency_ms}ms"
            + (f"  — {safe_error}" if safe_error else "")
        )
        if not qr.ok:
            all_ok = False

    return all_ok


@click.command()
@click.option("--judge", default=None, help="Judge name to qualify (or 'mock')")
@click.option("--all", "qualify_all", is_flag=True, help="Qualify every configured judge")
def main(judge: str | None, qualify_all: bool) -> None:
    """Qualify one or more judges with 3 real rubric calls each."""
    try:
        registry = get_judge_registry()
        secrets = registry.all_secrets()
    except Exception:
        secrets = []

    if qualify_all:
        names = [j.name for j in registry.judges] if registry.judges else []
        if not names:
            click.secho("No judges configured.", fg="yellow")
            raise SystemExit(1)
    elif judge is not None:
        names = [judge]
    else:
        click.secho("Specify --judge <name> or --all.", fg="red")
        raise SystemExit(1)

    click.echo("\nQualify Judge — 3 real rubric calls per judge")
    click.echo("Pass requires: non-empty reply, valid JSON, finish_reason == 'stop'")

    overall_ok = True
    for name in names:
        try:
            ok = _qualify_one(name, secrets)
        except Exception as exc:
            safe = mask_secrets_in_text(str(exc), secrets)
            click.secho(f"\n  Judge {name} errored: {safe}", fg="red")
            ok = False
        overall_ok = overall_ok and ok

    click.echo()
    if overall_ok:
        click.secho("PASS — all judges qualified.", fg="green", bold=True)
        raise SystemExit(0)
    else:
        click.secho("FAIL — one or more judges did not qualify.", fg="red", bold=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
