#!/usr/bin/env python3
"""Record (or verify) fidelity cassettes against real providers.

This is the ONLY code in the repo that spends money or needs an API key, and it
is deliberately not a CI workflow: keys stay on the maintainer's machine and
never enter GitHub. CI replays the committed cassettes instead.

    # Estimate cost without calling anything
    python scripts/record_fidelity.py --dry-run

    # Record missing cassettes (needs a key)
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/record_fidelity.py --record

    # One provider only
    python scripts/record_fidelity.py --record --model gpt-4o-mini

    # Re-record everything, e.g. after a model version bump
    python scripts/record_fidelity.py --refresh

    # Check committed cassettes still match what the provider says today
    python scripts/record_fidelity.py --verify

Keys are read from the environment, or from a local .env file which is
gitignored. Nothing read from either is ever written to a cassette --
see tests/benchmarks/fidelity/cassettes.py for the scrubbing rules.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taut.core.tokens import count_tokens  # noqa: E402
from taut.observability.pricing import estimate_cost, is_priced  # noqa: E402
from tests.benchmarks.fidelity.cassettes import CassetteStore  # noqa: E402
from tests.benchmarks.fidelity.runner import (  # noqa: E402
    ANTHROPIC_MODEL,
    MAX_ANSWER_TOKENS,
    MODELS,
    OPENAI_MODEL,
    run_task,
    summarise,
)
from tests.benchmarks.fidelity.tasks import build_tasks  # noqa: E402

KEY_FOR_MODEL = {
    OPENAI_MODEL: "OPENAI_API_KEY",
    ANTHROPIC_MODEL: "ANTHROPIC_API_KEY",
}


def load_dotenv() -> None:
    """Load a local .env if present. It is gitignored; never commit one."""
    env_file = REPO / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    print("loaded .env")


def estimate(models: list[str]) -> None:
    tasks = build_tasks()
    print(f"{len(tasks)} tasks x 2 (compressed/raw) x {len(models)} model(s)\n")
    grand_total = 0.0
    for model in models:
        # Raw pass sends the full context; the compressed pass sends less, so
        # this is an upper bound.
        context_tokens = sum(count_tokens(t.context, model) for t in tasks) * 2
        output_tokens = len(tasks) * 2 * MAX_ANSWER_TOKENS
        cost = estimate_cost(model, context_tokens, output_tokens)
        grand_total += cost
        known = "" if is_priced(model) else "   (no pricing data -- cost unknown)"
        print(f"  {model:34} <= {context_tokens:>8,} in / {output_tokens:>5,} out   ${cost:.4f}{known}")
    print(f"\n  {'TOTAL (upper bound)':34}                              ${grand_total:.4f}")


def make_provider(model: str):
    from taut.providers.litellm_provider import LiteLLMProvider

    env_var = KEY_FOR_MODEL.get(model)
    api_key = os.environ.get(env_var) if env_var else None
    if not api_key:
        raise SystemExit(
            f"{env_var} is not set, so {model} cannot be recorded.\n"
            f"Set it in your shell or in a local .env (gitignored), or skip this "
            f"model with --model."
        )
    return LiteLLMProvider(default_model=model, api_key=api_key)


async def record(models: list[str], mode: str) -> int:
    tasks = build_tasks()
    failures = 0
    for model in models:
        print(f"\n=== {mode}: {model} ===")
        provider = make_provider(model)
        outcomes = []
        for task in tasks:
            try:
                outcomes.append(await run_task(task, model, mode=mode, inner=provider))
                print(f"  ok   {task.id}")
            except Exception as exc:
                failures += 1
                print(f"  FAIL {task.id}: {type(exc).__name__}: {exc}")
        if outcomes:
            print(summarise(outcomes))
        print(f"cassettes on disk for {model}: {CassetteStore(model).count()}")
    return failures


async def verify(models: list[str]) -> int:
    """Re-ask the provider and compare with what is committed.

    Answers drift when a provider silently updates a model behind a stable
    alias. This detects that without changing the committed cassettes.
    """
    import tempfile

    tasks = build_tasks()
    drifted = 0
    for model in models:
        store = CassetteStore(model)
        if store.count() == 0:
            print(f"{model}: nothing recorded, skipping")
            continue
        print(f"\n=== verify: {model} ===")
        provider = make_provider(model)
        scratch = pathlib.Path(tempfile.mkdtemp())
        for task in tasks:
            committed = await run_task(task, model, mode="replay")
            fresh = await run_task(task, model, mode="record", inner=provider, root=scratch)
            for label, old, new in (
                ("raw", committed.raw_answer, fresh.raw_answer),
                ("compressed", committed.compressed_answer, fresh.compressed_answer),
            ):
                if old != new:
                    drifted += 1
                    print(f"  DRIFT {task.id}[{label}]: committed={old!r} now={new!r}")
        if not drifted:
            print("  no drift: committed cassettes match the provider today")
    return drifted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="estimate cost, call nothing")
    action.add_argument("--record", action="store_true", help="record cassettes that are missing")
    action.add_argument("--refresh", action="store_true", help="re-record everything")
    action.add_argument("--verify", action="store_true", help="compare committed cassettes with live answers")
    parser.add_argument("--model", action="append", choices=list(MODELS), help="limit to one model (repeatable)")
    args = parser.parse_args()

    models = args.model or list(MODELS)

    if args.dry_run:
        estimate(models)
        return 0

    load_dotenv()

    if args.verify:
        drifted = asyncio.run(verify(models))
        if drifted:
            print(f"\n{drifted} answer(s) drifted. Re-record with --refresh if the new answers are correct.")
            return 1
        return 0

    estimate(models)
    print()
    failures = asyncio.run(record(models, "refresh" if args.refresh else "record"))
    if failures:
        print(f"\n{failures} task(s) failed to record.")
        return 1
    print("\nDone. Review the cassette diff, then commit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
