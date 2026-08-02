"""Fidelity: does compression change the model's answer?

The product brief's north star is "without degrading the accuracy of the AI's
output". A compression benchmark that measures only size rewards destroying
meaning -- deleting the payload entirely scores 100%. These tests close that
gap.

They run from committed cassettes: no API key, no network, no spend, so they
execute on every PR including PRs from forks. See fidelity/cassettes.py for
why that is safe, and fidelity/README.md for how to record.

Until cassettes exist these tests skip -- and that skip is not allowed to be
forgotten: tests/claims/test_manifest.py cross-checks cassette presence
against the claim status in docs/claims.yaml, so an unrecorded suite forces
the claim to stay "unverified" and keeps the promise out of the docs.
"""
from __future__ import annotations

import pytest

from tests.benchmarks.fidelity.cassettes import CassetteStore
from tests.benchmarks.fidelity.runner import MODELS, run_task, summarise
from tests.benchmarks.fidelity.tasks import build_tasks

pytestmark = pytest.mark.asyncio


def _recorded_models() -> list[str]:
    return [m for m in MODELS if CassetteStore(m).count() > 0]


def _require(model: str) -> None:
    if CassetteStore(model).count() == 0:
        pytest.skip(
            f"no cassettes for {model}. Record them locally with:\n"
            "  python scripts/record_fidelity.py --record"
        )


@pytest.mark.parametrize("model", MODELS)
async def test_compression_causes_no_answer_regressions(model, capsys):
    """The gate: compression must never turn a right answer into a wrong one."""
    _require(model)

    outcomes = [await run_task(task, model) for task in build_tasks()]
    regressions = [o for o in outcomes if o.regressed]

    with capsys.disabled():
        print(f"\n=== fidelity: {model} ===")
        print(summarise(outcomes))

    assert not regressions, "compression changed correct answers into incorrect ones: " + "; ".join(
        f"{o.task_id}: raw={o.raw_answer!r} -> compressed={o.compressed_answer!r}"
        for o in regressions
    )


@pytest.mark.parametrize("model", MODELS)
async def test_compression_actually_shrank_the_payload(model):
    """Guards against a vacuous pass.

    If compression saved nothing, "no regressions" would be trivially true and
    the suite would prove nothing. At least one task must have been compressed.
    """
    _require(model)

    outcomes = [await run_task(task, model) for task in build_tasks()]
    compressed = [o for o in outcomes if o.tokens_compressed < o.tokens_raw]
    assert compressed, "no task was compressed at all -- the fidelity result would be vacuous"


@pytest.mark.parametrize("model", MODELS)
async def test_model_answers_most_tasks_correctly_on_raw_payload(model):
    """Sanity check on the task set itself.

    If the model cannot answer these questions even from the uncompressed
    payload, the tasks are testing the model rather than taut, and a
    "no regressions" result would be meaningless.
    """
    _require(model)

    outcomes = [await run_task(task, model) for task in build_tasks()]
    correct = sum(1 for o in outcomes if o.raw_correct)
    assert correct >= 0.6 * len(outcomes), (
        f"{model} answered only {correct}/{len(outcomes)} correctly on raw payloads; "
        "the task set is measuring model capability, not compression fidelity"
    )


def test_task_set_is_wellformed():
    """Runs without cassettes: the tasks and graders must be coherent."""
    tasks = build_tasks()
    assert len(tasks) >= 10, "too few tasks to be meaningful"
    assert len({t.id for t in tasks}) == len(tasks), "duplicate task ids"
    assert {t.kind for t in tasks} >= {"json", "code", "prose"}, (
        "fidelity must cover every compressor that actually transforms content"
    )
    for task in tasks:
        assert task.context.strip(), f"{task.id} has empty context"
        assert task.question.strip(), f"{task.id} has no question"
        # A grader that accepts anything would make the task worthless.
        assert not task.grader(""), f"{task.id} grader passes an empty answer"
        assert not task.grader("banana"), f"{task.id} grader passes an irrelevant answer"
