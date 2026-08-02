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


# Measured regressions, recorded rather than hidden. Every entry here is a
# case where compression *did* turn a right answer into a wrong one, and the
# reason is documented in docs/limitations.md ("Columnar JSON costs small
# models their column alignment").
#
# This is an inventory, not an excuse: an unlisted regression fails the suite,
# and so does a listed task that has stopped regressing. Either direction means
# the documented limitation no longer matches reality.
#
# gpt-4o-mini loses track of which column it is reading roughly 70 rows below
# the `COLS:` header and answers with the neighbouring column's value -- e.g.
# `1074` (the user_id of event 74) when asked for an event_id. It is not a
# margin-of-error effect: events.fastest_event has a 35.8% gap between the
# right answer and the runner-up and still regresses. The same tasks are clean
# on claude-haiku-4-5, and retrieval of a single named row is clean on both.
KNOWN_REGRESSIONS: dict[str, set[str]] = {
    "gpt-4o-mini": {
        "inventory.priciest_sku",
        "events.slowest_event",
        "events.fastest_event",
    },
    "claude-haiku-4-5-20251001": set(),
}


@pytest.mark.parametrize("model", MODELS)
async def test_compression_causes_no_answer_regressions(model, capsys):
    """The gate: compression must not turn a right answer into a wrong one.

    Exempting the measured, documented cases in KNOWN_REGRESSIONS -- which the
    test pins in both directions so they cannot quietly grow or go stale.
    """
    _require(model)

    outcomes = [await run_task(task, model) for task in build_tasks()]
    regressed = {o.task_id for o in outcomes if o.regressed}
    known = KNOWN_REGRESSIONS.get(model, set())

    with capsys.disabled():
        print(f"\n=== fidelity: {model} ===")
        print(summarise(outcomes))

    detail = {
        o.task_id: f"raw={o.raw_answer!r} -> compressed={o.compressed_answer!r}"
        for o in outcomes
        if o.regressed
    }
    new = sorted(regressed - known)
    assert not new, (
        "compression changed correct answers into incorrect ones on tasks that "
        "are not documented limitations: "
        + "; ".join(f"{task_id}: {detail[task_id]}" for task_id in new)
    )

    healed = sorted(known - regressed)
    assert not healed, (
        f"{healed} no longer regress under compression on {model}. Good news, but "
        "KNOWN_REGRESSIONS and docs/limitations.md now overstate the problem -- "
        "remove these entries and update the limitation."
    )


@pytest.mark.parametrize("model", MODELS)
async def test_compression_never_regresses_retrieval(model):
    """The unconditional invariant, exempted by nothing.

    Every documented regression is a *scan-and-compare* failure: find the
    extremum of a column across a hundred rows. Reading one named row back is
    a different operation, and compression must never damage it on any model.
    A failure here would mean the columnar transform loses data, rather than
    merely making it harder to scan.
    """
    _require(model)

    outcomes = [await run_task(task, model) for task in build_tasks()]
    lookups = [o for o in outcomes if o.probe == "lookup"]
    assert lookups, "no retrieval-control tasks -- the invariant would be vacuous"

    regressions = [o for o in lookups if o.regressed]
    assert not regressions, (
        "compression broke single-row retrieval, which points at data loss "
        "rather than a scanning limitation: "
        + "; ".join(
            f"{o.task_id}: raw={o.raw_answer!r} -> compressed={o.compressed_answer!r}"
            for o in regressions
        )
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
    # Both arms of the aggregation experiment plus its control must survive any
    # future edit to the task set, or the limitation in docs/limitations.md
    # stops being reproducible from this repo.
    from tests.benchmarks.fidelity.tasks import PROBES

    assert {t.probe for t in tasks} >= set(PROBES), (
        f"the aggregation experiment needs all of {PROBES}; "
        "see docs/limitations.md for what it establishes"
    )
    for task in tasks:
        assert task.context.strip(), f"{task.id} has empty context"
        assert task.question.strip(), f"{task.id} has no question"
        # A grader that accepts anything would make the task worthless.
        assert not task.grader(""), f"{task.id} grader passes an empty answer"
        assert not task.grader("banana"), f"{task.id} grader passes an irrelevant answer"
