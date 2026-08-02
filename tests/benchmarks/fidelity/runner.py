"""Runs fidelity tasks through the pipeline with compression on and off.

Everything except compression is held constant -- cache, routing, prefix
alignment and restraint are all disabled -- so any difference in the answer is
attributable to the compressor and nothing else.

The metric is *parity*, not absolute accuracy: a task the model gets wrong on
the raw payload is a model limitation, not a taut defect, and asserting
absolute accuracy would make the suite fail for reasons outside this repo's
control. What must hold is that compression never turns a right answer into a
wrong one.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

from taut.core.config import CompressionConfig, TautConfig
from taut.core.models import LLMRequest
from taut.core.pipeline import Pipeline
from taut.core.tokens import count_tokens

from .cassettes import CassetteMiss, CassetteProvider
from .tasks import FidelityTask

# Models the cassettes are recorded against. Changing either invalidates that
# model's cassettes.
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
MODELS = (OPENAI_MODEL, ANTHROPIC_MODEL)

# Answers are single facts; a low cap keeps recording cheap and discourages
# the model from padding.
MAX_ANSWER_TOKENS = 64


@dataclass
class TaskOutcome:
    task_id: str
    kind: str
    model: str
    raw_answer: str
    compressed_answer: str
    raw_correct: bool
    compressed_correct: bool
    tokens_raw: int
    tokens_compressed: int

    @property
    def regressed(self) -> bool:
        """Compression turned a correct answer into an incorrect one."""
        return self.raw_correct and not self.compressed_correct

    @property
    def saved_pct(self) -> float:
        if not self.tokens_raw:
            return 0.0
        return 100 * (1 - self.tokens_compressed / self.tokens_raw)


def _pipeline(model: str, compress: bool, provider) -> Pipeline:
    from taut.core.pipeline import create_pipeline

    config = TautConfig(
        cache=None,
        routing=None,
        prefix=None,
        restraint=None,
        compression=CompressionConfig(min_compress_tokens=10) if compress else None,
        default_model=model,
    )
    pipeline = create_pipeline(config)
    pipeline._provider = provider
    return pipeline


async def _ask(task: FidelityTask, model: str, compress: bool, provider) -> tuple[str, int]:
    pipeline = _pipeline(model, compress, provider)
    request = LLMRequest(
        intent=task.question,
        context=task.context,
        model=model,
        temperature=0.0,
        max_tokens=MAX_ANSWER_TOKENS,
    )
    try:
        response = await pipeline.run(request)
    except Exception as exc:
        # The pipeline treats any provider error as a failover candidate and
        # re-raises it as FallbackExhaustedError. A cassette miss is not a
        # provider outage -- it means "re-record" -- so surface it directly
        # rather than buried under an unrelated exception type.
        cause = exc
        while cause is not None:
            if isinstance(cause, CassetteMiss):
                raise cause from None
            cause = cause.__cause__
        raise
    # request.context is mutated in place by the compression layer, so this is
    # the payload actually sent.
    sent_tokens = count_tokens(request.context or "", model)
    return response.content.strip(), sent_tokens


async def run_task(
    task: FidelityTask,
    model: str,
    mode: str = "replay",
    inner=None,
    root: pathlib.Path | None = None,
) -> TaskOutcome:
    raw_provider = CassetteProvider(
        model, mode=mode, inner=inner, root=root, label=f"{task.id}[raw]"
    )
    compressed_provider = CassetteProvider(
        model, mode=mode, inner=inner, root=root, label=f"{task.id}[compressed]"
    )

    raw_answer, raw_tokens = await _ask(task, model, compress=False, provider=raw_provider)
    compressed_answer, compressed_tokens = await _ask(
        task, model, compress=True, provider=compressed_provider
    )

    return TaskOutcome(
        task_id=task.id,
        kind=task.kind,
        model=model,
        raw_answer=raw_answer,
        compressed_answer=compressed_answer,
        raw_correct=task.grader(raw_answer),
        compressed_correct=task.grader(compressed_answer),
        tokens_raw=raw_tokens,
        tokens_compressed=compressed_tokens,
    )


def summarise(outcomes: list[TaskOutcome]) -> str:
    if not outcomes:
        return "no outcomes"
    lines = [
        "",
        f"{'task':30} {'kind':6} {'raw':>5} {'comp':>5} {'saved':>7}  verdict",
        "-" * 72,
    ]
    for o in sorted(outcomes, key=lambda x: (x.model, x.task_id)):
        verdict = "REGRESSION" if o.regressed else ("ok" if o.compressed_correct else "both wrong")
        lines.append(
            f"{o.task_id:30} {o.kind:6} "
            f"{'PASS' if o.raw_correct else 'fail':>5} "
            f"{'PASS' if o.compressed_correct else 'fail':>5} "
            f"{o.saved_pct:>6.1f}%  {verdict}"
        )
    raw_correct = sum(1 for o in outcomes if o.raw_correct)
    comp_correct = sum(1 for o in outcomes if o.compressed_correct)
    regressions = sum(1 for o in outcomes if o.regressed)
    total_raw = sum(o.tokens_raw for o in outcomes)
    total_comp = sum(o.tokens_compressed for o in outcomes)
    overall = 100 * (1 - total_comp / total_raw) if total_raw else 0.0
    lines += [
        "-" * 72,
        f"correct on raw payload        : {raw_correct}/{len(outcomes)}",
        f"correct on compressed payload : {comp_correct}/{len(outcomes)}",
        f"regressions caused by taut    : {regressions}",
        f"context tokens saved overall  : {overall:.1f}%",
    ]
    return "\n".join(lines)
