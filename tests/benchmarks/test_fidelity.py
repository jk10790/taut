"""Fidelity: does compression change the answer?

The product brief's north star is "without degrading the accuracy of the AI's
output". Nothing tested that. A compression benchmark that measures only size
rewards destroying meaning -- deleting the whole payload scores 100%.

These tests ask a real model the same question with and without taut's
compression and compare the answers against known-correct facts drawn from the
corpus. They cost money, so they are marked `costly`, excluded from PR CI, and
skipped unless an API key is present. Run them before changing a compressor:

    OPENAI_API_KEY=... pytest tests/benchmarks -m costly
"""
import json
import os
import pathlib

import pytest

pytestmark = [pytest.mark.costly, pytest.mark.asyncio]

CORPUS = pathlib.Path(__file__).parent / "corpus"
MODEL = os.getenv("TAUT_FIDELITY_MODEL", "gpt-4o-mini")

# (question, substrings that must appear in a correct answer)
QUESTIONS = [
    ("How many events have status_code 500? Answer with just the number.", None),
    ("Which region appears most often? Answer with just the region name.", None),
]

FAQ_QUESTIONS = [
    ("How many seats does the Starter plan include? Answer with just the number.", ["5", "five"]),
    ("Is SCIM provisioning available on the Growth plan? Answer yes or no.", ["no"]),
    ("How long are deleted records kept in backups? Answer with just the number of days.",
     ["35", "thirty-five"]),
]


def _requires_key():
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("TAUT_API_KEY")):
        pytest.skip("no API key set; fidelity benchmarks cost money and are opt-in")


async def _ask(context: str, question: str, compress: bool) -> str:
    from taut.core.config import CompressionConfig, TautConfig
    from taut.core.models import LLMRequest
    from taut.core.pipeline import create_pipeline

    config = TautConfig(
        cache=None,
        routing=None,
        prefix=None,
        restraint=None,
        compression=CompressionConfig(min_compress_tokens=10) if compress else None,
        default_model=MODEL,
    )
    pipeline = create_pipeline(config)
    response = await pipeline.run(
        LLMRequest(
            intent=question,
            context=context,
            model=MODEL,
            temperature=0.0,
            system_prompt="Answer using only the provided context. Be extremely brief.",
        )
    )
    return response.content.strip().lower()


async def test_json_compression_preserves_factual_answers():
    """Columnar JSON must remain readable to the model.

    This is the risk that matters for compress.json.columnar: the format is
    lossless in principle, but only useful if the model still parses it.
    """
    _requires_key()
    context = (CORPUS / "json_logs" / "api_events.json").read_text()
    events = json.loads(context)

    expected_500 = str(sum(1 for e in events if e["status_code"] == 500))
    regions = {}
    for event in events:
        regions[event["region"]] = regions.get(event["region"], 0) + 1
    expected_region = max(regions, key=regions.get)

    raw_500 = await _ask(context, QUESTIONS[0][0], compress=False)
    compressed_500 = await _ask(context, QUESTIONS[0][0], compress=True)
    raw_region = await _ask(context, QUESTIONS[1][0], compress=False)
    compressed_region = await _ask(context, QUESTIONS[1][0], compress=True)

    # The bar is parity with uncompressed, not perfection -- if the model gets
    # it wrong on raw JSON too, that is a model limitation, not a taut defect.
    if expected_500 in raw_500:
        assert expected_500 in compressed_500, (
            f"compression lost the answer: raw said {raw_500!r}, "
            f"compressed said {compressed_500!r}"
        )
    if expected_region in raw_region:
        assert expected_region in compressed_region


async def test_prose_compression_preserves_factual_answers():
    """The filler-removal rules must not change meaning."""
    _requires_key()
    context = (CORPUS / "rag" / "product_faq.md").read_text()

    for question, acceptable in FAQ_QUESTIONS:
        raw = await _ask(context, question, compress=False)
        compressed = await _ask(context, question, compress=True)
        if any(token in raw for token in acceptable):
            assert any(token in compressed for token in acceptable), (
                f"{question!r}: raw={raw!r} compressed={compressed!r}"
            )


async def test_code_compression_preserves_behavioural_questions():
    """Stripping docstrings and annotations must not change what the code does."""
    _requires_key()
    context = (CORPUS / "python" / "service.py").read_text()

    question = (
        "What exception is raised when the seat identifier is too short? "
        "Answer with just the exception class name."
    )
    raw = await _ask(context, question, compress=False)
    compressed = await _ask(context, question, compress=True)
    if "valueerror" in raw:
        assert "valueerror" in compressed, f"raw={raw!r} compressed={compressed!r}"
