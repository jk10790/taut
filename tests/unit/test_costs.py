"""Cost and token accounting.

`estimated_cost_saved` was read in three places and assigned in none, so
/v1/taut/metrics reported $0.00 unconditionally while the README advertised
cost reporting. And token savings were estimated as len(text)//4, which
disagrees with the provider's own accounting on exactly the content types taut
compresses hardest.
"""
import json

import pytest

from taut.core.config import CompressionConfig, TautConfig, TieredRoutingConfig
from taut.core.models import LLMRequest, LLMResponse, TokenUsage
from taut.core.pipeline import create_pipeline
from taut.core.tokens import count_tokens
from taut.observability.pricing import cost_per_token, estimate_cost, is_priced
from taut.providers.base import BaseProvider


class _StubProvider(BaseProvider):
    def __init__(self, input_tokens=1200, output_tokens=300):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    async def complete(self, request, context):
        return LLMResponse(
            content="ok",
            model=request.model,
            usage=TokenUsage(input_tokens=self.input_tokens, output_tokens=self.output_tokens),
        )

    async def complete_stream(self, request, context):
        yield "ok"


def _payload(n=400):
    return json.dumps([{"id": i, "name": f"Item {i}", "status": "active"} for i in range(n)])


# --------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------


def test_count_tokens_uses_a_real_tokenizer_not_chars_over_four():
    text = _payload(50)
    counted = count_tokens(text, "gpt-4o")
    naive = len(text) // 4
    assert counted > 0
    # JSON tokenizes denser than 4 chars/token; if these matched exactly we
    # would still be on the heuristic.
    assert counted != naive


def test_count_tokens_handles_empty_and_unknown_model():
    assert count_tokens("", "gpt-4o") == 0
    assert count_tokens("hello world", "some-model-that-does-not-exist") > 0


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


def test_known_models_are_priced():
    assert is_priced("gpt-4o")
    assert is_priced("gpt-4o-mini")


def test_locally_served_models_cost_nothing():
    assert cost_per_token("ollama/llama3") == (0.0, 0.0)


def test_unknown_model_reports_unknown_rather_than_guessing():
    assert cost_per_token("totally-made-up-model-xyz") == (0.0, 0.0)
    assert not is_priced("totally-made-up-model-xyz")


def test_estimate_cost_scales_with_tokens():
    cheap = estimate_cost("gpt-4o", 1000, 100)
    dear = estimate_cost("gpt-4o", 10000, 1000)
    assert 0 < cheap < dear


# --------------------------------------------------------------------------
# End-to-end metric population
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_populates_cost_metrics():
    pipeline = create_pipeline(
        TautConfig(cache=None, routing=TieredRoutingConfig(),
                   compression=CompressionConfig(min_compress_tokens=10))
    )
    pipeline._provider = _StubProvider()

    response = await pipeline.run(LLMRequest(intent="Summarize the items", context=_payload()))
    metrics = response.metrics

    assert metrics.estimated_cost_with_taut > 0, "cost was never computed"
    assert metrics.estimated_cost_without_taut >= metrics.estimated_cost_with_taut
    assert metrics.estimated_cost_saved > 0
    assert metrics.total_tokens_saved > 0


@pytest.mark.asyncio
async def test_metrics_endpoint_payload_reports_nonzero_savings():
    pipeline = create_pipeline(
        TautConfig(cache=None, routing=TieredRoutingConfig(),
                   compression=CompressionConfig(min_compress_tokens=10))
    )
    pipeline._provider = _StubProvider()
    await pipeline.run(LLMRequest(intent="Summarize the items", context=_payload()))

    payload = pipeline.metrics.to_dict()
    assert payload["total_tokens_saved"] > 0
    assert payload["total_cost_saved"] > 0, "/v1/taut/metrics reported $0.00"


@pytest.mark.asyncio
async def test_no_savings_claimed_when_taut_changed_nothing():
    """Honesty check: a request taut cannot improve must report zero saved,
    not a flattering number."""
    pipeline = create_pipeline(
        TautConfig(cache=None, routing=None, compression=None, prefix=None, restraint=None)
    )
    pipeline._provider = _StubProvider()

    response = await pipeline.run(LLMRequest(intent="hello", model="gpt-4o"))
    assert response.metrics.total_tokens_saved == 0
    assert response.metrics.estimated_cost_saved == 0.0


@pytest.mark.asyncio
async def test_unpriced_model_reports_zero_saved_not_a_fabricated_number():
    pipeline = create_pipeline(
        TautConfig(cache=None, routing=None,
                   compression=CompressionConfig(min_compress_tokens=10),
                   prefix=None, restraint=None)
    )
    pipeline._provider = _StubProvider()

    response = await pipeline.run(
        LLMRequest(intent="Summarize", context=_payload(), model="unknown-local-model")
    )
    assert response.metrics.total_tokens_saved > 0  # tokens are still real
    assert response.metrics.estimated_cost_saved == 0.0  # but cost is unknown
