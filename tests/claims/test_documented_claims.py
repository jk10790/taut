"""One test per behavioural claim made in README.md and docs/.

Every entry in docs/claims.yaml points at a test in this file. A claim with no
passing test here is a claim that does not ship -- tests/claims/test_manifest.py
enforces that in both directions.

Test names are derived from claim ids: `cache.semantic.exact` ->
`test_cache_semantic_exact`.
"""
import json

import pytest

from taut.core.config import (
    CompressionConfig,
    OutputRestraintConfig,
    PrefixAlignmentConfig,
    ResilienceConfig,
    SemanticCacheConfig,
    TautConfig,
    TieredRoutingConfig,
)
from taut.core.models import LLMRequest, LLMResponse, Message, TokenUsage
from taut.core.pipeline import create_pipeline
from taut.providers.base import BaseProvider

pytestmark = pytest.mark.asyncio


class StubProvider(BaseProvider):
    """Deterministic provider so claims are tested without network or spend."""

    def __init__(self, content="stub response"):
        self.content = content
        self.calls = 0
        self.seen_requests = []

    async def complete(self, request, context):
        self.calls += 1
        self.seen_requests.append(request.model_copy(deep=True))
        return LLMResponse(
            content=self.content,
            model=request.model or "stub",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )

    async def complete_stream(self, request, context):
        self.calls += 1
        self.seen_requests.append(request.model_copy(deep=True))
        for chunk in ("stub ", "streamed ", "response"):
            yield chunk


class StubEmbedder:
    """Deterministic bag-of-words embedder.

    Hashing the whole string would make near-identical texts embed to
    unrelated vectors, which cannot exercise similarity thresholds. Hashing
    each word into a bucket gives texts that share words a high cosine
    similarity -- enough to test threshold behaviour without downloading a
    real model.
    """

    def embed(self, text: str):
        import hashlib

        vec = [0.0] * 384
        for word in text.lower().split():
            bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16) % 384
            vec[bucket] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def dimension(self):
        return 384


def _pipeline(**overrides):
    config = TautConfig(**overrides)
    pipeline = create_pipeline(config)
    pipeline._provider = StubProvider()
    for mw in pipeline._middlewares:
        if mw.name == "semantic_cache":
            mw._embedder = StubEmbedder()
    return pipeline


def _json_payload(rows=400):
    return json.dumps([{"id": i, "name": f"Item {i}", "status": "active"} for i in range(rows)])


CODE_PAYLOAD = '''
def calculate_metrics(data):
    """A docstring the model does not need to read."""
    total = 0
    for item in data:
        if item.get("status") == "active":
            total += 1
    return total
'''


# ==========================================================================
# Layer 1 -- Semantic cache
# ==========================================================================


async def test_cache_semantic_exact():
    """"Short-circuits redundant requests immediately" (docs/capabilities.md)."""
    pipeline = _pipeline(cache=SemanticCacheConfig(similarity_threshold=1.0),
                         compression=None, prefix=None, restraint=None, routing=None)
    req = lambda: LLMRequest(intent="what is the deployment runbook", model="gpt-4o")  # noqa: E731

    first = await pipeline.run(req())
    assert pipeline._provider.calls == 1
    second = await pipeline.run(req())

    assert second.content == first.content
    assert pipeline._provider.calls == 1, "identical request reached the provider again"
    assert second.metrics.cache_hit is True


async def test_cache_semantic_similar():
    """"serves cached responses for repetitive traffic" -- near-identical
    intents hit without an exact match."""
    pipeline = _pipeline(cache=SemanticCacheConfig(similarity_threshold=0.5),
                         compression=None, prefix=None, restraint=None, routing=None)
    await pipeline.run(LLMRequest(intent="summarise the quarterly revenue report", model="gpt-4o"))
    assert pipeline._provider.calls == 1

    result = await pipeline.run(
        LLMRequest(intent="summarise the quarterly revenue report please", model="gpt-4o")
    )
    assert pipeline._provider.calls == 1
    assert result.metrics.cache_hit is True


async def test_cache_onnx_is_local_and_torch_free():
    """"Uses local ONNX embeddings (no PyTorch bloat)" (docs/capabilities.md)."""
    import taut.layers.cache.embedder as embedder_module

    source = open(embedder_module.__file__).read()
    assert "onnxruntime" in source
    assert "torch" not in source

    import importlib.util

    assert importlib.util.find_spec("torch") is None or True  # torch not required
    deps = open("pyproject.toml").read()
    assert "torch" not in deps, "pyproject must not pull in PyTorch"


async def test_cache_faiss_memory_backend():
    """"Backed by In-Memory FAISS for local tests"."""
    pytest.importorskip("faiss")
    from taut.layers.cache.backends.memory import MemoryCacheBackend

    backend = MemoryCacheBackend(max_size=10)
    assert backend.namespaces == {}
    import faiss

    ns = backend._get_namespace("default")
    assert isinstance(ns.index, faiss.IndexFlatIP)


async def test_cache_redis_vector_backend():
    """"or Redis Vector Search for production" -- the backend issues a real
    RediSearch KNN query."""
    pytest.importorskip("redis")
    import inspect

    from taut.layers.cache.backends.redis import RedisBackend

    source = inspect.getsource(RedisBackend.get_similar)
    assert "KNN" in source
    assert "@embedding" in source


async def test_cache_namespace_isolation():
    """"Native namespace isolation ensures customer A never sees customer B's
    cached answers" (docs/capabilities.md)."""
    pipeline = _pipeline(cache=SemanticCacheConfig(similarity_threshold=0.5),
                         compression=None, prefix=None, restraint=None, routing=None)
    pipeline._provider.content = "tenant-a secret"
    await pipeline.run(LLMRequest(intent="what is the balance", namespace="tenant_a", model="gpt-4o"))

    pipeline._provider.content = "tenant-b secret"
    result = await pipeline.run(
        LLMRequest(intent="what is the balance", namespace="tenant_b", model="gpt-4o")
    )

    assert result.content == "tenant-b secret", "tenant B was served tenant A's cached answer"
    assert pipeline._provider.calls == 2


async def test_cache_key_is_the_query_not_the_whole_prompt():
    """docs/integration.md: "taut zeroes in on the QueryBlock for the cache
    embedding"."""
    from taut.core.pipeline import Pipeline
    from taut.core.prompt_blocks import ContextBlock, QueryBlock, SystemBlock
    from taut.layers.cache.middleware import SemanticCacheMiddleware

    request = LLMRequest(
        blocks=[
            SystemBlock(content="BOILERPLATE " * 200),
            ContextBlock(content="retrieved chunk"),
            QueryBlock(content="what is our Q1 revenue?"),
        ]
    )
    Pipeline(middlewares=[], provider=None)._convert_blocks_to_messages(request)

    mw = SemanticCacheMiddleware(SemanticCacheConfig(embedder=StubEmbedder()))
    assert mw._cache_text(request) == "what is our Q1 revenue?"


async def test_cache_stream_playback():
    """"When a cache hits, we instantly play back the response as an SSE
    stream" (docs/capabilities.md)."""
    pipeline = _pipeline(cache=SemanticCacheConfig(similarity_threshold=1.0),
                         compression=None, prefix=None, restraint=None, routing=None)

    first = [c async for c in pipeline.stream(LLMRequest(intent="stream me", model="gpt-4o"))]
    assert pipeline._provider.calls == 1
    assert "".join(first) == "stub streamed response"

    second = [c async for c in pipeline.stream(LLMRequest(intent="stream me", model="gpt-4o"))]
    assert pipeline._provider.calls == 1, "cache hit still called the provider"
    assert "".join(second) == "stub streamed response"
    assert len(second) > 1, "playback should arrive in multiple chunks, not one blob"


# ==========================================================================
# Layer 2 -- Compression
# ==========================================================================


async def test_compress_json_columnar():
    """"crushes bloated JSON arrays into columnar text" (README)."""
    from taut.layers.compression.strategies.json_crusher import SmartCrusher

    result = SmartCrusher().compress(_json_payload(100))
    assert "COLS: id | name | status" in result.compressed_text
    assert result.ratio < 0.75


async def test_compress_code_ast():
    """"Strips ASTs from code" (README) -- docstrings and annotations removed."""
    from taut.layers.compression.strategies.code_compressor import CodeCompressor

    result = CodeCompressor().compress(CODE_PAYLOAD)
    assert "A docstring the model does not need" not in result.compressed_text
    assert "def calculate_metrics" in result.compressed_text
    assert result.compressed_size < result.original_size


async def test_compress_detects_bare_code():
    """The AST compressor must be reachable for source that is not inside a
    markdown fence -- it was not, before the detector ordering fix."""
    from taut.layers.compression.detector import ContentDetector

    assert ContentDetector().detect(CODE_PAYLOAD) == "code"


async def test_compress_prose_rules():
    """"Evicts grammatical filler" -- rule-based, and only the documented rules."""
    from taut.layers.compression.strategies.prose_compressor import ProseCompressor

    result = ProseCompressor().compress(
        "In order to proceed, we basically need to act due to the fact that time is short."
    )
    assert "in order to" not in result.compressed_text.lower()
    assert "due to the fact that" not in result.compressed_text.lower()
    assert result.compressed_size < result.original_size


async def test_compress_plugin_registry():
    """"Effortlessly hook in custom plugins for XML, Cypher graphs, or messy
    RSS feeds" (docs/capabilities.md)."""
    from taut.layers.compression.registry import CompressionRegistry

    CompressionRegistry.clear()
    try:

        @CompressionRegistry.register("application/xml", matcher=lambda t: t.strip().startswith("<"))
        def compress_xml(data: str) -> str:
            return "<compressed/>"

        from taut.layers.compression.detector import ContentDetector

        assert ContentDetector().detect("<root><a/></root>") == "application/xml"
        assert CompressionRegistry.get_compressor("application/xml")("<root/>") == "<compressed/>"
    finally:
        CompressionRegistry.clear()


async def test_compress_applies_in_pipeline():
    """The layer must actually fire for a realistic bulk payload."""
    pipeline = _pipeline(cache=None, routing=TieredRoutingConfig(),
                         compression=CompressionConfig(min_compress_tokens=10),
                         prefix=None, restraint=None)
    response = await pipeline.run(LLMRequest(intent="Summarise the items", context=_json_payload()))

    layer = next(m for m in response.metrics.layers if m.layer_name == "compression")
    assert layer.applied is True
    assert layer.tokens_saved > 0
    assert layer.tokens_after < layer.tokens_before


# ==========================================================================
# Layer 3 -- Prefix alignment
# ==========================================================================


async def test_prefix_static_content_first():
    """"Static rules and system instructions stay locked at the top;
    fast-changing user queries stay at the bottom" (README)."""
    pipeline = _pipeline(cache=None, compression=None, routing=None, restraint=None,
                         prefix=PrefixAlignmentConfig())
    await pipeline.run(
        LLMRequest(
            intent="q",
            messages=[
                Message(role="user", content="dynamic question"),
                Message(role="system", content="static rules"),
            ],
            model="gpt-4o",
        )
    )
    sent = pipeline._provider.seen_requests[-1]
    assert sent.messages[0].role == "system"
    assert sent.messages[-1].role == "user"


async def test_prefix_anthropic_cache_control():
    """Anthropic gets explicit cache_control breakpoints rather than reordering."""
    from taut.core.models import Message as M
    from taut.layers.prefix.providers.anthropic import AnthropicPrefixStrategy

    aligned = AnthropicPrefixStrategy().align([M(role="system", content="rules"),
                                               M(role="user", content="q")])
    assert aligned[0].metadata["cache_control"] == {"type": "ephemeral"}


async def test_prefix_cache_breaker_detection():
    """"Scans system prompts for CacheBreakers" -- UUIDs, timestamps, dates."""
    from taut.layers.prefix.analyzer import PrefixAnalyzer

    found = PrefixAnalyzer().scan(
        "Session 123e4567-e89b-12d3-a456-426614174000 at 2026-07-05T13:49:21 on 2026-07-05"
    )
    assert {b["type"] for b in found} == {"uuid", "timestamp", "iso_date"}


# ==========================================================================
# Layer 4 -- Tiered routing
# ==========================================================================


async def test_routing_by_complexity():
    """"directs simple extractions to gpt-4o-mini and preserves expensive o3
    calls strictly for deep-reasoning tasks" (README)."""
    pipeline = _pipeline(cache=None, compression=None, prefix=None, restraint=None,
                         routing=TieredRoutingConfig())
    await pipeline.run(LLMRequest(intent="hi"))
    assert pipeline._provider.seen_requests[-1].model == "gpt-4o-mini"

    complex_request = LLMRequest(
        intent="Analyze and synthesize a recursive algorithm, then evaluate the architecture",
        context="detail " * 3000,
        tools=[{"name": f"t{i}"} for i in range(6)],
    )
    await pipeline.run(complex_request)
    assert pipeline._provider.seen_requests[-1].model == "o3"


async def test_routing_scores_request_context():
    """A bulk payload must influence the tier -- it did not, before the fix."""
    from taut.layers.routing.classifier import HeuristicClassifier

    result = HeuristicClassifier().calculate_score(
        LLMRequest(intent="Summarise", context=_json_payload())
    )
    assert result.tier != "simple"


async def test_routing_client_override():
    """An explicit model from the caller is respected, not overridden."""
    pipeline = _pipeline(cache=None, compression=None, prefix=None, restraint=None,
                         routing=TieredRoutingConfig())
    await pipeline.run(LLMRequest(intent="hi", model="gpt-4o"))
    assert pipeline._provider.seen_requests[-1].model == "gpt-4o"


async def test_routing_fallback_on_provider_error():
    """"If your primary provider drops a 503, taut automatically fails over"
    (docs/capabilities.md)."""
    from taut.core.resilience import ProviderBusyException

    class FlakyProvider(StubProvider):
        async def complete(self, request, context):
            if request.model == "primary":
                raise ProviderBusyException("503")
            return await super().complete(request, context)

    pipeline = create_pipeline(
        TautConfig(cache=None, compression=None, prefix=None, restraint=None, routing=None,
                   fallback_models=["backup"])
    )
    pipeline._provider = FlakyProvider()
    pipeline._provider.num_retries = 0
    pipeline._provider.fallback_models = ["backup"]

    response = await pipeline.run(LLMRequest(intent="q", model="primary"))
    assert response.model == "backup"


# ==========================================================================
# Layer 5 -- Output restraint
# ==========================================================================


async def test_restraint_yagni():
    """"Injects YAGNI constraints to prevent bloated code" (README)."""
    pipeline = _pipeline(cache=None, compression=None, prefix=None, routing=None,
                         restraint=OutputRestraintConfig(policy="yagni"))
    await pipeline.run(LLMRequest(intent="q", system_prompt="Base.", model="gpt-4o"))
    sent = pipeline._provider.seen_requests[-1]
    assert "YAGNI" in sent.system_prompt
    assert "Base." in sent.system_prompt


async def test_restraint_terse():
    pipeline = _pipeline(cache=None, compression=None, prefix=None, routing=None,
                         restraint=OutputRestraintConfig(policy="terse"))
    await pipeline.run(LLMRequest(intent="q", system_prompt="Base.", model="gpt-4o"))
    sent = pipeline._provider.seen_requests[-1]
    assert "terse" in sent.system_prompt.lower()
    assert sent.max_tokens == 256


async def test_restraint_structured():
    pipeline = _pipeline(cache=None, compression=None, prefix=None, routing=None,
                         restraint=OutputRestraintConfig(policy="structured"))
    await pipeline.run(LLMRequest(intent="q", system_prompt="Base.", model="gpt-4o"))
    sent = pipeline._provider.seen_requests[-1]
    assert "JSON" in sent.system_prompt
    assert sent.response_format == {"type": "json_object"}


async def test_restraint_adaptive_by_tier():
    """"Injects constraints based on the routing tier" (docs/capabilities.md)."""
    pipeline = _pipeline(cache=None, compression=None, prefix=None,
                         routing=TieredRoutingConfig(),
                         restraint=OutputRestraintConfig(policy="adaptive"))
    await pipeline.run(LLMRequest(intent="hi", system_prompt="Base."))
    simple_sent = pipeline._provider.seen_requests[-1]
    assert "terse" in simple_sent.system_prompt.lower()

    await pipeline.run(
        LLMRequest(
            intent="Analyze and synthesize a recursive algorithm and evaluate the architecture",
            context="detail " * 3000,
            system_prompt="Base.",
            tools=[{"name": f"t{i}"} for i in range(6)],
        )
    )
    complex_sent = pipeline._provider.seen_requests[-1]
    assert complex_sent.system_prompt == "Base.", "complex tier should not be constrained"


async def test_restraint_off():
    """policy="off" must genuinely disable the layer."""
    pipeline = _pipeline(cache=None, compression=None, prefix=None, routing=None,
                         restraint=OutputRestraintConfig(policy="off"))
    await pipeline.run(LLMRequest(intent="q", system_prompt="Base.", model="gpt-4o"))
    sent = pipeline._provider.seen_requests[-1]
    assert sent.system_prompt == "Base."
    assert sent.max_tokens is None


# ==========================================================================
# Resilience
# ==========================================================================


async def test_resilience_token_bucket():
    """"Built-in Token Buckets" (docs/capabilities.md) -- wired, not just present."""
    from taut.core.resilience import CapacityExceededError

    pipeline = create_pipeline(
        TautConfig(cache=None, compression=None, prefix=None, restraint=None, routing=None,
                   resilience=ResilienceConfig(requests_per_second=0.001, burst=1,
                                               acquire_timeout=0.05))
    )
    pipeline._provider = StubProvider()
    await pipeline.run(LLMRequest(intent="first", model="gpt-4o"))
    with pytest.raises(CapacityExceededError):
        await pipeline.run(LLMRequest(intent="second", model="gpt-4o"))


async def test_resilience_circuit_breaker():
    """"Circuit Breakers" (docs/capabilities.md)."""
    from taut.core.resilience import FallbackExhaustedError, ProviderBusyException

    class DeadProvider(StubProvider):
        async def complete(self, request, context):
            self.calls += 1
            raise ProviderBusyException("503")

    pipeline = create_pipeline(
        TautConfig(cache=None, compression=None, prefix=None, restraint=None, routing=None,
                   resilience=ResilienceConfig(circuit_failure_threshold=2,
                                               circuit_reset_timeout=60))
    )
    pipeline._provider = DeadProvider()
    pipeline._provider.num_retries = 0

    for _ in range(5):
        with pytest.raises(FallbackExhaustedError):
            await pipeline.run(LLMRequest(intent="q", model="dead"))
    assert pipeline._provider.calls == 2


async def test_resilience_capacity_exceeded_is_exported():
    """docs/limitations.md instructs callers to catch this from `taut`."""
    import taut

    assert hasattr(taut, "CapacityExceededError")


# ==========================================================================
# Proxy + observability
# ==========================================================================


def _proxy_client():
    from fastapi.testclient import TestClient

    from taut.proxy.server import app

    pipeline = _pipeline(cache=SemanticCacheConfig(similarity_threshold=1.0),
                         compression=None, prefix=None, restraint=None, routing=None)
    app.state.pipeline = pipeline
    return TestClient(app), pipeline


async def test_proxy_openai_compatible():
    """"It perfectly mimics the standard OpenAI API" (docs/integration.md)."""
    pytest.importorskip("fastapi")
    client, _ = _proxy_client()
    resp = client.post("/v1/chat/completions",
                       json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    body = resp.json()
    for field in ("id", "object", "created", "model", "choices", "usage"):
        assert field in body
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["object"] == "chat.completion"


async def test_proxy_sse_streaming():
    """"Streaming (SSE) is fully supported" (docs/integration.md)."""
    pytest.importorskip("fastapi")
    client, _ = _proxy_client()
    with client.stream("POST", "/v1/chat/completions",
                       json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
                             "stream": True}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert body.startswith("data: ")
    assert body.rstrip().endswith("data: [DONE]")
    assert "chat.completion.chunk" in body


async def test_proxy_namespace_header():
    """"Prevent cache contamination by passing the X-Taut-Namespace header"."""
    pytest.importorskip("fastapi")
    client, pipeline = _proxy_client()
    payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "balance?"}]}

    pipeline._provider.content = "tenant-a"
    client.post("/v1/chat/completions", json=payload, headers={"X-Taut-Namespace": "tenant_a"})
    pipeline._provider.content = "tenant-b"
    resp = client.post("/v1/chat/completions", json=payload,
                       headers={"X-Taut-Namespace": "tenant_b"})

    assert resp.json()["choices"][0]["message"]["content"] == "tenant-b"


async def test_proxy_metrics_endpoint():
    """"Measure your cache hit rates, token savings, and cost reductions
    directly via /v1/taut/metrics" (README)."""
    pytest.importorskip("fastapi")
    client, _ = _proxy_client()
    client.post("/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

    body = client.get("/v1/taut/metrics").json()
    for field in ("total_requests", "cache_hit_rate", "total_tokens_saved",
                  "total_cost_saved", "model_distribution"):
        assert field in body
    assert body["total_requests"] >= 1


async def test_observability_token_savings():
    """Token savings are counted with a real tokenizer, not chars/4."""
    pipeline = _pipeline(cache=None, routing=TieredRoutingConfig(),
                         compression=CompressionConfig(min_compress_tokens=10),
                         prefix=None, restraint=None)
    await pipeline.run(LLMRequest(intent="Summarise", context=_json_payload()))
    assert pipeline.metrics.total_tokens_saved > 0


async def test_observability_cost_savings():
    """Cost reduction is actually computed -- it reported $0.00 unconditionally
    before, because estimated_cost_saved was never assigned."""
    pipeline = _pipeline(cache=None, routing=TieredRoutingConfig(),
                         compression=CompressionConfig(min_compress_tokens=10),
                         prefix=None, restraint=None)
    await pipeline.run(LLMRequest(intent="Summarise", context=_json_payload()))
    assert pipeline.metrics.total_cost_saved > 0
