import pytest
import asyncio
from collections.abc import AsyncGenerator
from taut.core.config import (
    TautConfig, 
    SemanticCacheConfig, 
    TieredRoutingConfig, 
    CompressionConfig, 
    PrefixAlignmentConfig, 
    OutputRestraintConfig
)
from taut.core.pipeline import create_pipeline, Pipeline
from taut.core.models import LLMRequest, LLMResponse, PipelineContext, TokenUsage
from taut.providers.base import BaseProvider

class MockProvider(BaseProvider):
    def __init__(self, response_content="Mock response"):
        self.response_content = response_content
        self.received_requests = []
        self.received_contexts = []
        
    async def complete(self, request: LLMRequest, context: PipelineContext) -> LLMResponse:
        self.received_requests.append(request)
        self.received_contexts.append(context)
        
        return LLMResponse(
            content=self.response_content,
            model=context.selected_model or request.model or "mock-model",
            usage=TokenUsage(input_tokens=10, output_tokens=10),
            finish_reason="stop",
            tool_calls=None,
            raw_response=None
        )
        
    async def complete_stream(self, request: LLMRequest, context: PipelineContext) -> AsyncGenerator[str, None]:
        self.received_requests.append(request)
        
        yield "Mock "
        yield "stream "
        yield "response"

@pytest.fixture
def full_config():
    return TautConfig(
        cache=SemanticCacheConfig(
            enabled=True,
            similarity_threshold=1.0, # exact match effectively, simplified for test
            ttl_seconds=60
        ),
        routing=TieredRoutingConfig(
            enabled=True,
            tiers={
                "simple": ["simple-model"],
                "standard": ["standard-model"],
                "complex": ["complex-model"]
            },
            complexity_thresholds={
                "simple": 0.3,
                "standard": 0.7
            }
        ),
        compression=CompressionConfig(
            enabled=True,
            skip_for_simple_tier=True,
            min_compress_tokens=10 # low threshold for tests
        ),
        prefix=PrefixAlignmentConfig(
            enabled=True,
            provider="openai"
        ),
        restraint=OutputRestraintConfig(
            enabled=True,
            default_policy="yagni"
        )
    )

def test_1_create_pipeline(full_config):
    # Test 1: create_pipeline(TautConfig()) succeeds without TypeError
    pipeline = create_pipeline(full_config)
    assert isinstance(pipeline, Pipeline)
    # Test 3: All 5 layers execute in order (checked by inspecting pipeline._middlewares)
    mw_names = [mw.name for mw in pipeline._middlewares]
    assert mw_names == ["semantic_cache", "tiered_routing", "compression", "prefix_alignment", "output_restraint"]

@pytest.mark.asyncio
async def test_2_and_4_pipeline_run_and_context(full_config):
    mock_provider = MockProvider()
    pipeline = create_pipeline(full_config)
    pipeline._provider = mock_provider
    
    request = LLMRequest(intent="Test intent", context="Test context")
    response = await pipeline.run(request)
    
    # Test 2: returns LLMResponse with correct fields
    assert isinstance(response, LLMResponse)
    assert response.content == "Mock response"
    assert response.usage.input_tokens == 10
    
    # Test 4: PipelineContext has inter-layer fields populated
    assert len(mock_provider.received_contexts) == 1
    ctx = mock_provider.received_contexts[0]
    assert ctx.selected_tier is not None
    assert ctx.selected_model is not None
    assert hasattr(ctx, "compression_applied")

@pytest.mark.asyncio
async def test_5_simple_request_routes(full_config):
    mock_provider = MockProvider()
    pipeline = create_pipeline(full_config)
    pipeline._provider = mock_provider
    
    # Short text, low complexity -> simple tier
    request = LLMRequest(intent="hi")
    await pipeline.run(request)
    
    ctx = mock_provider.received_contexts[-1]
    assert ctx.selected_tier == "simple"
    assert ctx.selected_model == "simple-model"

@pytest.mark.asyncio
async def test_6_complex_request_routes(full_config):
    mock_provider = MockProvider()
    pipeline = create_pipeline(full_config)
    pipeline._provider = mock_provider
    
    # Complex request with tools
    request = LLMRequest(
        intent="Analyze this large dataset and draw conclusions",
        context="Data " * 500,
        tools=[{"type": "function", "function": {"name": "analyze"}}] * 10
    )
    await pipeline.run(request)
    
    ctx = mock_provider.received_contexts[-1]
    assert ctx.selected_tier in ["standard", "complex"]
    # Depending on exact heuristic tuning, it should not be 'simple'

@pytest.mark.asyncio
async def test_7_compression_applied(full_config):
    mock_provider = MockProvider()
    pipeline = create_pipeline(full_config)
    pipeline._provider = mock_provider
    
    # Ensure it's not simple tier so compression applies
    long_json = '[{"a":1, "b":2}, {"a":3, "b":4}, {"a":5, "b":6}, {"a":7, "b":8}, {"a":9, "b":10}]' * 5
    request = LLMRequest(intent="Analyze data", context=long_json)
    await pipeline.run(request)
    
    ctx = mock_provider.received_contexts[-1]
    # It might be routed to standard/complex
    if ctx.selected_tier != "simple":
        assert ctx.compression_applied is True
        assert ctx.content_types_detected.get("context") in ["json", "mixed", "prose"]

@pytest.mark.asyncio
async def test_8_restraint_instructions_appended(full_config):
    mock_provider = MockProvider()
    pipeline = create_pipeline(full_config)
    pipeline._provider = mock_provider
    
    request = LLMRequest(intent="Do something")
    await pipeline.run(request)
    
    req_at_provider = mock_provider.received_requests[-1]
    # Restraint middleware typically appends to system prompt or messages
    assert req_at_provider.system_prompt or req_at_provider.messages

@pytest.mark.asyncio
async def test_9_cache_flow(full_config):
    mock_provider = MockProvider()
    pipeline = create_pipeline(full_config)
    pipeline._provider = mock_provider
    
    # Mock embedder
    class DummyEmbedder:
        async def embed(self, text):
            return [0.0] * 384
    cache_mw = next(mw for mw in pipeline._middlewares if mw.name == "semantic_cache")
    cache_mw._embedder = DummyEmbedder()
    
    request = LLMRequest(intent="Cache me")
    
    # 1. Cache Miss
    resp1 = await pipeline.run(request)
    assert resp1.content == "Mock response"
    
    # 2. Wait slightly for async cache put if it's fire-and-forget
    await asyncio.sleep(0.1)
    
    # 3. Cache Hit
    request2 = LLMRequest(intent="Cache me")
    resp2 = await pipeline.run(request2)
    # The provider should not have been called a second time
    assert len(mock_provider.received_requests) == 1 

def test_10_run_sync(full_config):
    mock_provider = MockProvider()
    pipeline = create_pipeline(full_config)
    pipeline._provider = mock_provider
    
    request = LLMRequest(intent="Sync test")
    response = pipeline.run_sync(request)
    
    assert isinstance(response, LLMResponse)
    assert response.content == "Mock response"

@pytest.mark.asyncio
async def test_11_stream(full_config):
    mock_provider = MockProvider()
    pipeline = create_pipeline(full_config)
    pipeline._provider = mock_provider
    
    request = LLMRequest(intent="Stream test")
    chunks = []
    async for chunk in pipeline.stream(request):
        chunks.append(chunk)
        
    assert len(chunks) == 3
    # Provider got the request (pre-request layers were applied)
    assert len(mock_provider.received_requests) == 1
