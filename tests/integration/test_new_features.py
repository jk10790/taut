import pytest
import asyncio
from typing import AsyncGenerator

from taut.core.config import TautConfig, SemanticCacheConfig, TieredRoutingConfig
from taut.core.pipeline import create_pipeline
from taut.core.models import LLMRequest, LLMResponse, PipelineContext, TokenUsage
from taut.providers.base import BaseProvider
from taut.core.resilience import ProviderBusyException, FallbackExhaustedError

class FailingMockProvider(BaseProvider):
    def __init__(self, fallback_models=None):
        self.fallback_models = fallback_models
        self.received_models = []
        
    async def complete(self, request: LLMRequest, context: PipelineContext) -> LLMResponse:
        self.received_models.append(request.model)
        if request.model == "failing-model":
            raise ProviderBusyException("Simulated 503 error from failing-model")
        
        return LLMResponse(
            content=f"Success from {request.model}",
            model=request.model,
            usage=TokenUsage(input_tokens=10, output_tokens=10)
        )
        
    async def complete_stream(self, request: LLMRequest, context: PipelineContext) -> AsyncGenerator[str, None]:
        self.received_models.append(request.model)
        if request.model == "failing-model":
            raise ProviderBusyException("Simulated 503 error from failing-model stream")
            
        yield f"Success "
        yield f"from "
        yield request.model

@pytest.fixture
def test_config():
    return TautConfig(
        cache=SemanticCacheConfig(
            enabled=True,
            similarity_threshold=1.0,
            ttl_seconds=60
        ),
        fallback_models=["fallback-1", "fallback-2"]
    )

@pytest.mark.asyncio
async def test_fallback_routing_run(test_config):
    provider = FailingMockProvider(fallback_models=test_config.fallback_models)
    pipeline = create_pipeline(test_config)
    pipeline._provider = provider
    pipeline._provider.num_retries = 0 # Disable retries for faster test
    
    request = LLMRequest(intent="Test fallback", model="failing-model")
    response = await pipeline.run(request)
    
    assert response.content == "Success from fallback-1"
    assert response.model == "fallback-1"
    assert "failing-model" in provider.received_models
    assert "fallback-1" in provider.received_models

@pytest.mark.asyncio
async def test_fallback_routing_exhausted(test_config):
    # If all models are failing
    class AllFailingProvider(BaseProvider):
        async def complete(self, request, context):
            raise ProviderBusyException("All fail")
        async def complete_stream(self, request, context):
            raise ProviderBusyException("All fail")
            yield "never"
            
    pipeline = create_pipeline(test_config)
    pipeline._provider = AllFailingProvider()
    pipeline._provider.num_retries = 0
    
    request = LLMRequest(intent="Test exhausted", model="failing-model")
    with pytest.raises(FallbackExhaustedError):
        await pipeline.run(request)

@pytest.mark.asyncio
async def test_fallback_routing_stream(test_config):
    provider = FailingMockProvider(fallback_models=test_config.fallback_models)
    pipeline = create_pipeline(test_config)
    pipeline._provider = provider
    pipeline._provider.num_retries = 0
    
    request = LLMRequest(intent="Test stream fallback", model="failing-model")
    
    chunks = []
    async for chunk in pipeline.stream(request):
        chunks.append(chunk)
        
    assert "".join(chunks) == "Success from fallback-1"
    assert "failing-model" in provider.received_models
    assert "fallback-1" in provider.received_models

@pytest.mark.asyncio
async def test_streaming_cache_playback(test_config):
    class PlaybackProvider(BaseProvider):
        def __init__(self):
            self.call_count = 0
            
        async def complete(self, request, context):
            return LLMResponse(content="Should not be called", model="test")
            
        async def complete_stream(self, request, context):
            self.call_count += 1
            yield "Chunk1 "
            yield "Chunk2 "
            yield "Chunk3"

    provider = PlaybackProvider()
    pipeline = create_pipeline(test_config)
    pipeline._provider = provider
    
    class DummyEmbedder:
        async def embed(self, text):
            return [0.0] * 384
    cache_mw = next(mw for mw in pipeline._middlewares if mw.name == "semantic_cache")
    cache_mw._embedder = DummyEmbedder()
    
    # Run 1: Cache Miss
    req1 = LLMRequest(intent="Stream cache test")
    chunks1 = []
    async for chunk in pipeline.stream(req1):
        chunks1.append(chunk)
        
    assert "".join(chunks1) == "Chunk1 Chunk2 Chunk3"
    assert provider.call_count == 1
    
    # Small delay for cache write
    await asyncio.sleep(0.1)
    
    # Run 2: Cache Hit
    req2 = LLMRequest(intent="Stream cache test")
    chunks2 = []
    async for chunk in pipeline.stream(req2):
        chunks2.append(chunk)
        
    assert "".join(chunks2) == "Chunk1 Chunk2 Chunk3"
    assert provider.call_count == 1 # Provider should not be called again
