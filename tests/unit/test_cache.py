import pytest
import numpy as np
import time
from unittest.mock import AsyncMock, MagicMock
from taut.core.models import LLMRequest, LLMResponse, PipelineContext, TokenUsage
from taut.core.config import SemanticCacheConfig
from taut.layers.cache.middleware import SemanticCacheMiddleware
from taut.layers.cache.backends.memory import MemoryCacheBackend
from taut.layers.cache.backends.base import CacheEntry

pytestmark = pytest.mark.asyncio

class MockEmbedder:
    def __init__(self):
        self.dim = 384
        
    async def embed(self, text: str):
        h = hash(text) % 1000
        vec = np.zeros(self.dim)
        vec[0] = h / 1000.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def dimension(self):
        return self.dim


@pytest.fixture
def mock_embedder():
    return MockEmbedder()


@pytest.fixture
def memory_backend():
    try:
        import faiss  # noqa: F401
    except ImportError:
        pytest.skip("faiss not installed")
    return MemoryCacheBackend(max_size=10, embedding_dim=384)


@pytest.fixture
def cache_config():
    return SemanticCacheConfig(
        enabled=True,
        similarity_threshold=0.9,
        ttl_seconds=60
    )


@pytest.fixture
def cache_middleware(cache_config, memory_backend, mock_embedder):
    mw = SemanticCacheMiddleware(config=cache_config)
    mw._backend = memory_backend
    mw._embedder = mock_embedder
    return mw


async def test_exact_match_cache_hit(cache_middleware):
    request = LLMRequest(intent="test exact match", namespace="default")
    context = PipelineContext()
    response_val = LLMResponse(content="cached response", model="test", usage=TokenUsage(input_tokens=0, output_tokens=0))
    
    prompt = cache_middleware._cache_text(request) if hasattr(cache_middleware, "_cache_text") else str(request)
    exact_key = cache_middleware._compute_key(request) if hasattr(cache_middleware, "_compute_key") else "key1"
    
    entry = CacheEntry(
        key=exact_key,
        value=response_val,
        embedding=await cache_middleware._embedder.embed(prompt)
    )
    await cache_middleware._backend.put("default", entry)

    next_handler = AsyncMock()
    result = await cache_middleware.process(request, context, next_handler)

    assert result.content == "cached response"
    assert context.extra.get("cache_status") == "exact_hit"
    next_handler.assert_not_called()


async def test_semantic_match_cache_hit(cache_middleware):
    request1 = LLMRequest(intent="first prompt", namespace="default")
    request2 = LLMRequest(intent="second prompt, similar", namespace="default")
    context = PipelineContext()

    vec = [0.0] * 384
    vec[0] = 1.0
    
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=vec)
    cache_middleware._embedder = embedder

    response_val = LLMResponse(content="semantic cached response", model="test", usage=TokenUsage(input_tokens=0, output_tokens=0))
    
    entry = CacheEntry(
        key="some_key",
        value=response_val,
        embedding=vec
    )
    await cache_middleware._backend.put("default", entry)

    next_handler = AsyncMock()
    result = await cache_middleware.process(request2, context, next_handler)

    assert result.content == "semantic cached response"
    assert context.extra.get("cache_status") == "semantic_hit"
    next_handler.assert_not_called()


async def test_cache_miss_stores_value(cache_middleware):
    request = LLMRequest(intent="miss prompt", namespace="default")
    context = PipelineContext()
    
    response_val = LLMResponse(content="new response", model="test", usage=TokenUsage(input_tokens=0, output_tokens=0))
    
    next_handler = AsyncMock(return_value=response_val)

    # Assert cache is empty
    # For a memory backend we assume we can peek at size or use get_exact to verify
    
    result = await cache_middleware.process(request, context, next_handler)

    assert result.content == "new response"
    assert context.extra.get("cache_status") == "miss"
    next_handler.assert_called_once_with(request, context)
    
    # Assert cache now has the value
    exact_key = cache_middleware._compute_key(request) if hasattr(cache_middleware, "_compute_key") else str(request)
    cached_entry = await cache_middleware._backend.get_exact("default", exact_key)
    
    # Some implementations fire-and-forget; if the store is not awaited fully or if it's awaited inline:
    # Assuming inline for the memory test
    assert cached_entry is not None
    assert cached_entry.value.content == "new response"


async def test_memory_backend_eviction(memory_backend, mock_embedder):
    memory_backend._max_size = 2 # Assuming it uses _max_size or max_size
    if hasattr(memory_backend, 'max_size'):
        memory_backend.max_size = 2
        
    ns = "default"
    entry1 = CacheEntry(key="1", value="one", embedding=await mock_embedder.embed("one"))
    entry2 = CacheEntry(key="2", value="two", embedding=await mock_embedder.embed("two"))
    entry3 = CacheEntry(key="3", value="three", embedding=await mock_embedder.embed("three"))
    
    await memory_backend.put(ns, entry1)
    await memory_backend.put(ns, entry2)
    
    # Access entry1 to make it recently used
    await memory_backend.get_exact(ns, "1")
    
    # Adding entry3 should evict entry2 (the least recently used)
    await memory_backend.put(ns, entry3)
    
    assert await memory_backend.get_exact(ns, "2") is None
    assert await memory_backend.get_exact(ns, "1") is not None
    assert await memory_backend.get_exact(ns, "3") is not None


async def test_memory_backend_expiration(memory_backend):
    ns = "default"
    entry = CacheEntry(
        key="expiring",
        value="value",
        embedding=[0.0]*384,
        ttl=0.1
    )
    await memory_backend.put(ns, entry)
    
    assert await memory_backend.get_exact(ns, "expiring") is not None
    time.sleep(0.2)
    assert await memory_backend.get_exact(ns, "expiring") is None
