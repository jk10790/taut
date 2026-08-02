import asyncio
import hashlib
import logging
from collections.abc import Callable

from taut.core.middleware import Middleware
from taut.core.models import LLMRequest, LLMResponse, PipelineContext, LayerMetrics
from taut.core.config import SemanticCacheConfig
from .backends.base import CacheBackend, CacheEntry
from .guards import blocks_reuse, discriminative_tokens

logger = logging.getLogger("taut.cache")

class SemanticCacheMiddleware(Middleware):
    """
    Middleware for semantic caching of LLM requests and responses.
    Two-tier lookup: Exact hash match, then semantic similarity.
    """
    name = "semantic_cache"

    def __init__(self, config: SemanticCacheConfig):
        self._config = config
        self._backend = self._create_backend(config)
        if config.embedder is not None:
            self._embedder = config.embedder
        else:
            from .embedder import ONNXEmbedder
            self._embedder = ONNXEmbedder(config.embedding_model)

    def _create_backend(self, config: SemanticCacheConfig) -> CacheBackend:
        if config.backend == "redis":
            from .backends.redis import RedisBackend
            if not config.redis_url:
                raise ValueError("redis_url must be provided when backend is 'redis'")
            return RedisBackend(redis_url=config.redis_url, max_entries=config.max_entries)
        else:
            from .backends.memory import MemoryCacheBackend
            return MemoryCacheBackend(max_size=config.max_entries)

    def _compute_key(self, request: LLMRequest) -> str:
        prompt = self._cache_text(request)
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        
    def _cache_text(self, request: LLMRequest) -> str:
        """Return the text the cache key and embedding are built from.

        Only the dynamic part of the prompt is used. Embedding the whole
        request would let a large static system prompt dominate the vector, so
        two unrelated queries sharing boilerplate could land above the
        similarity threshold and serve each other's answers.
        """
        if getattr(request, 'intent', None):
            return request.intent

        # No explicit intent: fall back to the last user turn, which is the
        # dynamic part of a chat-shaped request.
        for msg in reversed(request.messages or []):
            if msg.role == "user":
                if isinstance(msg.content, str):
                    return msg.content
                text_parts = [
                    part.get("text", "")
                    for part in msg.content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                if text_parts:
                    return "\n".join(text_parts)

        # Nothing dynamic to key on. Use the full request rather than risk
        # collapsing distinct requests onto one key.
        return str(request)

    def _guard_rejects(self, query_text: str, entry) -> bool:
        """Veto a semantic hit whose numeric or date tokens differ.

        The cached query text is carried in the entry's metadata. An entry
        written before this guard existed has no such text; those are let
        through rather than invalidated, since the alternative is silently
        dropping every pre-upgrade cache entry.
        """
        if not self._config.discriminative_guard:
            return False
        cached_text = (entry.metadata or {}).get("cache_text")
        if not cached_text:
            return False
        if blocks_reuse(query_text, cached_text):
            logger.debug(
                "semantic hit vetoed: numeric/date tokens differ "
                f"({discriminative_tokens(query_text)} != {discriminative_tokens(cached_text)})"
            )
            return True
        return False

    async def _embed_text(self, text: str) -> list[float]:
        if asyncio.iscoroutinefunction(self._embedder.embed):
            return await self._embedder.embed(text)
        else:
            return await asyncio.to_thread(self._embedder.embed, text)

    def _record_metrics(self, context: PipelineContext, hit: bool):
        context.metrics.cache_hit = hit
        context.metrics.layers.append(
            LayerMetrics(layer_name=self.name, applied=hit)
        )

    async def process(self, request: LLMRequest, context: PipelineContext, next_handler: Callable) -> LLMResponse:
        cache_key = self._compute_key(request)
        namespace = request.namespace or "default"

        # Tier 1: Exact Match
        try:
            exact_hit = await self._backend.get_exact(namespace, cache_key)
            if exact_hit:
                context.extra["cache_status"] = "exact_hit"
                self._record_metrics(context, hit=True)
                return exact_hit.value
        except Exception as e:
            logger.warning(f"Cache get_exact failed: {e}")

        # Tier 2: Semantic Similarity
        embedding = None
        if self._config.similarity_threshold < 1.0:
            try:
                query_text = self._cache_text(request)
                embedding = await self._embed_text(query_text)
                similar_hit = await self._backend.get_similar(
                    namespace, embedding, self._config.similarity_threshold
                )
                if similar_hit and self._guard_rejects(query_text, similar_hit):
                    # Falls through to a miss. Recorded separately because the
                    # status is overwritten with "miss" below, and a vetoed
                    # lookup is worth distinguishing from a plain miss when
                    # tuning the threshold.
                    context.extra["cache_vetoed"] = True
                    similar_hit = None
                if similar_hit:
                    context.extra["cache_status"] = "semantic_hit"
                    self._record_metrics(context, hit=True)
                    return similar_hit.value
            except Exception as e:
                logger.warning(f"Cache get_similar failed: {e}")

        # Cache miss — continue pipeline
        context.extra["cache_status"] = "miss"
        response = await next_handler(request, context)

        # Store response
        try:
            if embedding is None:
                embedding = await self._embed_text(self._cache_text(request))
            entry = CacheEntry(
                key=cache_key,
                embedding=embedding,
                value=response,
                ttl=float(self._config.ttl_seconds) if self._config.ttl_seconds else None,
                # The guard compares an incoming query against the query this
                # entry was written for, so that text has to travel with it.
                metadata={"cache_text": self._cache_text(request)},
            )
            await self._backend.put(namespace, entry)
        except Exception as e:
            logger.warning(f"Cache put failed: {e}")

        self._record_metrics(context, hit=False)
        return response

    async def before_request(self, request: LLMRequest, context: PipelineContext) -> LLMRequest:
        return request

    async def stream_process(self, request: LLMRequest, context: PipelineContext, next_handler: Callable):
        cache_key = self._compute_key(request)
        namespace = request.namespace or "default"

        # Tier 1: Exact Match
        try:
            exact_hit = await self._backend.get_exact(namespace, cache_key)
            if exact_hit:
                context.extra["cache_status"] = "exact_hit"
                self._record_metrics(context, hit=True)
                
                content = exact_hit.value.content
                chunk_size = max(1, len(content) // 20)
                for i in range(0, len(content), chunk_size):
                    yield content[i:i+chunk_size]
                    await asyncio.sleep(0.01)
                return
        except Exception as e:
            logger.warning(f"Cache get_exact failed during stream: {e}")

        # Tier 2: Semantic Similarity
        embedding = None
        if self._config.similarity_threshold < 1.0:
            try:
                query_text = self._cache_text(request)
                embedding = await self._embed_text(query_text)
                similar_hit = await self._backend.get_similar(
                    namespace, embedding, self._config.similarity_threshold
                )
                if similar_hit and self._guard_rejects(query_text, similar_hit):
                    # Falls through to a miss. Recorded separately because the
                    # status is overwritten with "miss" below, and a vetoed
                    # lookup is worth distinguishing from a plain miss when
                    # tuning the threshold.
                    context.extra["cache_vetoed"] = True
                    similar_hit = None
                if similar_hit:
                    context.extra["cache_status"] = "semantic_hit"
                    self._record_metrics(context, hit=True)
                    
                    content = similar_hit.value.content
                    chunk_size = max(1, len(content) // 20)
                    for i in range(0, len(content), chunk_size):
                        yield content[i:i+chunk_size]
                        await asyncio.sleep(0.01)
                    return
            except Exception as e:
                logger.warning(f"Cache get_similar failed during stream: {e}")

        # Cache miss — continue pipeline
        context.extra["cache_status"] = "miss"
        
        chunks = []
        async for chunk in next_handler(request, context):
            chunks.append(chunk)
            yield chunk
            
        full_text = "".join(chunks)
        
        from taut.core.models import TokenUsage
        response = LLMResponse(
            content=full_text,
            model=context.selected_model or request.model or "unknown",
            usage=TokenUsage()
        )
        
        try:
            if embedding is None:
                embedding = await self._embed_text(self._cache_text(request))
            entry = CacheEntry(
                key=cache_key,
                embedding=embedding,
                value=response,
                ttl=float(self._config.ttl_seconds) if self._config.ttl_seconds else None,
                # The guard compares an incoming query against the query this
                # entry was written for, so that text has to travel with it.
                metadata={"cache_text": self._cache_text(request)},
            )
            await self._backend.put(namespace, entry)
        except Exception as e:
            logger.warning(f"Cache put failed during stream: {e}")

        self._record_metrics(context, hit=False)
