"""Redis vector search backend for Semantic Cache."""
from __future__ import annotations
import json
import logging
import time

from taut.layers.cache.backends.base import CacheBackend, CacheEntry

logger = logging.getLogger("taut.cache.redis")

class RedisBackend(CacheBackend):
    """Redis cache backend using RediSearch for vector similarity."""
    
    def __init__(self, redis_url: str, max_entries: int = 10000, embedding_dim: int = 384):
        try:
            import redis.asyncio as redis
            self.redis_module = redis
        except ImportError:
            raise ImportError(
                "redis[search] is required for RedisBackend. Run `pip install taut[redis]`"
            ) from None
            
        self.client = self.redis_module.Redis.from_url(redis_url)
        self.index_name = "taut_cache_idx"
        self.max_entries = max_entries
        self.embedding_dim = embedding_dim
        self._index_ensured = False
        
    async def _ensure_index(self):
        """Create the RediSearch index if it doesn't exist."""
        if self._index_ensured:
            return
            
        try:
            from redis.commands.search.field import NumericField, TagField, VectorField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        except ImportError:
            raise ImportError("redis[search] is required. Run `pip install taut[redis]`") from None

        try:
            await self.client.ft(self.index_name).info()
        except Exception:
            logger.info("Creating RediSearch index for taut cache...")
            schema = (
                TagField("namespace"),
                TagField("hash"),
                NumericField("timestamp"),
                VectorField("embedding", "FLAT", {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dim,
                    "DISTANCE_METRIC": "IP"
                })
            )
            definition = IndexDefinition(prefix=["taut:cache:"], index_type=IndexType.HASH)
            try:
                await self.client.ft(self.index_name).create_index(schema, definition=definition)
            except Exception as e:
                # If another process created it concurrently, ignore
                if "Index already exists" not in str(e):
                    logger.warning(f"Could not create Redis index: {e}")
        
        self._index_ensured = True
            
    async def get_exact(self, namespace: str, key: str) -> CacheEntry | None:
        """Get an exact match by hash (key)."""
        await self._ensure_index()
        redis_key = f"taut:cache:{namespace}:{key}"
        data = await self.client.hget(redis_key, "response")
        
        if data:
            entry = CacheEntry(**json.loads(data))
            if not entry.is_expired:
                return entry
            await self.delete(namespace, key)
            
        return None

    async def get_similar(self, namespace: str, embedding: list[float], threshold: float = 0.9) -> CacheEntry | None:
        """Get a semantic match using vector search."""
        await self._ensure_index()
        import numpy as np
        from redis.commands.search.query import Query
        
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vec_bytes = vec.tobytes()
        
        # We filter by namespace exactly
        q = Query(f"(@namespace:{{{namespace}}})=>[KNN 1 @embedding $vec AS score]")\
            .sort_by("score")\
            .return_fields("score", "response", "hash")\
            .dialect(2)
            
        res = await self.client.ft(self.index_name).search(q, {"vec": vec_bytes})
        if res.docs:
            score = float(res.docs[0].score)
            if score >= threshold:
                data = json.loads(res.docs[0].response)
                entry = CacheEntry(**data)
                if not entry.is_expired:
                    return entry
                await self.delete(namespace, res.docs[0].hash)
        return None

    async def put(self, namespace: str, entry: CacheEntry) -> None:
        """Store an entry in Redis."""
        await self._ensure_index()
        import numpy as np
        
        redis_key = f"taut:cache:{namespace}:{entry.key}"
        mapping = {
            "namespace": namespace,
            "hash": entry.key,
            "timestamp": int(time.time()),
            "response": entry.model_dump_json()
        }
        
        if entry.embedding:
            vec = np.array(entry.embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            mapping["embedding"] = vec.tobytes()
        
        await self.client.hset(redis_key, mapping=mapping)
        if entry.ttl:
            await self.client.expire(redis_key, int(entry.ttl))

    async def delete(self, namespace: str, key: str) -> None:
        await self.client.delete(f"taut:cache:{namespace}:{key}")

    async def clear(self, namespace: str) -> None:
        """Clear all entries from the cache namespace using SCAN."""
        cursor = 0
        match_pattern = f"taut:cache:{namespace}:*"
        while True:
            cursor, keys = await self.client.scan(cursor=cursor, match=match_pattern, count=1000)
            if keys:
                await self.client.delete(*keys)
            if cursor == 0:
                break

    async def size(self, namespace: str) -> int:
        """Count entries in a namespace using RediSearch."""
        await self._ensure_index()
        from redis.commands.search.query import Query
        q = Query(f"@namespace:{{{namespace}}}").paging(0, 0)
        try:
            res = await self.client.ft(self.index_name).search(q)
            return res.total
        except Exception:
            return 0
