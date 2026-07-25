import asyncio
from collections import OrderedDict
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from .base import CacheBackend, CacheEntry

class NamespaceCache:
    def __init__(self, embedding_dim: int):
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.index = faiss.IndexFlatIP(embedding_dim) if faiss else None
        self.id_to_key: dict[int, str] = {}
        self.next_id = 0

class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend using FAISS for semantic search and OrderedDict for LRU."""

    def __init__(self, max_size: int = 1000, embedding_dim: int = 384):
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.namespaces: dict[str, NamespaceCache] = {}
        self.lock = asyncio.Lock()
        
        if faiss is None:
            raise ImportError("faiss is required for MemoryCacheBackend. Install it with `pip install faiss-cpu`.")

    def _get_namespace(self, namespace: str) -> NamespaceCache:
        if namespace not in self.namespaces:
            self.namespaces[namespace] = NamespaceCache(self.embedding_dim)
        return self.namespaces[namespace]

    def _normalize(self, embedding: list[float]) -> np.ndarray:
        """Normalize embedding for cosine similarity with IndexFlatIP."""
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.reshape(1, -1)

    async def get_exact(self, namespace: str, key: str) -> CacheEntry | None:
        async with self.lock:
            ns = self._get_namespace(namespace)
            if key in ns.cache:
                entry = ns.cache[key]
                if entry.is_expired:
                    await self._delete_locked(namespace, key)
                    return None
                # Update LRU
                ns.cache.move_to_end(key)
                return entry
            return None

    async def get_similar(self, namespace: str, embedding: list[float], threshold: float = 0.9) -> CacheEntry | None:
        async with self.lock:
            ns = self._get_namespace(namespace)
            if len(ns.cache) == 0 or not embedding:
                return None

            # Clean up expired entries proactively when searching (basic maintenance)
            expired_keys = [k for k, v in ns.cache.items() if v.is_expired]
            for k in expired_keys:
                await self._delete_locked(namespace, k)

            if len(ns.cache) == 0:
                return None

            vec = self._normalize(embedding)
            distances, indices = ns.index.search(vec, 1)
            
            if len(distances) > 0 and len(distances[0]) > 0:
                similarity = distances[0][0]
                idx = indices[0][0]
                
                if similarity >= threshold and idx in ns.id_to_key:
                    key = ns.id_to_key[idx]
                    entry = ns.cache.get(key)
                    if entry and not entry.is_expired:
                        ns.cache.move_to_end(key)
                        return entry
            
            return None

    async def put(self, namespace: str, entry: CacheEntry) -> None:
        async with self.lock:
            ns = self._get_namespace(namespace)
            if entry.key in ns.cache:
                await self._delete_locked(namespace, entry.key)

            if len(ns.cache) >= self.max_size:
                # Evict least recently used
                oldest_key, _ = ns.cache.popitem(last=False)
                self._remove_from_index(ns, oldest_key)

            ns.cache[entry.key] = entry
            
            if entry.embedding:
                vec = self._normalize(entry.embedding)
                ns.index.add(vec)
                ns.id_to_key[ns.next_id] = entry.key
                ns.next_id += 1

    def _remove_from_index(self, ns: NamespaceCache, key: str) -> None:
        """Rebuild index without the deleted key. (FAISS IndexFlatIP doesn't support selective deletion without IDMap)"""
        ns.index = faiss.IndexFlatIP(self.embedding_dim)
        new_id_to_key = {}
        ns.next_id = 0
        
        for k, entry in ns.cache.items():
            if k != key and entry.embedding:
                vec = self._normalize(entry.embedding)
                ns.index.add(vec)
                new_id_to_key[ns.next_id] = k
                ns.next_id += 1
                
        ns.id_to_key = new_id_to_key

    async def delete(self, namespace: str, key: str) -> None:
        async with self.lock:
            await self._delete_locked(namespace, key)

    async def _delete_locked(self, namespace: str, key: str) -> None:
        ns = self._get_namespace(namespace)
        if key in ns.cache:
            del ns.cache[key]
            self._remove_from_index(ns, key)

    async def clear(self, namespace: str) -> None:
        async with self.lock:
            if namespace in self.namespaces:
                ns = self.namespaces[namespace]
                ns.cache.clear()
                ns.index = faiss.IndexFlatIP(self.embedding_dim)
                ns.id_to_key.clear()
                ns.next_id = 0

    async def size(self, namespace: str) -> int:
        async with self.lock:
            if namespace in self.namespaces:
                return len(self.namespaces[namespace].cache)
            return 0
