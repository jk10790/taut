import abc
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class CacheEntry(BaseModel):
    """Represents an entry in the cache."""
    key: str
    value: Any
    embedding: Optional[List[float]] = None
    created_at: float = Field(default_factory=time.time)
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)

class CacheBackend(abc.ABC):
    """Abstract base class for cache backends."""

    @abc.abstractmethod
    async def get_exact(self, namespace: str, key: str) -> Optional[CacheEntry]:
        """Retrieve an entry by exact key match."""
        pass

    @abc.abstractmethod
    async def get_similar(self, namespace: str, embedding: List[float], threshold: float = 0.9) -> Optional[CacheEntry]:
        """Retrieve an entry by semantic similarity."""
        pass

    @abc.abstractmethod
    async def put(self, namespace: str, entry: CacheEntry) -> None:
        """Store a cache entry."""
        pass

    @abc.abstractmethod
    async def delete(self, namespace: str, key: str) -> None:
        """Delete an entry from the cache."""
        pass

    @abc.abstractmethod
    async def clear(self, namespace: str) -> None:
        """Clear all entries from the cache namespace."""
        pass

    @abc.abstractmethod
    async def size(self, namespace: str) -> int:
        """Return the number of items in the cache namespace."""
        pass
