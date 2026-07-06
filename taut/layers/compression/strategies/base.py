import abc
from typing import Any, Dict
from pydantic import BaseModel

class CompressionResult(BaseModel):
    """Result of a compression operation."""
    original_text: str
    compressed_text: str
    original_size: int
    compressed_size: int
    metadata: Dict[str, Any] = {}

    @property
    def ratio(self) -> float:
        """Calculate the compression ratio (smaller is better)."""
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size

class CompressionStrategy(abc.ABC):
    """Abstract base class for compression strategies."""

    @abc.abstractmethod
    def compress(self, text: str) -> CompressionResult:
        """Compress the given text."""
        pass
