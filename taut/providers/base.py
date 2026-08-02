from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from taut.core.models import LLMRequest, LLMResponse, PipelineContext

class BaseProvider(ABC):
    """Base class for all LLM providers."""
    
    @abstractmethod
    async def complete(self, request: LLMRequest, context: PipelineContext) -> LLMResponse:
        """Execute a completion request."""
        pass
        
    @abstractmethod
    async def complete_stream(self, request: LLMRequest, context: PipelineContext) -> AsyncIterator[str]:
        """Execute a streaming completion request."""
        pass
