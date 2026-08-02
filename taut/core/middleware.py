"""Abstract base class for all taut pipeline middleware layers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, AsyncIterator
from taut.core.models import LLMRequest, LLMResponse, PipelineContext

class Middleware(ABC):
    """Base class for all taut pipeline middleware layers.
    
    Each middleware can inspect/modify the request, optionally short-circuit
    the pipeline (e.g., return a cached response), or pass through to the 
    next handler in the chain.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this middleware layer."""
        ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'name') and isinstance(getattr(cls, 'name', None), str):
            # Allow class attribute instead of property
            pass
    
    @abstractmethod
    async def process(
        self,
        request: LLMRequest,
        context: PipelineContext,
        next_handler: Callable[[LLMRequest, PipelineContext], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        """Process the request through this middleware layer.
        
        Args:
            request: The LLM request to process.
            context: Shared pipeline context for metrics and state.
            next_handler: Callable to invoke the next middleware in the chain.
            
        Returns:
            The LLM response, either from cache, transformation, or downstream.
        """
        ...
    
    async def before_request(self, request: LLMRequest, context: PipelineContext) -> LLMRequest:
        """Pre-request transformation. Used by stream() to apply layers without running the full chain.
        Default: return request unmodified. Override for layers that only transform requests."""
        return request
    
    async def on_error(
        self,
        error: Exception,
        request: LLMRequest,
        context: PipelineContext,
    ) -> LLMResponse | None:
        """Handle errors during processing.
        
        Return an LLMResponse to recover from the error, or None to re-raise.
        """
        return None

    async def stream_process(
        self,
        request: LLMRequest,
        context: PipelineContext,
        next_handler: Callable[[LLMRequest, PipelineContext], AsyncIterator[str]],
    ) -> AsyncIterator[str]:
        """Process a streaming request through this middleware layer.
        
        By default, yields chunks from the next handler.
        """
        if hasattr(self, 'before_request'):
            request = await self.before_request(request, context)
        async for chunk in next_handler(request, context):
            yield chunk
