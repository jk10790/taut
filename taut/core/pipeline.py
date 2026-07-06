"""Pipeline orchestrator for taut."""
import asyncio
import logging
import time
from typing import Callable

from taut.core.config import (
    SemanticCacheConfig, CompressionConfig, 
    PrefixAlignmentConfig, TieredRoutingConfig, OutputRestraintConfig,
    TautConfig
)
from taut.core.middleware import Middleware
from taut.core.models import LLMRequest, LLMResponse, PipelineContext
from taut.observability.metrics import MetricsCollector
from taut.observability.events import EventBus
from taut.providers.base import BaseProvider

logger = logging.getLogger("taut.pipeline")

class Pipeline:
    """Chain of Responsibility pipeline for LLM request optimization."""
    
    def __init__(
        self,
        middlewares: list[Middleware],
        provider: BaseProvider,
        metrics_collector: MetricsCollector | None = None,
        event_bus: EventBus | None = None,
    ):
        self._middlewares = middlewares
        self._provider = provider
        self._metrics = metrics_collector or MetricsCollector()
        self._events = event_bus or EventBus()
    
    def _convert_blocks_to_messages(self, request: LLMRequest):
        """Convert PromptBlocks to Messages at start of processing."""
        if hasattr(request, "blocks") and request.blocks:
            from taut.core.models import Message
            from taut.core.prompt_blocks import SystemBlock, ToolsBlock
            if request.messages is None:
                request.messages = []
            
            for block in request.blocks:
                role = "system" if isinstance(block, SystemBlock) else "user"
                msg = Message(
                    role=role,
                    content=block.content,
                    metadata={"stability": block.stability, "cache_eligible": block.cache_eligible}
                )
                request.messages.append(msg)
                
                if isinstance(block, ToolsBlock) and block.tools:
                    if request.tools is None:
                        request.tools = []
                    request.tools.extend(block.tools)
            
            request.blocks = None

    async def run(self, request: LLMRequest) -> LLMResponse:
        """Execute the full optimization pipeline."""
        self._convert_blocks_to_messages(request)
        context = PipelineContext(
            original_request=request.model_copy(deep=True),
        )
        context.metrics.request_id = context.request_id
        
        # Build chain from inside out
        handler = self._call_provider
        for mw in reversed(self._middlewares):
            handler = self._wrap_middleware(mw, handler)
        
        try:
            response = await handler(request, context)
        except Exception as e:
            await self._events.emit("error", error=str(e), request_id=context.request_id)
            raise
        
        # Finalize metrics
        context.metrics.total_latency_ms = (time.time() - context.start_time) * 1000
        context.metrics.total_tokens_saved = sum(
            lm.tokens_saved for lm in context.metrics.layers
        )
        response.metrics = context.metrics
        
        # Record and emit
        self._metrics.record(context.metrics)
        await self._events.emit(
            "request_complete",
            request_id=context.request_id,
            tokens_saved=context.metrics.total_tokens_saved,
            cost_saved=context.metrics.estimated_cost_saved,
            model=response.model,
            cache_hit=context.metrics.cache_hit,
        )
        
        return response
    
    def run_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous wrapper for run()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(request))
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.run(request)).result()

    async def stream(self, request: LLMRequest):
        """Execute the full optimization pipeline and stream the response."""
        from typing import AsyncIterator
        
        self._convert_blocks_to_messages(request)
        context = PipelineContext(
            original_request=request.model_copy(deep=True),
        )
        context.metrics.request_id = context.request_id
        
        # Run pre-request middleware hooks (if available)
        for mw in self._middlewares:
            if hasattr(mw, "before_request"):
                request = await mw.before_request(request, context)
        
        model = context.selected_model or getattr(self._provider, "default_model", None)
        if model:
            request.model = model
            
        async for chunk in self._provider.complete_stream(request, context):
            yield chunk
            
        # Emit stream complete event
        await self._events.emit("stream_complete", request=request, context=context)
    
    async def _call_provider(
        self, request: LLMRequest, context: PipelineContext
    ) -> LLMResponse:
        """Terminal handler — calls the actual LLM provider."""
        if context.selected_model:
            request.model = context.selected_model
        response = await self._provider.complete(request, context)
        context.metrics.model_used = response.model
        return response
    
    def _wrap_middleware(self, mw: Middleware, next_handler: Callable):
        """Wrap a middleware with error handling and timing."""
        async def handler(request: LLMRequest, context: PipelineContext) -> LLMResponse:
            
            # Wrapper around next_handler to track downstream time
            downstream_time = 0.0
            async def wrapped_next(req, ctx):
                nonlocal downstream_time
                ds_start = time.time()
                try:
                    return await next_handler(req, ctx)
                finally:
                    downstream_time += time.time() - ds_start

            start = time.time()
            try:
                response = await mw.process(request, context, wrapped_next)
            except Exception as e:
                logger.warning(f"Middleware '{mw.name}' error: {e}")
                recovery = await mw.on_error(e, request, context)
                if recovery is not None:
                    return recovery
                raise
            
            # Elapsed time for THIS middleware is total time minus downstream time
            elapsed = (time.time() - start - downstream_time) * 1000
            for lm in context.metrics.layers:
                if lm.layer_name == mw.name and lm.latency_ms == 0.0:
                    lm.latency_ms = elapsed
                    break
            return response
        return handler
    
    @property
    def metrics(self) -> MetricsCollector:
        """Access the metrics collector."""
        return self._metrics
    
    def on(self, event: str, callback: Callable) -> None:
        """Register an event listener."""
        self._events.on(event, callback)


def create_pipeline(config: TautConfig | None = None, **kwargs) -> Pipeline:
    """Factory function to create a taut Pipeline with sensible defaults."""
    if config is None:
        config = TautConfig(**kwargs)
        
    from taut.providers.litellm_provider import LiteLLMProvider
    
    llm_provider = LiteLLMProvider(
        default_model=config.default_model or _default_model_for(config.provider),
        api_key=config.api_key,
        base_url=config.base_url,
    )
    
    middlewares: list[Middleware] = []
    
    if config.cache is not None:
        from taut.layers.cache.middleware import SemanticCacheMiddleware
        middlewares.append(SemanticCacheMiddleware(config=config.cache))
    
    if config.routing is not None:
        from taut.layers.routing.middleware import TieredRoutingMiddleware
        middlewares.append(TieredRoutingMiddleware(config=config.routing))

    if config.compression is not None:
        from taut.layers.compression.middleware import CompressionMiddleware
        middlewares.append(CompressionMiddleware(config=config.compression))
    
    if config.prefix is not None:
        from taut.layers.prefix.middleware import PrefixAlignmentMiddleware
        middlewares.append(PrefixAlignmentMiddleware(config=config.prefix))
    
    if config.restraint is not None:
        from taut.layers.restraint.middleware import OutputRestraintMiddleware
        middlewares.append(OutputRestraintMiddleware(config=config.restraint))
    
    return Pipeline(
        middlewares=middlewares,
        provider=llm_provider,
    )


def _default_model_for(provider: str) -> str:
    """Return a sensible default model for the given provider."""
    defaults = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-2.5-flash",
    }
    return defaults.get(provider, "gpt-4o")
