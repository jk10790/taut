"""Pipeline orchestrator for taut."""
import asyncio
import logging
import time
from collections.abc import Callable, AsyncIterator

from taut.core.config import (
    TautConfig
)
from taut.core.middleware import Middleware
from taut.core.models import LLMRequest, LLMResponse, PipelineContext
from taut.observability.metrics import MetricsCollector
from taut.observability.events import EventBus
from taut.providers.base import BaseProvider
from taut.core.resilience import with_retries, ProviderBusyException, FallbackExhaustedError

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
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(request))
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.run(request)).result()

    async def stream(self, request: LLMRequest):
        """Execute the full optimization pipeline and stream the response."""
        self._convert_blocks_to_messages(request)
        context = PipelineContext(
            original_request=request.model_copy(deep=True),
        )
        context.metrics.request_id = context.request_id
        
        async def terminal_handler(req: LLMRequest, ctx: PipelineContext) -> AsyncIterator[str]:
            models_to_try = []
            if ctx.selected_model:
                models_to_try.append(ctx.selected_model)
            elif req.model:
                models_to_try.append(req.model)
            else:
                default_mod = getattr(self._provider, "default_model", None)
                if default_mod:
                    models_to_try.append(default_mod)
            
            fallbacks = getattr(self._provider, "fallback_models", None) or []
            if hasattr(ctx, "routing_fallbacks") and ctx.routing_fallbacks:
                fallbacks = ctx.routing_fallbacks
                
            models_to_try.extend([m for m in fallbacks if m not in models_to_try])
            if not models_to_try:
                models_to_try = ["unknown"]

            last_error = None
            for model in models_to_try:
                req.model = model
                ctx.selected_model = model
                
                try:
                    # Very simple retry logic for stream (generator can't easily be decorated with @with_retries without fully buffering)
                    # We'll rely on the provider's native streaming retries, or just try fallbacks if it fails initially.
                    generator_started = False
                    async for chunk in self._provider.complete_stream(req, ctx):
                        generator_started = True
                        yield chunk
                    return # Successfully finished stream
                except Exception as e:
                    logger.warning(f"Stream Model {model} failed: {e}. Trying next fallback...")
                    last_error = e
                    if generator_started:
                        # Cannot fallback if we already yielded chunks!
                        raise
            
            raise FallbackExhaustedError(f"All stream models failed. Last error: {last_error}") from last_error
                
        handler = terminal_handler
        for mw in reversed(self._middlewares):
            handler = self._wrap_stream_middleware(mw, handler)

        async for chunk in handler(request, context):
            yield chunk
            
        # Emit stream complete event
        await self._events.emit("stream_complete", request=request, context=context)

    def _wrap_stream_middleware(self, mw: Middleware, next_handler: Callable):
        async def handler(request: LLMRequest, context: PipelineContext):
            async for chunk in mw.stream_process(request, context, next_handler):
                yield chunk
        return handler
    
    async def _call_provider(
        self, request: LLMRequest, context: PipelineContext
    ) -> LLMResponse:
        """Terminal handler — calls the actual LLM provider with fallback & retries."""
        models_to_try = []
        if context.selected_model:
            models_to_try.append(context.selected_model)
        elif request.model:
            models_to_try.append(request.model)
        else:
            default_mod = getattr(self._provider, "default_model", None)
            if default_mod:
                models_to_try.append(default_mod)
        
        # Add fallbacks from config if present
        fallbacks = getattr(self._provider, "fallback_models", None) or []
        if hasattr(context, "routing_fallbacks") and context.routing_fallbacks:
            fallbacks = context.routing_fallbacks
            
        models_to_try.extend([m for m in fallbacks if m not in models_to_try])
        if not models_to_try:
            models_to_try = ["unknown"]

        last_error = None
        for model in models_to_try:
            request.model = model
            context.selected_model = model
            
            # Decorate the inner call with retries for transient errors
            @with_retries(max_retries=getattr(self._provider, 'num_retries', 2), exceptions=(ProviderBusyException, TimeoutError))
            async def _do_call():
                try:
                    return await self._provider.complete(request, context)
                except Exception as e:
                    error_str = str(e).lower()
                    if "timeout" in error_str or "429" in error_str or "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
                        raise ProviderBusyException(str(e)) from e
                    raise

            try:
                response = await _do_call()
                context.metrics.model_used = response.model
                return response
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}. Trying next fallback...")
                last_error = e

        raise FallbackExhaustedError(f"All models failed. Last error: {last_error}") from last_error
    
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
        fallback_models=config.fallback_models,
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
