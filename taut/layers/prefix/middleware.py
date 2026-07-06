"""Middleware for prefix alignment."""
import time
from typing import Awaitable, Callable
from taut.core.middleware import Middleware
from taut.core.models import LLMRequest, LLMResponse, PipelineContext, LayerMetrics, Message
from taut.core.config import PrefixAlignmentConfig
from .analyzer import PrefixAnalyzer
from .providers.openai import OpenAIPrefixStrategy
from .providers.anthropic import AnthropicPrefixStrategy
from .providers.google import GooglePrefixStrategy

class PrefixAlignmentMiddleware(Middleware):
    """Detects provider, runs analyzer, applies strategy, records metrics."""
    
    @property
    def name(self) -> str:
        return "prefix_alignment"
        
    def __init__(self, config: PrefixAlignmentConfig | None = None) -> None:
        self.config = config or PrefixAlignmentConfig()
        self.analyzer = PrefixAnalyzer()
        self.strategies = {
            "openai": OpenAIPrefixStrategy(),
            "anthropic": AnthropicPrefixStrategy(),
            "google": GooglePrefixStrategy(),
        }
        
    async def process(
        self,
        request: LLMRequest,
        context: PipelineContext,
        next_handler: Callable[[LLMRequest, PipelineContext], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        start_time = time.time()
        
        provider = context.provider_name or "openai"
        strategy = self.strategies.get(provider)
        
        applied = False
        breakers_found = 0
        
        if not request.messages:
            request.messages = []
            if request.system_prompt:
                request.messages.append(Message(role="system", content=request.system_prompt))
            if request.context:
                # request.context can be str | list[ContentBlock]
                # If it's a list, we might just str() it or assume str for now. 
                # Let's handle str and list of ContentBlock safely.
                ctx_content = request.context
                if isinstance(ctx_content, list):
                    ctx_str = "\n".join(getattr(b, "content", str(b)) for b in ctx_content)
                else:
                    ctx_str = str(ctx_content)
                request.messages.append(Message(role="user", content=f"Context:\n{ctx_str}"))
            if request.intent:
                request.messages.append(Message(role="user", content=request.intent))
                
        if request.messages and strategy:
            for msg in request.messages:
                if msg.role == "system" and isinstance(msg.content, str):
                    breakers = self.analyzer.scan(msg.content)
                    breakers_found += len(breakers)
            
            request.messages = strategy.align(request.messages)
            applied = True
            
        latency = (time.time() - start_time) * 1000
        
        metrics = LayerMetrics(
            layer_name=self.name,
            latency_ms=latency,
            applied=applied,
            details={"provider": provider, "breakers_found": breakers_found}
        )
        context.metrics.layers.append(metrics)
        
        return await next_handler(request, context)
