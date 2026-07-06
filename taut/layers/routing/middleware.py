"""Middleware for tiered routing."""
import time
from typing import Awaitable, Callable
from taut.core.middleware import Middleware
from taut.core.models import LLMRequest, LLMResponse, PipelineContext, LayerMetrics
from taut.core.config import TieredRoutingConfig
from .classifier import HeuristicClassifier

class TieredRoutingMiddleware(Middleware):
    """Evaluates complexity and overrides request.model."""
    
    @property
    def name(self) -> str:
        return "tiered_routing"
        
    def __init__(self, config: TieredRoutingConfig) -> None:
        self.config = config
        self.classifier = HeuristicClassifier(keywords=config.complexity_keywords)
        
    async def process(
        self,
        request: LLMRequest,
        context: PipelineContext,
        next_handler: Callable[[LLMRequest, PipelineContext], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        start_time = time.time()
        
        original_model = request.model
        
        if request.model:
            # Client provided a model, skip classification
            selected_model = request.model
            # Attempt to back-map the model to a tier
            selected_tier = None
            for t_name, models in self.config.tiers.items():
                if selected_model in models:
                    selected_tier = t_name
                    break
            
            if not selected_tier:
                selected_tier = self.config.default_tier
                
            context.selected_tier = selected_tier
            context.selected_model = selected_model
            context.routing_reason = "client_override"
            
            latency = (time.time() - start_time) * 1000
            metrics = LayerMetrics(
                layer_name=self.name,
                latency_ms=latency,
                applied=False,
                details={
                    "selected_tier": selected_tier,
                    "routed_model": selected_model,
                    "reason": "client_override"
                }
            )
            context.metrics.layers.append(metrics)
            return await next_handler(request, context)
            
        result = self.classifier.calculate_score(request, thresholds=self.config.complexity_thresholds)
        
        tier_models = self.config.tiers.get(result.tier)
        if not tier_models:
            tier_models = self.config.tiers.get(self.config.default_tier, [])
            
        selected_model = tier_models[0] if tier_models else "gpt-3.5-turbo"
        
        request.model = selected_model
        context.selected_model = selected_model
        context.selected_tier = result.tier
        context.routing_reason = result.reason
        
        latency = (time.time() - start_time) * 1000
        
        metrics = LayerMetrics(
            layer_name=self.name,
            latency_ms=latency,
            applied=(original_model != selected_model),
            details={
                "complexity_score": result.score,
                "selected_tier": result.tier,
                "original_model": original_model,
                "routed_model": selected_model,
                "reason": result.reason
            }
        )
        context.metrics.layers.append(metrics)
        
        return await next_handler(request, context)
