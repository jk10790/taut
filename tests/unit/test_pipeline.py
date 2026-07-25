import pytest
from taut.core.pipeline import Pipeline
from taut.core.middleware import Middleware
from taut.core.models import LLMRequest, LLMResponse, PipelineContext, TokenUsage
from taut.providers.base import BaseProvider

class DummyProvider(BaseProvider):
    async def complete(self, request: LLMRequest, context: PipelineContext) -> LLMResponse:
        return LLMResponse(
            content="Provider response",
            model=request.model or "dummy-model",
            usage=TokenUsage(input_tokens=10, output_tokens=10)
        )
        
    async def complete_stream(self, request: LLMRequest, context: PipelineContext):
        yield "Provider "
        yield "response"

class DummyMiddleware(Middleware):
    def __init__(self, name: str, append_to: list):
        self._name = name
        self.append_to = append_to
        
    @property
    def name(self) -> str:
        return self._name
        
    async def process(self, request: LLMRequest, context: PipelineContext, next_handler):
        self.append_to.append(f"{self.name}_before")
        response = await next_handler(request, context)
        self.append_to.append(f"{self.name}_after")
        return response

@pytest.mark.asyncio
async def test_pipeline_execution_order():
    """Test that pipeline executes middlewares in the correct order."""
    call_order = []
    middlewares = [
        DummyMiddleware("mw1", call_order),
        DummyMiddleware("mw2", call_order),
        DummyMiddleware("mw3", call_order),
    ]
    
    provider = DummyProvider()
    pipeline = Pipeline(middlewares=middlewares, provider=provider)
    
    request = LLMRequest(intent="test", messages=[])
    
    response = await pipeline.run(request)
    
    # Middleware list is [mw1, mw2, mw3]
    # Chain is built inside out, so mw1 is the outermost.
    # Order should be mw1 -> mw2 -> mw3 -> provider -> mw3 -> mw2 -> mw1
    assert call_order == [
        "mw1_before",
        "mw2_before",
        "mw3_before",
        "mw3_after",
        "mw2_after",
        "mw1_after",
    ]
    assert response.content == "Provider response"

@pytest.mark.asyncio
async def test_pipeline_short_circuit():
    """Test that a middleware can short-circuit the pipeline."""
    call_order = []
    
    class ShortCircuitMiddleware(Middleware):
        @property
        def name(self):
            return "short_circuit"
            
        async def process(self, request, context, next_handler):
            call_order.append("short_circuit_hit")
            return LLMResponse(
                content="Short circuit",
                model="short-circuit-model",
                usage=TokenUsage()
            )
            
    middlewares = [
        DummyMiddleware("mw1", call_order),
        ShortCircuitMiddleware(),
        DummyMiddleware("mw3", call_order),
    ]
    
    provider = DummyProvider()
    pipeline = Pipeline(middlewares=middlewares, provider=provider)
    
    request = LLMRequest(intent="test", messages=[])
    response = await pipeline.run(request)
    
    assert call_order == [
        "mw1_before",
        "short_circuit_hit",
        "mw1_after"
    ]
    assert response.content == "Short circuit"
    
@pytest.mark.asyncio
async def test_pipeline_error_recovery():
    """Test middleware error recovery handling."""
    call_order = []
    
    class ErrorMiddleware(Middleware):
        @property
        def name(self):
            return "error_mw"
            
        async def process(self, request, context, next_handler):
            call_order.append("error_hit")
            raise ValueError("Test error")
            
    class RecoveryMiddleware(Middleware):
        @property
        def name(self):
            return "recovery_mw"
            
        async def process(self, request, context, next_handler):
            call_order.append("recovery_before")
            return await next_handler(request, context)
            
        async def on_error(self, error, request, context):
            call_order.append("recovery_recovered")
            return LLMResponse(
                content="Recovered response",
                model="recovery-model",
                usage=TokenUsage()
            )
            
    middlewares = [
        RecoveryMiddleware(),
        ErrorMiddleware(),
    ]
    
    provider = DummyProvider()
    pipeline = Pipeline(middlewares=middlewares, provider=provider)
    
    request = LLMRequest(intent="test", messages=[])
    response = await pipeline.run(request)
    
    assert call_order == [
        "recovery_before",
        "error_hit",
        "recovery_recovered"
    ]
    assert response.content == "Recovered response"
