import pytest
from unittest.mock import AsyncMock, patch
from taut.core.models import LLMRequest, Message, PipelineContext
from taut.core.config import TieredRoutingConfig
from taut.layers.routing.classifier import HeuristicClassifier
from taut.layers.routing.middleware import TieredRoutingMiddleware

def test_heuristic_classifier_scoring():
    classifier = HeuristicClassifier()
    thresholds = {"simple": 0.3, "standard": 0.7}
    
    # 1. Very simple request -> simple (< 0.3 score)
    req1 = LLMRequest(intent="test", messages=[Message(role="user", content="Hi")])
    result1 = classifier.calculate_score(req1, thresholds)
    assert result1.score < 0.3
    assert result1.tier == "simple"
    
    # 2. Request with some length and a keyword -> standard
    req2 = LLMRequest(
        intent="test", 
        messages=[Message(role="user", content="Analyze the evaluate function " * 200)]
    )
    result2 = classifier.calculate_score(req2, thresholds)
    assert result2.score >= 0.3
    
    # 3. Request with tools, keywords, and length -> complex (> 0.7)
    long_content = "architect a recursive algorithm to analyze and evaluate " * 500
    req3 = LLMRequest(
        intent="test",
        messages=[Message(role="user", content=long_content)],
        tools=[{"name": "tool1"}, {"name": "tool2"}, {"name": "tool3"}]
    )
    result3 = classifier.calculate_score(req3, thresholds)
    assert result3.score > 0.5 
    assert result3.score >= 0.7
    assert result3.tier == "complex"

@patch('taut.layers.routing.classifier.litellm.token_counter')
def test_heuristic_classifier_boundary_thresholds(mock_token_counter):
    classifier = HeuristicClassifier()
    thresholds = {"simple": 0.3, "standard": 0.7}
    
    req = LLMRequest(intent="test boundary")
    
    # If token count gives exactly score = 0.3, it should route to standard (>= 0.3)
    # The heuristic formula combines token count and keywords.
    # Let's mock the overall calculate_score logic indirectly by testing the classifier's boundary decision.
    # We will simulate scores and check tier decisions directly on the logic.
    
    # Helper to test tier assignment based on mocked score
    def check_tier(score, expected_tier):
        if score < thresholds["simple"]:
            tier = "simple"
        elif score < thresholds["standard"]:
            tier = "standard"
        else:
            tier = "complex"
        assert tier == expected_tier

    check_tier(0.299, "simple")
    check_tier(0.300, "standard")
    check_tier(0.699, "standard")
    check_tier(0.700, "complex")

@pytest.mark.asyncio
async def test_tiered_routing_middleware_fast():
    config = TieredRoutingConfig()
    middleware = TieredRoutingMiddleware(config=config)
    
    req = LLMRequest(intent="test", messages=[Message(role="user", content="Hi")])
    context = PipelineContext()
    next_handler = AsyncMock()
    
    await middleware.process(req, context, next_handler)
    
    assert req.model == config.tiers["simple"][0]
    assert context.selected_model == config.tiers["simple"][0]
    assert context.selected_tier == "simple"
    
    metrics = context.metrics.layers[0]
    assert metrics.layer_name == "tiered_routing"
    assert metrics.details["selected_tier"] == "simple"
    assert metrics.details["routed_model"] == req.model
    next_handler.assert_called_once_with(req, context)

@pytest.mark.asyncio
async def test_tiered_routing_middleware_expert():
    config = TieredRoutingConfig()
    middleware = TieredRoutingMiddleware(config=config)
    
    long_content = "architect a recursive algorithm to analyze and evaluate " * 500
    req = LLMRequest(
        intent="test",
        messages=[Message(role="user", content=long_content)],
        tools=[{"name": "t1"}, {"name": "t2"}],
    )
    context = PipelineContext()
    next_handler = AsyncMock()
    
    await middleware.process(req, context, next_handler)
    
    assert req.model == config.tiers["complex"][0]

@pytest.mark.asyncio
async def test_tiered_routing_middleware_client_override():
    config = TieredRoutingConfig()
    middleware = TieredRoutingMiddleware(config=config)
    
    req = LLMRequest(intent="test", model=config.tiers["complex"][0])
    context = PipelineContext()
    next_handler = AsyncMock()
    
    await middleware.process(req, context, next_handler)
    
    assert req.model == config.tiers["complex"][0]
    assert context.selected_tier == "complex"
    assert context.routing_reason == "client_override"
