import pytest
from unittest.mock import AsyncMock
from taut.core.models import LLMRequest, Message, PipelineContext
from taut.core.config import OutputRestraintConfig
from taut.layers.restraint.middleware import OutputRestraintMiddleware
from taut.layers.restraint.policies.yagni import YAGNIPolicy
from taut.layers.restraint.policies.base import RestraintPolicy

class MockPolicy(RestraintPolicy):
    def __init__(self, config=None):
        pass
        
    @property
    def name(self) -> str:
        return "mock"
        
    def get_instructions(self, context=None) -> str:
        return "BE MOCK"

    def get_max_tokens(self, context=None) -> int | None:
        return 42

    def get_response_format(self, provider: str | None = None, context=None) -> dict | None:
        return {"type": "json_object"}

@pytest.mark.asyncio
async def test_output_restraint_injects_system_prompt():
    config = OutputRestraintConfig()
    middleware = OutputRestraintMiddleware(config=config)
    middleware.default_policy = MockPolicy(config)
    
    req = LLMRequest(intent="test", system_prompt="Original system prompt.")
    context = PipelineContext()
    next_handler = AsyncMock()
    
    await middleware.process(req, context, next_handler)
    
    assert "Original system prompt." in req.system_prompt
    assert "BE MOCK" in req.system_prompt
    assert req.max_tokens == 42
    assert req.response_format == {"type": "json_object"}
    
    next_handler.assert_called_once_with(req, context)

@pytest.mark.asyncio
async def test_output_restraint_modifies_existing_system_message():
    config = OutputRestraintConfig()
    middleware = OutputRestraintMiddleware(config=config)
    middleware.default_policy = MockPolicy(config)
    
    req = LLMRequest(
        intent="test",
        messages=[Message(role="system", content="System msg"), Message(role="user", content="Hi")]
    )
    context = PipelineContext()
    next_handler = AsyncMock()
    
    await middleware.process(req, context, next_handler)
    
    assert req.messages[0].role == "system"
    assert "System msg" in req.messages[0].content
    assert "BE MOCK" in req.messages[0].content
    assert req.max_tokens == 42
    
@pytest.mark.asyncio
async def test_output_restraint_inserts_new_system_message():
    config = OutputRestraintConfig()
    middleware = OutputRestraintMiddleware(config=config)
    middleware.default_policy = MockPolicy(config)
    
    req = LLMRequest(
        intent="test",
        messages=[Message(role="user", content="Hi")]
    )
    context = PipelineContext()
    next_handler = AsyncMock()
    
    await middleware.process(req, context, next_handler)
    
    assert req.messages[0].role == "system"
    assert req.messages[0].content == "BE MOCK"
    assert req.messages[1].role == "user"

@pytest.mark.asyncio
async def test_output_restraint_max_tokens_override():
    config = OutputRestraintConfig()
    middleware = OutputRestraintMiddleware(config=config)
    middleware.default_policy = MockPolicy(config)
    
    # Request already has a lower max_tokens
    req = LLMRequest(intent="test", max_tokens=10)
    context = PipelineContext()
    next_handler = AsyncMock()
    
    await middleware.process(req, context, next_handler)
    
    # Should not increase max_tokens if it was already lower
    assert req.max_tokens == 10

@pytest.mark.asyncio
async def test_output_restraint_yagni_default():
    config = OutputRestraintConfig()
    middleware = OutputRestraintMiddleware(config=config)
    req = LLMRequest(intent="test", metadata={"restraint_policy": "yagni"})
    context = PipelineContext()
    next_handler = AsyncMock()
    
    await middleware.process(req, context, next_handler)
    
    # Using YAGNIPolicy which is now configured via OutputRestraintConfig
    policy = YAGNIPolicy(config)
    
    # Restraint middleware typically modifies request.system_prompt if there are no messages, 
    # but the old test checked messages. Let's check both or whatever is populated.
    assert req.system_prompt or req.messages
    if req.messages:
        assert req.messages[0].role == "system"
        assert policy.get_instructions() in req.messages[0].content
    else:
        assert policy.get_instructions() in req.system_prompt
    assert req.max_tokens == policy.get_max_tokens()
