import pytest
import pytest_asyncio
from taut.core.models import LLMRequest, Message, TokenUsage, LLMResponse, PipelineContext
from taut.core.prompt_blocks import PromptBlock, SystemBlock, ToolsBlock, ContextBlock, QueryBlock
from taut.core.pipeline import Pipeline
from taut.providers.base import BaseProvider

class MockProvider(BaseProvider):
    def __init__(self):
        self.received_messages = []
        
    async def complete(self, request: LLMRequest, context: PipelineContext) -> LLMResponse:
        self.received_messages = request.messages
        return LLMResponse(
            content="Mock response",
            model="mock",
            usage=TokenUsage(input_tokens=10, output_tokens=10),
            finish_reason="stop",
            tool_calls=None,
            raw_response=None
        )
        
    async def complete_stream(self, request, model):
        pass

@pytest.mark.asyncio
async def test_prompt_block_conversion_in_pipeline():
    # According to the plan, if blocks are provided, they are converted to messages
    # at the beginning of the pipeline execution.
    blocks = [
        SystemBlock(content="You are a helpful AI."),
        ToolsBlock(content="Use these tools.", tools=[{"name": "tool1"}]),
        ContextBlock(content="Here is some data."),
        QueryBlock(content="Answer my query.")
    ]
    
    request = LLMRequest(intent="test", blocks=blocks)
    provider = MockProvider()
    
    # We create a bare pipeline with no middleware to test block conversion
    pipeline = Pipeline(middlewares=[], provider=provider)
    await pipeline.run(request)
    
    # Assert messages were populated
    assert provider.received_messages is not None
    assert len(provider.received_messages) == 4
    
    assert provider.received_messages[0].role == "system"
    assert provider.received_messages[0].content == "You are a helpful AI."
    
    assert provider.received_messages[1].role == "user" # Tools block typically translates to user depending on implementation, but order is preserved
    assert provider.received_messages[1].content == "Use these tools."
    
    assert provider.received_messages[2].role == "user" # Context block typically translates to user
    assert provider.received_messages[2].content == "Here is some data."
    
    assert provider.received_messages[3].role == "user"
    assert provider.received_messages[3].content == "Answer my query."
    
    # We also need to check that tools are extracted and placed on the request
    # but the test focuses on order preservation and conversion.
    assert request.messages == provider.received_messages

def test_prompt_block_properties():
    # Test stability hints
    sys_block = SystemBlock(content="sys")
    assert sys_block.stability == "immutable"
    assert sys_block.cache_eligible is True
    
    tools_block = ToolsBlock(content="tools", tools=[])
    assert tools_block.stability == "stable"
    assert tools_block.cache_eligible is True
    
    ctx_block = ContextBlock(content="ctx")
    assert ctx_block.stability == "semi_stable"
    
    q_block = QueryBlock(content="q")
    assert q_block.stability == "dynamic"
