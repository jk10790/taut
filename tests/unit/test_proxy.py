import pytest
import json
from fastapi.testclient import TestClient
from taut.proxy.server import app
from taut.core.pipeline import Pipeline
from taut.core.models import LLMResponse, PipelineMetrics
from unittest.mock import MagicMock

client = TestClient(app)

class MockPipeline:
    def __init__(self):
        self.response_metrics = PipelineMetrics(
            total_tokens_saved=150,
            estimated_cost_saved=0.002
        )
        self.metrics = MagicMock()
        self.metrics.to_dict.return_value = {"total_tokens_saved": 150}
        self.received_requests = []
    
    async def run(self, request):
        self.received_requests.append(request)
        return LLMResponse(
            content="This is a mocked response from taut.",
            model="gpt-4o",
            metrics=self.response_metrics
        )
        
    async def stream(self, request):
        self.received_requests.append(request)
        yield json.dumps({"choices": [{"delta": {"content": "This "}}]})
        yield json.dumps({"choices": [{"delta": {"content": "is "}}]})
        yield json.dumps({"choices": [{"delta": {"content": "a "}}]})
        yield json.dumps({"choices": [{"delta": {"content": "stream."}}]})

@pytest.fixture
def mock_pipeline():
    # Replace the actual pipeline with a mock
    pipeline = MockPipeline()
    app.state.pipeline = pipeline
    return pipeline

def test_chat_completions_sync(mock_pipeline):
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "stream": False
    }
    
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "This is a mocked response from taut."
    assert "taut_metrics" in data
    assert data["taut_metrics"]["total_tokens_saved"] == 150

def test_chat_completions_stream(mock_pipeline):
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Tell me a story."}
        ],
        "stream": True
    }
    
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        content = "".join([chunk for chunk in response.iter_text()])
        
        # Check that it streamed multiple SSE chunks with correct prefix and terminator
        assert "data: " in content
        assert "data: [DONE]\n\n" in content
        assert "stream." in content

def test_chat_completions_tool_forwarding(mock_pipeline):
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What's the weather?"}
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "tool_choice": "auto"
    }
    
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    
    req = mock_pipeline.received_requests[0]
    assert req.tools == [{"type": "function", "function": {"name": "get_weather"}}]
    assert req.tool_choice == "auto"

def test_metrics_endpoint(mock_pipeline):
    response = client.get("/v1/taut/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_tokens_saved" in data
