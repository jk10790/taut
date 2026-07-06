import os
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8001/v1"
os.environ["OPENAI_API_KEY"] = "dummy"
import pytest
import asyncio
import threading
import uvicorn
import httpx
import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from taut.proxy.server import app as taut_app
import taut.core.config as config
from taut.core.pipeline import create_pipeline

upstream_app = FastAPI()

@upstream_app.post("/v1/chat/completions")
async def completions(request: Request):
    data = await request.json()
    model = data.get("model", "default")
    
    if "gpt-4" in model and "gpt-4o" not in model:
        return JSONResponse(status_code=503, content={"error": "Provider Busy"})
        
    if data.get("stream"):
        async def generate():
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "Hello "}}]}) + "\n\n"
            await asyncio.sleep(0.01)
            yield "data: " + json.dumps({"choices": [{"delta": {"content": f"from {model}"}}]}) + "\n\n"
            await asyncio.sleep(0.01)
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    return JSONResponse({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": f"Hello from {model}"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    })

class ServerThread(threading.Thread):
    def __init__(self, app, port):
        super().__init__()
        self.config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        self.server = uvicorn.Server(self.config)
        
    def run(self):
        self.server.run()
        
    def stop(self):
        self.server.should_exit = True
        self.join()

@pytest.fixture(scope="module")
def e2e_servers():
    test_config = config.TautConfig(
        provider="openai",
        openai_api_key="dummy",
        openai_base_url="http://127.0.0.1:8001/v1",
        cache=config.SemanticCacheConfig(enabled=True, similarity_threshold=1.0),
        fallback_models=["gpt-3.5-turbo"]
    )
    taut_app.state.pipeline = create_pipeline(test_config)
    
    upstream_thread = ServerThread(upstream_app, 8001)
    upstream_thread.start()
    
    taut_thread = ServerThread(taut_app, 8000)
    taut_thread.start()
    
    time.sleep(2)
    yield
    upstream_thread.stop()
    taut_thread.stop()

@pytest.mark.asyncio
async def test_e2e_standard_and_cache(e2e_servers):
    async with httpx.AsyncClient() as client:
        payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "E2E Test 1"}]}
        resp1 = await client.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=10.0)
        assert resp1.status_code == 200
        assert "Hello from gpt-3.5-turbo" in resp1.json()["choices"][0]["message"]["content"]
        
        resp2 = await client.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=10.0)
        assert resp2.status_code == 200
        assert "Hello from gpt-3.5-turbo" in resp2.json()["choices"][0]["message"]["content"]
        assert "taut_metrics" in resp2.json()

@pytest.mark.asyncio
async def test_e2e_fallback_routing(e2e_servers):
    async with httpx.AsyncClient() as client:
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "Fallback Test"}]}
        resp = await client.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=10.0)
        assert resp.status_code == 200
        assert "Hello from gpt-3.5-turbo" in resp.json()["choices"][0]["message"]["content"]

@pytest.mark.asyncio
async def test_e2e_streaming(e2e_servers):
    async with httpx.AsyncClient() as client:
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Stream Test"}], "stream": True}
        chunks = []
        async with client.stream("POST", "http://127.0.0.1:8000/v1/chat/completions", json=payload) as resp:
            assert resp.status_code == 200
            async for chunk in resp.aiter_text():
                if chunk.strip() and not chunk.endswith("[DONE]\n\n"):
                    chunks.append(chunk)
        full_text = "".join(chunks)
        assert "data: " in full_text
        assert "Hello " in full_text
        assert "from gpt-4o" in full_text
