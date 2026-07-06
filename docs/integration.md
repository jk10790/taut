# Integration & Best Practices

`taut` supports two distinct integration paths depending on your architectural needs. We built it to be as frictionless as possible.

---

## 1. The Proxy Server (Drop-In Magic)

The easiest way to integrate `taut` into LangChain, Node.js, Go, Rust, or Ruby applications is by running it as a **FastAPI Proxy**. It perfectly mimics the standard OpenAI API—zero refactoring required.

### Starting the Server
```bash
python -m taut.proxy.server --port 8000
```

### Updating Your Client
Just change your application's base URL to point to the proxy:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1" # Point to taut!
)
# Streaming (SSE) is fully supported, cache playback included!
```

---

## 2. Python Library (The Power User API)

For tight integration where you want to orchestrate the exact prompts, you can embed `taut` directly inside your Python application. We've unified the exports so you can grab what you need directly from `taut.__init__.py`.

```python
import asyncio
from taut import (
    TautConfig, SemanticCache, TieredRoutingConfig, Compression,
    LLMRequest, SystemBlock, ContextBlock, QueryBlock,
    create_pipeline, CapacityExceededError
)

async def main():
    config = TautConfig(
        provider="litellm",
        num_retries=3,
        timeout=60.0,
        fallback_models=["gpt-4o-mini", "groq/llama3"], # Seamless failovers!
        cache=SemanticCache(backend="redis", redis_url="redis://localhost:6379"),
        routing=TieredRoutingConfig(),
        compression=Compression(json=True, code=True)
    )
    pipeline = create_pipeline(config)
    
    request = LLMRequest(
        blocks=[
            SystemBlock(content="You are a helpful assistant."),
            ContextBlock(content='[{"id": 1, "task": "read docs"}]'),
            QueryBlock(content="Summarize these active items")
        ],
        model="gpt-4o",
        namespace="tenant_123"
    )
    
    try:
        response = await pipeline.run(request)
        print(response.content)
    except CapacityExceededError:
        print("Compute is overwhelmed! Queueing job for later...")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Best Practices for Maximum Efficiency

### Avoiding Semantic Cache Collisions
If you send your entire prompt (boilerplate + context + query) as a single `user` message, the Semantic Cache will embed the whole block. Since the boilerplate makes up 90% of the text, completely different queries might trigger a false cache hit!

**Solution:** Always maintain strict separation of concerns.
- **Proxy Clients**: Separate your static boilerplate into the `system` role, and the dynamic query into the `user` role. `taut` builds its semantic cache key *exclusively* from the final `user` message.
- **Python SDK Clients**: Use the modular `PromptBlock` API. `taut` zeroes in on the `QueryBlock` for the cache embedding.

### Utilizing the PromptBlock API
`PromptBlock` lets you construct modular prompts to leverage the Prefix Alignment layer.
- **`SystemBlock`**: Highest stability. Pinned to the top of the context window.
- **`ContextBlock`**: Used for RAG chunks. Ordered but flexible.
- **`QueryBlock`**: Lowest stability. Kept at the absolute bottom.

### Enforcing Multi-Tenancy in the Proxy
Prevent cache contamination by passing the `X-Taut-Namespace` header! If you don't enforce this, customer A might get customer B's cached data.
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-Taut-Namespace: tenant_123" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Query"}]}'
```

---

## Environment Variables
If running as a proxy, configure `taut` using these environment variables:
- `TAUT_PROVIDER`: LLM provider (default: `litellm`).
- `TAUT_API_KEY`: Your provider API key.
- `TAUT_DEFAULT_MODEL`: Default fallback model.
- `TAUT_CACHE_BACKEND`: `redis` or `memory`.
- `TAUT_ROUTING_TIERS`: JSON string mapping tier names to models.
