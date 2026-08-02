# Integration & Best Practices

Two integration paths: run `taut` as a proxy, or embed it as a library.

Every Python example on this page is executed by
`tests/claims/test_docs_examples.py`, and every symbol it references is checked
against the package. If an example here stops working, CI fails.

---

## 1. The Proxy Server

The easiest path for LangChain, Node.js, Go, Rust or Ruby applications. The
proxy exposes an OpenAI-compatible API, so pointing an existing client at it is
the only change required.

### Install and start

```bash
# The proxy needs FastAPI and uvicorn from the `proxy` extra.
pip install -e ".[cache,proxy]"

python -m taut.proxy.server --port 8000
```

### Point your client at it

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1",
)
```

Streaming works: set `stream=True` and responses arrive as server-sent events,
including on a cache hit, which is replayed chunk by chunk.

---

## 2. Python Library

For tighter control over prompt assembly.

```python
import asyncio

from taut import (
    CapacityExceededError,
    CompressionConfig,
    ContextBlock,
    LLMRequest,
    QueryBlock,
    SemanticCacheConfig,
    SystemBlock,
    TautConfig,
    TieredRoutingConfig,
    create_pipeline,
)


async def main():
    config = TautConfig(
        provider="litellm",
        num_retries=3,
        timeout=60.0,
        fallback_models=["gpt-4o-mini"],
        cache=SemanticCacheConfig(backend="memory", similarity_threshold=0.95),
        routing=TieredRoutingConfig(),
        compression=CompressionConfig(json=True, code=True),
    )
    pipeline = create_pipeline(config)

    request = LLMRequest(
        blocks=[
            SystemBlock(content="You are a helpful assistant."),
            ContextBlock(content='[{"id": 1, "task": "read docs"}]'),
            QueryBlock(content="Summarize these active items"),
        ],
        model="gpt-4o",
        namespace="tenant_123",
    )

    try:
        response = await pipeline.run(request)
        print(response.content)
    except CapacityExceededError:
        print("Compute saturated; queue this job for later.")


if __name__ == "__main__":
    asyncio.run(main())
```

For Redis-backed caching in production, swap the cache config:

```python
from taut import SemanticCacheConfig

cache = SemanticCacheConfig(
    backend="redis",
    redis_url="redis://localhost:6379",
    similarity_threshold=0.95,
)
```

Requires the `redis` extra: `pip install -e ".[redis]"`.

---

## Best Practices

### Keep the query separate from the boilerplate

The semantic cache embeds the *dynamic* part of your request, not all of it. It
finds that part in one of three ways, in order:

1. `LLMRequest.intent`, if you set it.
2. The `QueryBlock`, if you use the PromptBlock API.
3. The last `user` message.

If you concatenate boilerplate, context and question into a single user message,
the boilerplate dominates the embedding and unrelated questions can collide
above the similarity threshold. Keep static content in the `system` role or a
`SystemBlock`.

### Use the PromptBlock API for prefix alignment

Blocks carry stability hints that drive prefix ordering:

| Block | Stability | Placement |
|---|---|---|
| `SystemBlock` | immutable | pinned to the top |
| `ToolsBlock` | stable | near the top |
| `ContextBlock` | semi-stable | middle |
| `QueryBlock` | dynamic | bottom, and used as the cache key |

### Pass a namespace for every tenant

Namespace isolation is enforced, but only on the value you pass. Omit it and
every tenant shares the `default` namespace.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-Taut-Namespace: tenant_123" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Query"}]}'
```

Treat this as a security requirement. `taut` cannot infer tenancy.

### Tune the similarity threshold against your own traffic

The default of `0.95` is validated against the labelled pairs in
`tests/benchmarks/test_cache_precision.py`. Your queries are not those queries.
Add your own paraphrase and near-miss pairs and rerun
`pytest tests/benchmarks -m bench` to see the precision/recall curve for your
domain before lowering it.

### Enable backpressure if local compute is the bottleneck

```python
from taut import TautConfig
from taut.core.config import ResilienceConfig

config = TautConfig(
    resilience=ResilienceConfig(
        requests_per_second=10,
        burst=20,
        acquire_timeout=5.0,
    )
)
```

Without `requests_per_second`, no limiter is installed and
`CapacityExceededError` is never raised.

---

## Environment Variables

Used by the proxy when no config is passed explicitly:

- `TAUT_PROVIDER` — LLM provider (default: `litellm`)
- `TAUT_API_KEY` — provider API key
- `TAUT_BASE_URL` — provider base URL
- `TAUT_DEFAULT_MODEL` — model used when routing is disabled
- `TAUT_CACHE_BACKEND` — `memory` or `redis`
- `TAUT_EMBEDDING_MODEL` — ONNX embedding model id
- `TAUT_ROUTING_TIERS` — JSON object mapping tier names to model lists
- `TAUT_PROXY_HOST` — bind address (default: `127.0.0.1`)
- `TAUT_PROXY_CORS_ORIGINS` — comma-separated CORS origins (default: `*`)
