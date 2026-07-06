# taut — AI Efficiency Middleware

> **Zero Waste Compute.** A drop-in library and proxy server that intercepts, compresses, and routes LLM requests through a 5-layer optimization pipeline to minimize API costs and latency without degrading output accuracy.

## Why taut? (Use Cases)
Modern LLM applications waste massive amounts of compute. `taut` is designed for high-throughput AI pipelines where cost and latency matter:

* 📚 **RAG (Retrieval-Augmented Generation)**: You are passing 20,000 tokens of retrieved documents per query. `taut` compresses the documents down to 12,000 tokens (a 20-60% reduction depending on content type) using rule-based compression (Layer 3), and pins the context for OpenAI KV Cache discounts (Layer 4).
* 🤖 **Customer Support Chatbots**: Users often ask the exact same questions ("How do I reset my password?"). `taut` semantically caches the answers in Redis (Layer 1). The LLM is completely bypassed for 40% of your traffic.
* 🛠️ **Automated Code Review (CI/CD)**: You are sending massive codebases to an LLM. `taut` strips comments, docstrings, and type-hints from the AST (Layer 3) before sending, saving thousands of tokens.
* 🔀 **Multi-Agent Systems**: You have simple formatting agents and deep-reasoning agents. `taut` automatically routes simple tasks to `gpt-4o-mini` and complex tasks to `o3` based on prompt heuristics (Layer 2).

---

## Capabilities: The 5-Layer Pipeline

`taut` sits between your application and your LLM provider, utilizing a **Chain of Responsibility** pattern:

1. **🔒 Semantic Cache**: Short-circuits the request if a highly similar intent was recently processed. Supports In-Memory (FAISS) or Distributed (Redis Vector Search) backends using lightning-fast `ONNX` embeddings (zero PyTorch bloat). `namespace` isolation is built-in for multi-tenant apps.
2. **🔀 Tiered Routing**: Evaluates request complexity on the fly. Routes simple tasks to fast/cheap models, reserving deep reasoning tasks for flagship models.
3. **📦 Payload Compression**: Content-aware token reduction. Converts JSON arrays to columnar formats, strips AST from Python code, and removes filler words from prose (tier-aware).
4. **📌 Prefix Alignment**: Restructures prompts to pin static content at the top, maximizing provider-side KV cache hits. Uses the `PromptBlock` API for stability hints.
5. **✂️ Output Restraint**: Employs an `AdaptivePolicy` to dynamically inject strict YAGNI constraints or terse formatting rules based on the request tier, ensuring models return exactly what was asked and nothing more.

---

## Integration

`taut` supports **two** integration paths depending on your architecture.

### Option 1: The Proxy Server (Any Language / Framework)
The easiest way to use `taut` with LangChain, Node.js, Go, or Ruby is via the **FastAPI Proxy**. It perfectly mimics the OpenAI API.

1. Install the SDK (with cache dependencies if needed):
```bash
pip install "taut[cache]"
```
2. Start the server:
```bash
python -m taut.proxy.server --port 8000
```
3. Change your application's base URL:
```python
# In your existing application (e.g. LangChain or OpenAI SDK)
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1" # Point to taut!
)
# All requests are now cached, compressed, and routed automatically.
# Streaming (SSE) is fully supported!
```

### Option 2: Python Library
For tight integration and programmatic control, use the native Python library.

```python
import asyncio
import taut

async def main():
    # Initialize the pipeline via TautConfig
    config = taut.TautConfig(
        provider="litellm",
        num_retries=3,
        timeout=60.0,
        fallback_models=["gpt-4o-mini"],
        cache=taut.SemanticCacheConfig(backend="redis", redis_url="redis://localhost:6379"),
        routing=taut.TieredRoutingConfig(),
        compression=taut.CompressionConfig(json=True, code=True)
    )
    pipeline = taut.create_pipeline(config)
    
    # Send a request using the PromptBlock API
    request = taut.LLMRequest(
        blocks=[
            taut.SystemBlock(content="You are a helpful assistant."),
            taut.ContextBlock(content='[{"id": 1, "task": "read docs"}, {"id": 2, "task": "write code"}]'),
            taut.QueryBlock(content="Summarize these active items")
        ],
        model="gpt-4o",
        namespace="tenant_123" # Isolates cache entries
    )
    
    response = await pipeline.run(request)
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Advanced Features & Configuration

### Avoiding Semantic Cache Collisions (Best Practices)
If you send your entire prompt (including long static instructions and few-shot examples) as a single `user` message, the Semantic Cache will embed the entire block. Because 90% of the text is boilerplate, completely different queries may mistakenly exceed the `0.95` cosine similarity threshold, leading to false cache hits!

To ensure maximum efficiency and prevent collisions, always adhere to the following strict separation of concerns:
* **Proxy Clients**: Separate your static boilerplate into the `system` role and your dynamic query/content into the `user` role. `taut` builds its semantic cache key *exclusively* from the final `user` message.
* **Python SDK Clients**: Use the `PromptBlock` API. Put your instructions in a `SystemBlock` and the dynamic request in a `QueryBlock`. `taut` uses the `QueryBlock` for the cache embedding.

### The PromptBlock API
When using the Python library, `PromptBlock` allows you to construct modular prompts that the `taut` Prefix Alignment layer can optimize.
* **`SystemBlock`**: Highest stability. Always pinned to the top of the context window.
* **`ContextBlock`**: Used for RAG chunks or tool outputs. `taut` will try to preserve order but may shift these down if they change frequently.
* **`QueryBlock`**: Lowest stability (highly dynamic user input). Kept at the absolute bottom.

These blocks are converted to standard `Message` objects under the hood, but their `cache_eligible` and `stability` metadata are preserved, drastically increasing provider-side KV cache hits.

### Proxy Multi-Tenancy
If using the FastAPI proxy in a multi-tenant application, cache contamination is a serious risk. Prevent this by passing the `X-Taut-Namespace` header in your HTTP requests:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-Taut-Namespace: tenant_123" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "What is my balance?"}]}'
```
This isolates the Semantic Cache (FAISS or Redis) so `tenant_456` never receives a cached response belonging to `tenant_123`.

### Environment Variables
When running `taut` as a Proxy Server (or initializing via `TautConfig.from_env()`), you can configure the pipeline using the following environment variables:
* **`TAUT_PROVIDER`**: The LLM provider to use (default: `litellm`).
* **`TAUT_API_KEY`**: Your provider API key.
* **`TAUT_BASE_URL`**: Optional custom base URL for your provider.
* **`TAUT_DEFAULT_MODEL`**: The default fallback model if routing isn't specified.
* **`TAUT_CACHE_BACKEND`**: The semantic caching backend, e.g., `redis` or `memory`.
* **`TAUT_EMBEDDING_MODEL`**: The Hugging Face repo ID for the ONNX embedding model (default: `Xenova/all-MiniLM-L6-v2`).
* **`TAUT_ROUTING_TIERS`**: A JSON string overriding the default model tiers. Example: `'{"simple": ["llama-3-8b"], "standard": ["gpt-4o"], "complex": ["claude-3.5-sonnet"]}'`

### Configuration Deep Dive
`TautConfig` manages the entire pipeline and allows you to tune the middleware:
* **`cache`**: Set `SemanticCacheConfig(backend="redis", redis_url="...")` or use `"memory"` (FAISS). You can also inject custom `ONNX` embedders here.
* **`routing`**: Use `TieredRoutingConfig(complexity_keywords=["analyze", "synthesize"])`. By default, it routes to `gpt-4o-mini` for simple tasks and your main model for complex ones.
* **`compression`**: Configure `CompressionConfig(json=True, code=True, skip_for_simple_tier=True)`. Code is compressed via AST pruning; JSON is flattened to columnar tables.
* **Retries & Fallbacks**: Configure `num_retries=3`, `timeout=60.0`, and `fallback_models=["gpt-3.5-turbo"]` directly in `TautConfig` to handle provider rate-limits.

---

## Out of Scope
`taut` is focused entirely on request transformation and optimization. The following concerns are **out of scope** and should be handled by your application or API gateway:
* Authentication & Authorization
* Rate Limiting (we provide built-in Retry/Timeout configs, but not rate limiting)
* Streaming Cache Storage (taut will optimize the request but not cache the streamed response chunks)
* Multi-tenancy enforcement (taut provides `namespace` for cache isolation, but your app must pass it in safely)

---

## Observability

You can't optimize what you can't measure. `taut` provides deep, layer-by-layer observability into your pipeline's efficiency.

If using the Proxy, simply hit `GET /v1/taut/metrics`. If using Python, call `pipeline.metrics.summary()`:

```text
┌─────────────────────────┬──────────────┐
│ Total Requests          │        1,247 │
│ Cache Hit Rate          │        34.2% │
│ Total Tokens Saved      │    2,450,123 │
│ Estimated Cost Saved    │       $47.82 │
│ Model Distribution      │              │
│   gpt-4o-mini           │        41.3% │
│   gpt-4o                │        48.2% │
│   o3                    │        10.5% │
└─────────────────────────┴──────────────┘
```
The `EventBus` also allows you to hook directly into the pipeline to stream these metrics to Datadog, Prometheus, or Grafana.

---

## Supported Providers
`taut` uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, meaning it supports 100+ LLM providers out of the box including OpenAI, Anthropic, Google Gemini, Azure, Cohere, and local open-source models via Ollama.

## License
MIT
