<div align="center">
  <h1><code>taut</code></h1>
  <p><b>AI Efficiency Middleware</b></p>
  <p><i>Zero-cost local North Star caching, routing, and compression for LLM pipelines.</i></p>
</div>

---

## ⚡️ Zero Waste Compute

Modern LLM applications waste massive amounts of compute. **taut** is a drop-in library and proxy server that intercepts, compresses, and routes LLM requests through an advanced optimization pipeline. Its North Star is **Zero Waste Compute**—minimizing API costs and execution latency without degrading the accuracy of the AI's output.

If data can be cached locally, stripped of boilerplate, or handled by a cheaper model, it *never* reaches your expensive flagship LLM.

### Why choose `taut`?

- 📚 **RAG Optimization**: Compresses retrieved documents (up to 60% reduction) and aligns contexts to maximize provider KV Cache discounts.
- 🤖 **Semantic Caching**: Completely bypass the LLM for highly similar intents. `taut` intelligently serves cached responses for repetitive traffic.
- 🛠️ **Payload Compression**: Strips ASTs from code, transforms JSONs into columnar layouts, and removes grammatical scaffolding before it hits the network.
- 🔀 **Dynamic Routing**: Autonomously directs simple extractions to `gpt-4o-mini` and preserves expensive `o3` calls strictly for deep-reasoning tasks.

---

## 📖 Documentation

Dive deeper into what makes `taut` the ultimate middleware for your AI applications:

- [Capabilities & Features](docs/capabilities.md) — Discover the 5-layer pipeline, Streaming Cache Playback, and Resilience Primitives.
- [Limitations & Scope](docs/limitations.md) — Understand what `taut` *is not*, and how it fits into your broader architecture.
- [Integration & Best Practices](docs/integration.md) — Learn how to set up the proxy, use the Python SDK, and configure cache isolation.

---

## 🚀 Quick Start

The easiest way to use `taut` with LangChain, Node.js, Go, or Ruby is via the **FastAPI Proxy**.

```bash
# Install the SDK with cache dependencies
pip install "taut[cache]"

# Start the proxy server
python -m taut.proxy.server --port 8000
```

Point your existing OpenAI-compatible client to the proxy:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1" # Point to taut!
)
```

## 📊 Observability

`taut` provides deep, layer-by-layer observability out of the box. Measure your cache hit rates, token savings, and cost reductions directly via `/v1/taut/metrics`.

## License

MIT
