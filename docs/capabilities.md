# Capabilities

`taut` sits directly between your app and your LLM provider as an intelligent, intercepting proxy. It's built on a 5-layer pipeline that strictly enforces our **Zero-Cost Local North Star** by intercepting, routing, and crushing payloads before they ever hit the wire.

## The 5-Layer Pipeline

### 1. 🔒 Semantic Cache (The Gatekeeper)
Short-circuits redundant requests immediately! If a highly similar query was just asked, `taut` serves the cached response.
- **Fast & Light**: Uses local `ONNX` embeddings (no PyTorch bloat).
- **Flexible**: Backed by In-Memory FAISS for local tests or Redis Vector Search for production.
- **Multi-Tenant**: Native `namespace` isolation ensures customer A never sees customer B's cached answers.

### 2. 🔀 Tiered Routing (Health & Complexity)
Dynamically routes requests to the right provider.
- **By Complexity**: Simple JSON extractions go to fast, cheap models (like local Ollama or `gpt-4o-mini`). Deep reasoning goes to flagship models (like `o3`).
- **By Health (Dynamic Fallback)**: If your local Ollama is overwhelmed (100% CPU) or your primary provider drops a 503, `taut` automatically, seamlessly fails over to a fallback (like Groq Llama-3) so your bot never times out.

### 3. 📦 Payload Compression
Shrinks your prompt context massively (40-80%) before leaving the network!
- **Extensible Plugin System**: Effortlessly hook in custom plugins for XML, Cypher graphs, or messy RSS feeds.
- **Code & JSON**: Strips AST scaffolding from Python and crushes bloated JSON arrays into columnar text.
- **Prose**: Evicts grammatical filler so you only pay for raw semantic meaning.

### 4. 📌 Prefix Alignment
Forces strict structure to maximize KV cache discounts at the provider. Static rules and system instructions stay locked at the top; fast-changing user queries stay at the bottom.

### 5. ✂️ Output Restraint
Injects YAGNI (You Aren't Gonna Need It) constraints based on the routing tier to prevent your LLM from hallucinating unnecessary pleasantries and bloated code blocks.

---

## 🌟 Advanced Resilience

`taut` isn't just about efficiency—it's built to handle provider turbulence gracefully:

* 🌊 **Streaming Cache Playback**: We intercept the raw socket! When a cache hits, we instantly "play back" the response as an SSE stream so your Chatbot UX remains buttery smooth.
* 🛡️ **Resilience Primitives**: Built-in Token Buckets and Circuit Breakers! We act as the traffic cop. If compute limits are hit, we explicitly raise a `CapacityExceededError` so your orchestrator (like Prefect) can handle backpressure effectively.
* 🛠️ **Resilient Defaults**: The SDK swallows provider idiosyncrasies (like missing `cached_tokens` fields from local models) without crashing the entire proxy pipeline!
