# Capabilities

`taut` sits between your app and your LLM provider as an intercepting proxy or
an in-process pipeline. It is built from five middleware layers, each of which
can be configured independently or disabled entirely.

Every capability on this page is registered in [`claims.yaml`](claims.yaml) and
bound to a test in `tests/claims/`. If a capability is described here without a
passing test, CI fails.

## The 5-Layer Pipeline

Layers run in this order: cache → routing → compression → prefix → restraint.
Routing runs before compression so the tier decision can control whether
compression is worth doing.

### 1. Semantic Cache

Short-circuits redundant requests. Lookup is two-tier: an exact hash match
first, then vector similarity above a configurable threshold.

- **Local embeddings**: ONNX runtime with `Xenova/all-MiniLM-L6-v2`. No PyTorch.
- **Backends**: in-memory FAISS (`IndexFlatIP`, LRU eviction) for development;
  Redis with RediSearch KNN for production.
- **Namespace isolation**: entries are partitioned per namespace, so one tenant
  cannot be served another's cached answer.
- **Keyed on the query**: the embedding is built from the request's intent — the
  `QueryBlock`, or the last user message — not the whole prompt. Embedding the
  full prompt lets a large static system prompt dominate the vector, so
  unrelated questions sharing boilerplate can collide.

Choosing `similarity_threshold` is a precision/recall trade-off. The default is
`0.95`. `pytest tests/benchmarks -m bench` prints the measured curve against
labelled paraphrase and near-miss pairs, and fails if any near-miss pair
collides at the default.

### 2. Tiered Routing

Scores request complexity and selects a model.

- **By complexity**: a heuristic over tool count, token length (including
  `request.context`, where bulk payloads live) and keywords such as "analyze"
  or "recursive". Below 0.3 routes to the simple tier, above 0.7 to complex.
- **Client override**: an explicitly supplied `model` is always respected.
- **Failover**: on a `503`, timeout or other transient error, `taut` retries with
  backoff and then moves to the next configured fallback model.

### 3. Payload Compression

Content type is detected per field, then the matching strategy is applied.

| Content | Strategy | Measured reduction |
|---|---|---|
| JSON | Uniform object arrays → columnar `COLS:`/rows | **54.5%** |
| Python | AST rewrite: docstrings, annotations, comments removed | **58.2%** |
| Prose | 11 filler-phrase substitutions | **~0%** |

Measured on `tests/benchmarks/corpus/` with the `gpt-4o` tokenizer; see
[`baseline.json`](../tests/benchmarks/baseline.json).

The prose figure is honest, not a placeholder: rule-based filler removal does
almost nothing on real text. Do not expect `taut` to compress retrieved
documents or chat history. It compresses structure.

Compression is skipped for the simple tier by default
(`skip_for_simple_tier`), on the reasoning that CPU spent compressing is not
repaid on the cheapest model. Set it to `False` to compress everything.

**Extensible**: register your own compressor for any content type.

```python
from taut import register_compressor

@register_compressor("application/xml", matcher=lambda t: t.strip().startswith("<"))
def compress_xml(data: str) -> str:
    return data.replace("\n", "")
```

### 4. Prefix Alignment

Maximises provider-side KV cache hits by keeping the prompt prefix stable.
System content is moved to the top and dynamic content to the bottom. For
Anthropic, explicit `cache_control: {"type": "ephemeral"}` breakpoints are
injected instead (up to the provider limit of four).

The analyzer also scans system prompts for cache breakers — UUIDs, timestamps,
ISO dates — and reports the count in layer metrics, so you can see what is
defeating your prefix cache.

### 5. Output Restraint

Injects constraints into the system prompt and caps `max_tokens`. Output tokens
are the expensive ones.

| Policy | Effect |
|---|---|
| `yagni` | Forbids filler, caveats and unnecessary code |
| `terse` | Telegraphic style, caps `max_tokens` at 256 |
| `structured` | Forces valid JSON, sets `response_format` |
| `adaptive` | Varies by routing tier: terse for simple, YAGNI for standard, nothing for complex |
| `off` | Disables the layer |

`adaptive` leaves the complex tier unconstrained deliberately — restraint on a
deep-reasoning request costs more in answer quality than it saves in tokens.

---

## Streaming

Cache hits are played back as a chunked stream so a chat UI behaves the same on
a hit as on a miss. On a miss, chunks are forwarded as they arrive and the
complete response is written to cache once generation finishes.

## Resilience

- **Circuit breaker** (on by default): a model that fails `circuit_failure_threshold`
  times consecutively is skipped without being called until `circuit_reset_timeout`
  elapses, then gets one probe. A dead upstream costs one call per window
  instead of a full retry budget on every request.
- **Token bucket** (opt-in via `ResilienceConfig.requests_per_second`): applies
  backpressure before spend. On timeout it raises `CapacityExceededError` for
  your orchestrator to handle — see [Limitations](limitations.md).
- **Retries**: exponential backoff with jitter on transient provider errors.
- **Tolerant defaults**: provider quirks such as a missing `cached_tokens` field
  are absorbed rather than propagated as pipeline failures.

## Observability

Per-layer metrics — tokens before/after, latency, whether the layer applied —
are collected on every request and exposed at `/v1/taut/metrics`.

Token counts use the model's real tokenizer. Cost is computed from litellm's
pricing table; a model with no pricing data reports zero saved rather than an
invented figure.
