# Limitations & Boundaries

`taut` is a stateless request-transformation middleware. It optimises and routes
individual requests; it does not manage state, workflows or infrastructure.
This page is the honest counterpart to [Capabilities](capabilities.md) —
capabilities listed there are test-backed, and limitations listed here are
things we have deliberately not built.

## Compression only helps structured payloads

The compression layer earns its keep on JSON, logs and source code. On prose it
does effectively nothing: `ProseCompressor` is eleven filler-phrase regexes and
measures ~0% on real retrieved documents (see
[`baseline.json`](../tests/benchmarks/baseline.json)).

If your context is mostly retrieved natural-language documents, `taut`'s
compression layer will not shrink it. Use the caching and routing layers, and
handle document reduction before it reaches `taut`.

## No health-based or load-aware routing

`taut` routes by *complexity*, not by provider health. There is no CPU probing,
no latency-based steering and no load metric anywhere in the codebase.

What does exist:

- **Failover on error** — a `503`, timeout or transient failure moves the
  request to the next configured fallback model.
- **Circuit breaking** — a model that fails repeatedly is skipped without being
  called for a cooldown window.

Both are reactive. If your local Ollama is saturated but still accepting
connections slowly, `taut` will keep sending it traffic until requests actually
fail.

## Not a persistent message queue

`taut` intercepts synchronous requests inline. It enforces backpressure by
raising `CapacityExceededError` when the token bucket is exhausted, but it does
not hold jobs anywhere.

It is not a replacement for Prefect, Celery, RabbitMQ or Temporal. Your caller
must catch `CapacityExceededError` and decide what to do:

```python
from taut import CapacityExceededError

try:
    response = await pipeline.run(request)
except CapacityExceededError:
    await queue.enqueue(job)  # your orchestrator's problem, not taut's
```

Rate limiting is opt-in. With the default `ResilienceConfig`, no limiter is
installed and `CapacityExceededError` is never raised.

## Relies on the client for conversational state

`taut` holds no memory between requests beyond the semantic cache. It does not
track conversation history or agentic workflow state. Assemble your context and
pass it in.

## Streamed chunks are not cached incrementally

A cache hit is replayed as a stream, but a cache *miss* is only written once
generation completes. An interrupted generation caches nothing.

## No authentication or global rate limiting

`taut` assumes it runs inside a trusted network. It provides no user
authentication and no per-user quota enforcement. Put it behind your API
gateway (NGINX, Traefik, Kong) for those.

The token bucket is a backpressure mechanism protecting your compute budget —
not a multi-tenant fairness mechanism.

## Multi-tenancy requires the caller to pass a namespace

Namespace isolation works, but only if you use it. The proxy reads
`X-Taut-Namespace`; the SDK reads `LLMRequest.namespace`. If your backend does
not pass a tenant identifier, every tenant shares the `default` namespace and
can be served each other's cached answers.

There is no way for `taut` to infer tenancy on its own. Treat passing the
namespace as a security requirement, not a nicety.

## Cost figures are estimates

Cost savings are computed from litellm's pricing table against a counterfactual
("what this request would have cost unoptimised"). Two caveats:

- A model with no pricing entry reports **zero** saved, not a guess. Locally
  served models are priced at zero by design.
- The counterfactual assumes you would otherwise have sent the uncompressed
  payload to the model routing displaced. That is an estimate, not billing data.
  Reconcile against your provider invoice.
