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

## Columnar JSON costs small models their column alignment

`JSONCompressor` turns an array of uniform records into a header plus rows:

```
COLS: event_id | user_id | action | status_code | region | latency_ms | timestamp
0 | 1000 | login | 404 | ap-south-1 | 414.66 | 2026-07-01T00:15:00Z
1 | 1001 | login | 201 | eu-west-2 | 470.31 | 2026-07-02T01:15:00Z
```

The transform is lossless — every value survives — and it is where roughly 55%
of the measured token saving comes from. But the column names appear *once*, at
the top. A weaker model reading row 74 has to carry seven column positions in
its head across a hundred lines, and some cannot.

Measured by the fidelity suite (`tests/benchmarks/test_fidelity.py`, 22 tasks,
both models recorded at temperature 0):

| task arm | what it asks | gpt-4o-mini raw → compressed | claude-haiku-4-5 raw → compressed |
|---|---|---|---|
| retrieval (4 tasks) | one named row, e.g. the status of `SKU-00042` | 4 → 4 | 4 → 4 |
| aggregation, narrow gap (3) | extremum where the runner-up is within 2% | 2 → **0** | 3 → 3 |
| aggregation, wide gap (2) | extremum where the runner-up is 35%+ away | 1 → **0** | 1 → 1 |

The failure is specific and visible in the answers. Asked which `event_id` has
the highest `latency_ms`, gpt-4o-mini answers `74` on the raw JSON and `1074`
on the columnar payload — `1074` is the `user_id` of event 74. It found the
right row and read the wrong column.

Two things this is *not*:

- **Not a near-tie effect.** `events.fastest_event` has a 35.8% gap between the
  correct answer and the runner-up and still regresses. Margin width does not
  predict the failure.
- **Not data loss.** Single-row retrieval is clean on both models, all four
  tasks, both payloads. `test_compression_never_regresses_retrieval` pins this
  invariant with no exemptions.

So the practical boundary: **if you send large uniform record sets to a small
model and ask it to scan a column** — max, min, "which row has the highest X" —
compression can change the answer. Retrieval, code and prose questions are
unaffected, and a mid-tier model handles the aggregations fine.

Mitigations, in order of cost:

- Route aggregation-style questions to a stronger model. The routing layer
  already exists for this.
- Do the aggregation yourself before the payload reaches the model. Asking an
  LLM to find a maximum over a hundred rows is an expensive way to run `max()`.
- Disable compression for that call by leaving `compression` unset on the
  request's config.

Repeating the `COLS:` header every N rows would likely help, and is deliberately
not implemented: it would be tuning the wire format to one weak model's failure
mode, and it invalidates every recorded cassette. The exempted cases are pinned
in `KNOWN_REGRESSIONS` in the fidelity suite, which fails if the list grows *or*
if a listed case quietly starts passing.

## The semantic cache is, in practice, close to an exact-match cache

At the shipped `similarity_threshold` of `0.95`, measured against 50 labelled
paraphrase and near-miss pairs (`tests/benchmarks/test_cache_precision.py`):

| threshold | recall | precision | false hits | with guard |
|---|---|---|---|---|
| 0.80 | 96% | 65% | 13 | 8 |
| 0.85 | 88% | 76% | 7 | 3 |
| 0.90 | 56% | 74% | 5 | 2 |
| **0.95** | **12%** | **60% → 75%** | **2** | **1** |

Only trivially reworded questions hit — "how do I reset my password" against
"how can I reset my password". A genuine paraphrase such as "list all active
users" against "show me every active user" scores 0.879 and misses.

Lowering the threshold does not fix this, because the classes overlap badly:
the worst paraphrase scores 0.790 while the worst near-miss scores 0.995. No
threshold is both safe and useful.

> **These numbers replace an earlier, rosier table.** The first version of this
> benchmark used 13 pairs and reported 100% precision with zero false hits at
> 0.95. That was the corpus flattering the cache, not the cache being good:
> expanding to 50 pairs written blind to the model's scores exposed false hits
> at the shipped default. A benchmark small enough to pass is worse than none.

### The discriminative guard

Vector similarity answers "are these about the same thing?", not "do they have
the same answer". The gap is concentrated in high-information tokens: one month
name in nine words barely moves a pooled vector, but it changes the answer
completely.

So a semantic hit must now clear a second bar — an exact match on numbers,
years, months and quarters (`taut/layers/cache/guards.py`). Measured: it blocks
**8 of 25** near-miss pairs and **0 of 25** paraphrases. It can only ever turn
a hit into a miss, so it is on by default; set
`SemanticCacheConfig(discriminative_guard=False)` to disable it.

```
"show costs for July 2026"  vs  "show costs for June 2026"
  cosine 0.940  -> above any usable threshold
  guard         -> VETOED, {july} != {june}
```

### What the guard cannot fix

One class of collision survives, and no token-based rule can reach it:

```
0.978  'is the Pro plan cheaper than Growth' | 'is the Growth plan cheaper than Pro'
0.913  'who approved this pull request'      | 'who requested this pull request'
0.862  'list all active users'               | 'list all inactive users'
```

The first pair has an *identical* bag of words and differs only in argument
order. Separating it needs order-sensitive comparison, not a better threshold
and not a bigger token set. `test_default_threshold_admits_no_false_hits` is
marked `xfail(strict=True)` on exactly this pair, so if it is ever resolved the
suite will say so.

This is a property of `all-MiniLM-L6-v2` on short queries, not of the cache
plumbing. A stronger model shifts the numbers without closing the gap: measured
on `bge-small-en-v1.5`, recall at 0.95 rises from 12% to 40% but false hits do
not reach zero either.

Expect the cache to earn its keep on genuinely repeated requests — retries,
polling, fan-out over identical prompts — rather than on natural-language
variety. The default is deliberately tuned for safety: serving a confidently
wrong answer costs more than a cache miss.

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
