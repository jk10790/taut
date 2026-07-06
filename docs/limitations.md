# Limitations & Boundaries

To use `taut` effectively, it is critical to understand its architectural boundaries. `taut` is purely a **Stateless AI Efficiency Middleware** focused strictly on request transformation, optimization, and fallback routing. We intentionally keep it lightweight so it never bloats your infrastructure.

## What `taut` is NOT

### ❌ It is NOT a Persistent Message Queue
`taut` is designed for synchronous, inline interception of HTTP/REST-like LLM requests. We enforce backpressure (e.g., throwing a `CapacityExceededError`), but we don't hold jobs in a database!
- It is **not** a replacement for Prefect, Celery, RabbitMQ, or temporal.io.
- Your client orchestrator must handle catching the `CapacityExceededError` and queueing the job for later if your local compute is saturated.

### ❌ It Relies on Clients for Complex State
`taut` is totally stateless between distinct requests (excluding the explicit semantic cache). 
- It does not maintain conversational memory or manage agentic workflow states. 
- Your application must continue managing its own conversation history (e.g., in a local SQLite DB) and pass the assembled context into `taut`.

### ❌ Streaming Cache Storage is Out of Scope
While `taut` flawlessly *plays back* cached responses as an SSE stream to keep your Chatbot UX smooth, it **does not cache the actual streamed chunks in real-time**. The cache is saved atomically only after a complete, successful generation finishes.

### ❌ No Built-In Authentication or Global Rate Limiting
`taut` assumes it is running within a trusted internal network (like your local Docker network).
- It does not provide user authentication (OAuth, JWT).
- It does not provide global rate limiting for your end-users.
- For these features, keep `taut` safely behind your standard API gateway (e.g., NGINX, Traefik, Kong).

### ❌ Multi-Tenancy Requires Client Discipline
While `taut` provides robust `namespace` isolation to absolutely prevent cache cross-contamination between different users, it relies entirely on the upstream client to pass this namespace safely (e.g., via the `X-Taut-Namespace` header). 

If your backend fails to pass the correct tenant ID, `taut` cannot stop semantic cache collisions on its own. Strict discipline is required on your backend!
