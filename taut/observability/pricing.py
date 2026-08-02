"""Cost estimation for pipeline metrics.

`PipelineMetrics.estimated_cost_saved` was previously read in three places and
assigned in none, so `/v1/taut/metrics` reported `total_cost_saved: 0.0`
unconditionally while the README advertised cost reporting. This module
supplies the missing computation.

Prices come from litellm's maintained model_cost table so they track provider
changes; the small fallback table keeps estimation working for unknown or
locally-served models rather than silently reporting zero.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("taut.pricing")

# USD per single token (input, output). Fallback only -- litellm is preferred.
_FALLBACK_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50e-6, 10.00e-6),
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "o3": (2.00e-6, 8.00e-6),
    "claude-sonnet-4-20250514": (3.00e-6, 15.00e-6),
    "gemini-2.5-flash": (0.30e-6, 2.50e-6),
}

# Models served locally cost nothing per token.
_LOCAL_PREFIXES = ("ollama/", "ollama_chat/", "huggingface/", "local/")


def cost_per_token(model: str | None) -> tuple[float, float]:
    """Return (input_cost_per_token, output_cost_per_token) in USD."""
    if not model:
        return (0.0, 0.0)

    if any(model.startswith(p) for p in _LOCAL_PREFIXES):
        return (0.0, 0.0)

    try:
        import litellm

        entry = litellm.model_cost.get(model)
        if entry:
            return (
                float(entry.get("input_cost_per_token", 0.0) or 0.0),
                float(entry.get("output_cost_per_token", 0.0) or 0.0),
            )
    except Exception as exc:  # pragma: no cover - depends on litellm internals
        logger.debug("litellm pricing unavailable for %s (%s)", model, exc)

    if model in _FALLBACK_PRICING:
        return _FALLBACK_PRICING[model]

    # Unknown model: report zero rather than inventing a number. Callers treat
    # a zero rate as "cost unknown", which keeps savings honest instead of
    # fabricating a headline figure.
    logger.debug("no pricing data for model %s; cost treated as unknown", model)
    return (0.0, 0.0)


def estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost of one call."""
    in_rate, out_rate = cost_per_token(model)
    return (input_tokens * in_rate) + (output_tokens * out_rate)


def is_priced(model: str | None) -> bool:
    """True if we have real pricing data for this model."""
    return cost_per_token(model) != (0.0, 0.0)
