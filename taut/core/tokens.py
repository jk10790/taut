"""Token counting.

Everything the pipeline reports as "tokens saved" flows through here. The
previous implementation estimated tokens as ``len(text) // 4`` inline, which
made the savings figure on ``/v1/taut/metrics`` an unvalidated guess -- and one
that disagreed with the provider's own accounting, since 4 chars/token is a
poor approximation for JSON and source code (the two content types taut
compresses hardest).

litellm ships real tokenizers for the supported model families, so we use it
and keep the heuristic strictly as a fallback for when the model is unknown.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("taut.tokens")

DEFAULT_COUNTING_MODEL = "gpt-4o"

# Average characters per token, used only when a real tokenizer is unavailable.
_CHARS_PER_TOKEN = 4


def count_tokens(text: str, model: str | None = None) -> int:
    """Count tokens in ``text`` for ``model``.

    Falls back to a character heuristic if litellm cannot tokenize for the
    given model, so token counting never raises into the request path.
    """
    if not text:
        return 0
    try:
        import litellm

        return litellm.token_counter(model=model or DEFAULT_COUNTING_MODEL, text=text)
    except Exception as exc:  # pragma: no cover - depends on litellm internals
        logger.debug("token_counter unavailable for model=%s (%s); using heuristic", model, exc)
        return len(text) // _CHARS_PER_TOKEN


def count_message_tokens(messages, model: str | None = None) -> int:
    """Count tokens across a list of taut Messages."""
    if not messages:
        return 0
    total = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += count_tokens(part.get("text", ""), model)
    return total
