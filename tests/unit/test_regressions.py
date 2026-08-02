"""Regression tests for defects found in the README-vs-code audit.

Each test here corresponds to a specific behaviour the documentation claimed
but the code did not deliver. They exist so those claims cannot silently
become false again.
"""
import json

import pytest

from taut.core.config import (
    CompressionConfig,
    OutputRestraintConfig,
    PrefixAlignmentConfig,
    SemanticCacheConfig,
    TieredRoutingConfig,
)
from taut.core.models import LLMRequest, Message, PipelineContext
from taut.layers.compression.detector import ContentDetector
from taut.layers.restraint.middleware import OutputRestraintMiddleware
from taut.layers.routing.classifier import HeuristicClassifier

# --------------------------------------------------------------------------
# Content detection: bare source code must reach the AST compressor.
# --------------------------------------------------------------------------


def test_bare_python_detects_as_code_not_prose():
    """is_prose() used to run before is_code(), so any source containing an
    attribute access read as prose and the AST compressor never ran."""
    code = (
        "def calculate(data):\n"
        "    total = 0\n"
        "    for item in data:\n"
        '        if item.get("status") == "active":\n'
        "            total += item.value\n"
        "    return total\n"
    )
    assert ContentDetector().detect(code) == "code"


def test_prose_containing_python_keywords_still_detects_as_prose():
    text = "If you def a class with import and return, it might look like code."
    assert ContentDetector().detect(text) == "prose"


def test_json_still_wins_over_code():
    assert ContentDetector().detect('[{"a": 1}, {"a": 2}]') == "json"


# --------------------------------------------------------------------------
# Output restraint: "off" must be off, "adaptive" must be reachable.
# --------------------------------------------------------------------------


def test_restraint_policy_off_is_actually_off():
    """policy="off" fell through the dict lookup default and became YAGNI."""
    mw = OutputRestraintMiddleware(OutputRestraintConfig(policy="off"))
    assert mw.enabled is False
    assert mw.default_policy is None


def test_restraint_adaptive_policy_is_registered():
    mw = OutputRestraintMiddleware(OutputRestraintConfig(policy="adaptive"))
    assert mw.default_policy is not None
    assert mw.default_policy.name == "adaptive"


def test_restraint_rejects_unknown_policy():
    cfg = OutputRestraintConfig.model_construct(
        policy="nonsense", max_tokens_cap=None, custom_constraints=[]
    )
    with pytest.raises(ValueError, match="Unknown output restraint policy"):
        OutputRestraintMiddleware(cfg)


@pytest.mark.asyncio
async def test_restraint_off_injects_nothing():
    from unittest.mock import AsyncMock

    mw = OutputRestraintMiddleware(OutputRestraintConfig(policy="off"))
    req = LLMRequest(intent="test", system_prompt="Original.")
    await mw.process(req, PipelineContext(), AsyncMock())
    assert req.system_prompt == "Original."
    assert req.max_tokens is None


@pytest.mark.asyncio
async def test_adaptive_policy_varies_by_tier():
    """The tier-aware behaviour docs/capabilities.md describes requires the
    pipeline context to reach the policy."""
    from unittest.mock import AsyncMock

    mw = OutputRestraintMiddleware(OutputRestraintConfig(policy="adaptive"))

    simple_ctx = PipelineContext()
    simple_ctx.selected_tier = "simple"
    simple_req = LLMRequest(intent="t", system_prompt="S")
    await mw.process(simple_req, simple_ctx, AsyncMock())

    complex_ctx = PipelineContext()
    complex_ctx.selected_tier = "complex"
    complex_req = LLMRequest(intent="t", system_prompt="S")
    await mw.process(complex_req, complex_ctx, AsyncMock())

    # Simple tier gets terse instructions and a tight cap.
    assert "terse" in simple_req.system_prompt.lower()
    assert simple_req.max_tokens == 256
    # Complex tier is left alone -- restraint would cost reasoning quality.
    assert complex_req.system_prompt == "S"
    assert complex_req.max_tokens is None


def test_custom_constraints_are_applied():
    cfg = OutputRestraintConfig(policy="yagni", custom_constraints=["Never use emoji."])
    mw = OutputRestraintMiddleware(cfg)
    assert mw.config.custom_constraints == ["Never use emoji."]


# --------------------------------------------------------------------------
# Routing: request.context is the bulk payload and must be scored.
# --------------------------------------------------------------------------


def test_classifier_scores_request_context():
    """A 20KB context scored 0.0 and routed to the cheapest tier, which then
    tripped skip_for_simple_tier and disabled compression too."""
    big = json.dumps([{"id": i, "name": f"Item {i}", "status": "active"} for i in range(400)])
    result = HeuristicClassifier().calculate_score(LLMRequest(intent="Summarize", context=big))
    assert result.score > 0.0
    assert result.tier != "simple"


def test_classifier_still_routes_trivial_requests_to_simple():
    result = HeuristicClassifier().calculate_score(LLMRequest(intent="hi", messages=[]))
    assert result.tier == "simple"


# --------------------------------------------------------------------------
# Semantic cache keying: embed the query, not the whole request.
# --------------------------------------------------------------------------


class _StubEmbedder:
    def embed(self, text):
        return [0.0] * 384

    def dimension(self):
        return 384


def _cache_middleware():
    from taut.layers.cache.middleware import SemanticCacheMiddleware

    return SemanticCacheMiddleware(SemanticCacheConfig(embedder=_StubEmbedder()))


def test_block_requests_derive_intent_from_query_block():
    from taut.core.pipeline import Pipeline
    from taut.core.prompt_blocks import ContextBlock, QueryBlock, SystemBlock

    req = LLMRequest(
        blocks=[
            SystemBlock(content="X" * 400),
            ContextBlock(content="ctx"),
            QueryBlock(content="What is our Q1 revenue?"),
        ]
    )
    Pipeline(middlewares=[], provider=None)._convert_blocks_to_messages(req)
    assert req.intent == "What is our Q1 revenue?"


def test_cache_text_is_the_query_not_the_whole_request():
    from taut.core.pipeline import Pipeline
    from taut.core.prompt_blocks import QueryBlock, SystemBlock

    req = LLMRequest(
        blocks=[SystemBlock(content="X" * 4000), QueryBlock(content="short query")]
    )
    Pipeline(middlewares=[], provider=None)._convert_blocks_to_messages(req)
    text = _cache_middleware()._cache_text(req)
    assert text == "short query"


def test_cache_text_falls_back_to_last_user_message():
    req = LLMRequest(
        messages=[
            Message(role="system", content="Y" * 2000),
            Message(role="user", content="the actual question"),
        ]
    )
    assert _cache_middleware()._cache_text(req) == "the actual question"


# --------------------------------------------------------------------------
# Config surface: fields that are declared must be read.
# --------------------------------------------------------------------------


def test_prefix_alignment_respects_provider_hint():
    from taut.layers.prefix.middleware import PrefixAlignmentMiddleware

    mw = PrefixAlignmentMiddleware(PrefixAlignmentConfig(provider_hints="anthropic"))
    resolved = mw._resolve_provider(LLMRequest(intent="t", model="gpt-4o"), PipelineContext())
    assert resolved == "anthropic"


def test_prefix_alignment_auto_infers_provider_from_model():
    from taut.layers.prefix.middleware import PrefixAlignmentMiddleware

    mw = PrefixAlignmentMiddleware(PrefixAlignmentConfig(provider_hints="auto"))
    ctx = PipelineContext()
    ctx.selected_model = "claude-sonnet-4-20250514"
    assert mw._resolve_provider(LLMRequest(intent="t"), ctx) == "anthropic"


def test_removed_config_fields_are_gone():
    """These were declared but never read; leaving them invites configuring a
    setting that does nothing."""
    for dead in ("min_tokens_to_compress", "llmlingua_target_ratio"):
        assert dead not in CompressionConfig.model_fields
    assert "static_blocks" not in PrefixAlignmentConfig.model_fields


def test_every_declared_config_field_is_read_somewhere():
    """Guard against re-introducing settings that silently do nothing."""
    import pathlib
    import re

    source = "\n".join(
        p.read_text()
        for p in pathlib.Path("taut").rglob("*.py")
        if p.name != "config.py"
    )
    configs = [
        CompressionConfig,
        PrefixAlignmentConfig,
        TieredRoutingConfig,
        OutputRestraintConfig,
        SemanticCacheConfig,
    ]
    unread = [
        f"{cfg.__name__}.{field}"
        for cfg in configs
        for field in cfg.model_fields
        if not re.search(rf"\b{re.escape(field)}\b", source)
    ]
    assert not unread, f"config fields declared but never read: {unread}"
