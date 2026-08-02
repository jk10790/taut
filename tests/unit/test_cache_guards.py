"""The lexical veto on semantic cache hits.

These tests pin the rule itself. The measured effect on real embeddings --
how many labelled near-misses it blocks, and that it blocks no paraphrases --
lives in tests/benchmarks/test_cache_precision.py.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from taut.core.config import SemanticCacheConfig
from taut.core.models import LLMRequest, LLMResponse, PipelineContext, TokenUsage
from taut.layers.cache.backends.base import CacheEntry
from taut.layers.cache.guards import blocks_reuse, discriminative_tokens
from taut.layers.cache.middleware import SemanticCacheMiddleware

class TestDiscriminativeTokens:
    def test_ignores_ordinary_words(self):
        assert discriminative_tokens("how do I reset my password") == frozenset()

    def test_picks_up_anything_containing_a_digit(self):
        assert discriminative_tokens("what does error code 429 mean") == {"429"}
        assert discriminative_tokens("what is the p99 latency") == {"p99"}
        assert discriminative_tokens("show costs for 2026") == {"2026"}

    def test_keeps_iso_dates_whole(self):
        """Splitting on punctuation would make 2026-07-01 and 2026-01-07 equal as sets."""
        assert discriminative_tokens("the deployment on 2026-07-01") == {"2026-07-01"}
        assert blocks_reuse("deploy on 2026-07-01", "deploy on 2026-01-07")

    def test_picks_up_months_and_quarters(self):
        assert discriminative_tokens("show costs for July") == {"july"}
        assert discriminative_tokens("what is our Q1 revenue") == {"q1"}


class TestBlocksReuse:
    def test_allows_reordering(self):
        """Set comparison, not sequence: word order is the embedding's job."""
        assert not blocks_reuse("what is our Q1 revenue", "what was our revenue in Q1")

    def test_allows_pure_rewording(self):
        assert not blocks_reuse("how do I reset my password", "how can I reset my password")

    @pytest.mark.parametrize(
        "left,right",
        [
            ("show costs for July 2026", "show costs for June 2026"),
            ("what is the Q1 revenue", "what is the Q3 revenue"),
            ("logs from the last 7 days", "logs from the last 30 days"),
            ("top 10 slowest queries", "top 50 slowest queries"),
            ("what does error code 429 mean", "what does error code 502 mean"),
            ("what is the p50 latency", "what is the p99 latency"),
        ],
    )
    def test_blocks_value_mismatches(self, left, right):
        assert blocks_reuse(left, right)

    def test_blocks_when_one_side_omits_the_qualifier(self):
        """"costs" and "costs for 2026" are different questions."""
        assert blocks_reuse("show costs", "show costs for 2026")


def _middleware(guard: bool):
    config = SemanticCacheConfig(similarity_threshold=0.9, ttl_seconds=60,
                                 discriminative_guard=guard)
    mw = SemanticCacheMiddleware(config=config)
    # One vector for every query: similarity is always 1.0, so the guard is
    # the only thing that can prevent a hit. Without it these tests would be
    # measuring the embedder rather than the veto.
    vec = [0.0] * 384
    vec[0] = 1.0
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=vec)
    mw._embedder = embedder
    return mw, vec


async def _seed(mw, vec, cached_query: str):
    from taut.layers.cache.backends.memory import MemoryCacheBackend

    mw._backend = MemoryCacheBackend(max_size=10, embedding_dim=384)
    await mw._backend.put("default", CacheEntry(
        key="seeded",
        value=LLMResponse(content="June's costs", model="test",
                          usage=TokenUsage(input_tokens=0, output_tokens=0)),
        embedding=vec,
        metadata={"cache_text": cached_query},
    ))


@pytest.mark.asyncio
async def test_guard_turns_a_date_mismatch_into_a_miss():
    pytest.importorskip("faiss")
    mw, vec = _middleware(guard=True)
    await _seed(mw, vec, "show costs for June 2026")

    context = PipelineContext()
    fresh = LLMResponse(content="July's costs", model="test",
                        usage=TokenUsage(input_tokens=0, output_tokens=0))
    next_handler = AsyncMock(return_value=fresh)

    result = await mw.process(
        LLMRequest(intent="show costs for July 2026", namespace="default"), context, next_handler
    )

    assert result.content == "July's costs", "served June's answer for a July question"
    assert context.extra.get("cache_vetoed") is True
    next_handler.assert_called_once()


@pytest.mark.asyncio
async def test_without_the_guard_the_same_lookup_is_a_false_hit():
    """The control. If this passed too, the test above would prove nothing."""
    pytest.importorskip("faiss")
    mw, vec = _middleware(guard=False)
    await _seed(mw, vec, "show costs for June 2026")

    context = PipelineContext()
    next_handler = AsyncMock()
    result = await mw.process(
        LLMRequest(intent="show costs for July 2026", namespace="default"), context, next_handler
    )

    assert result.content == "June's costs"
    assert context.extra.get("cache_status") == "semantic_hit"
    next_handler.assert_not_called()


@pytest.mark.asyncio
async def test_guard_still_allows_a_genuine_paraphrase():
    pytest.importorskip("faiss")
    mw, vec = _middleware(guard=True)
    await _seed(mw, vec, "what was our revenue in Q1")

    context = PipelineContext()
    next_handler = AsyncMock()
    result = await mw.process(
        LLMRequest(intent="what is our Q1 revenue", namespace="default"), context, next_handler
    )

    assert context.extra.get("cache_status") == "semantic_hit"
    assert result.content == "June's costs"  # the seeded value, whatever it says
    next_handler.assert_not_called()


@pytest.mark.asyncio
async def test_entries_written_before_the_guard_existed_still_hit():
    """Upgrade path: no cache_text in metadata means no basis to veto."""
    pytest.importorskip("faiss")
    mw, vec = _middleware(guard=True)
    from taut.layers.cache.backends.memory import MemoryCacheBackend

    mw._backend = MemoryCacheBackend(max_size=10, embedding_dim=384)
    await mw._backend.put("default", CacheEntry(
        key="legacy",
        value=LLMResponse(content="legacy answer", model="test",
                          usage=TokenUsage(input_tokens=0, output_tokens=0)),
        embedding=vec,
    ))

    context = PipelineContext()
    next_handler = AsyncMock()
    result = await mw.process(
        LLMRequest(intent="show costs for July 2026", namespace="default"), context, next_handler
    )

    assert result.content == "legacy answer"
    next_handler.assert_not_called()


@pytest.mark.asyncio
async def test_the_write_path_records_the_query_text():
    """Without this the guard has nothing to compare on the next lookup."""
    pytest.importorskip("faiss")
    mw, vec = _middleware(guard=True)
    from taut.layers.cache.backends.memory import MemoryCacheBackend

    mw._backend = MemoryCacheBackend(max_size=10, embedding_dim=384)
    fresh = LLMResponse(content="fresh", model="test",
                        usage=TokenUsage(input_tokens=0, output_tokens=0))

    request = LLMRequest(intent="show costs for July 2026", namespace="default")
    await mw.process(request, PipelineContext(), AsyncMock(return_value=fresh))

    stored = await mw._backend.get_exact("default", mw._compute_key(request))
    assert stored.metadata["cache_text"] == "show costs for July 2026"
