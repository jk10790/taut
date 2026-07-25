"""Semantic cache precision.

Hit rate alone is a dangerous metric for a semantic cache: a threshold of 0.0
gives a 100% hit rate and serves every user someone else's answer. The number
that matters is precision -- of the requests we served from cache, how many
were genuinely the same question.

This sweep measures both against labelled pairs, so `similarity_threshold` can
be chosen from data rather than guessed. The shipped default is 0.95.

Requires the real ONNX embedder (downloads a ~90MB model on first run), so it
skips when the model cannot be fetched.
"""
import pytest

pytestmark = pytest.mark.bench

# Pairs that mean the same thing: the cache SHOULD serve these from one entry.
PARAPHRASES = [
    ("how do I reset my password", "how can I reset my password"),
    ("what is our Q1 revenue", "what was our revenue in Q1"),
    ("summarise the incident report", "give me a summary of the incident report"),
    ("list all active users", "show me every active user"),
    ("why did the deployment fail", "what caused the deployment failure"),
    ("how many seats are on the Growth plan", "what is the seat count for the Growth plan"),
]

# Pairs that look similar but mean different things: serving one for the other
# is a correctness bug, and the expensive kind -- a confidently wrong answer.
NEAR_MISSES = [
    ("what is our Q1 revenue", "what is our Q1 loss"),
    ("how do I enable SSO", "how do I disable SSO"),
    ("list all active users", "list all inactive users"),
    ("delete the staging database", "back up the staging database"),
    ("what is the refund policy", "what is the cancellation policy"),
    ("show costs for July 2026", "show costs for June 2026"),
    ("increase the rate limit", "what is the rate limit"),
]

THRESHOLDS = [0.80, 0.85, 0.90, 0.925, 0.95, 0.97, 0.99]
SHIPPED_DEFAULT = 0.95


@pytest.fixture(scope="module")
def embedder():
    from taut.layers.cache.embedder import ONNXEmbedder

    emb = ONNXEmbedder()
    try:
        emb._get_model()
    except Exception as exc:  # offline, or HF unreachable
        pytest.skip(f"real embedder unavailable ({type(exc).__name__}: {exc})")
    return emb


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def similarities(embedder):
    import asyncio

    async def embed_all(pairs):
        out = []
        for left, right in pairs:
            vl = await embedder.embed(left)
            vr = await embedder.embed(right)
            out.append(_cosine(vl, vr))
        return out

    return {
        "paraphrase": asyncio.run(embed_all(PARAPHRASES)),
        "near_miss": asyncio.run(embed_all(NEAR_MISSES)),
    }


def test_default_threshold_admits_no_false_hits(similarities, request):
    """At the shipped default, no near-miss pair may collide.

    A false hit means a user receives an answer to a question they did not
    ask -- worse than a cache miss, which merely costs money.
    """
    false_hits = [
        (pair, round(sim, 4))
        for pair, sim in zip(NEAR_MISSES, similarities["near_miss"], strict=True)
        if sim >= SHIPPED_DEFAULT
    ]
    assert not false_hits, (
        f"at similarity_threshold={SHIPPED_DEFAULT} these distinct questions collide: "
        f"{false_hits}"
    )


def test_threshold_sweep_is_reported(similarities, capsys):
    """Print the precision/recall curve so the default is a choice, not a guess."""
    lines = [
        "",
        f"{'threshold':>10} {'recall':>8} {'precision':>10} {'false hits':>11}",
        "-" * 42,
    ]
    for threshold in THRESHOLDS:
        true_hits = sum(1 for s in similarities["paraphrase"] if s >= threshold)
        false_hits = sum(1 for s in similarities["near_miss"] if s >= threshold)
        recall = true_hits / len(PARAPHRASES)
        precision = true_hits / (true_hits + false_hits) if (true_hits + false_hits) else 1.0
        lines.append(
            f"{threshold:>10.3f} {recall:>7.0%} {precision:>10.0%} {false_hits:>11}"
        )
    with capsys.disabled():
        print("\n".join(lines))


def test_paraphrases_score_above_near_misses(similarities):
    """The embedding must separate the two classes at all.

    If the worst paraphrase scores below the best near-miss, no threshold can
    give both good recall and good precision, and the layer cannot be made
    safe by tuning.
    """
    worst_paraphrase = min(similarities["paraphrase"])
    best_near_miss = max(similarities["near_miss"])
    assert worst_paraphrase > best_near_miss, (
        f"classes overlap: worst paraphrase {worst_paraphrase:.4f} <= "
        f"best near-miss {best_near_miss:.4f}; no threshold separates them"
    )
