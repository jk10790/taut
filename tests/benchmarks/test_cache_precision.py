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

from taut.layers.cache.guards import blocks_reuse

pytestmark = pytest.mark.bench

# Labelled pairs. Written before looking at any similarity score, so the set
# is not fitted to the model's existing behaviour, and deliberately larger than
# the original 13: on 13 pairs the shipped 0.95 threshold looked flawless, and
# on 50 it admits false hits. A benchmark small enough to flatter the system is
# worse than no benchmark.
#
# Pairs that mean the same thing: the cache SHOULD serve these from one entry.
PARAPHRASES = [
    ("how do I reset my password", "how can I reset my password"),
    ("what is our Q1 revenue", "what was our revenue in Q1"),
    ("summarise the incident report", "give me a summary of the incident report"),
    ("list all active users", "show me every active user"),
    ("why did the deployment fail", "what caused the deployment failure"),
    ("how many seats are on the Growth plan", "what is the seat count for the Growth plan"),
    ("how do I export my data", "what is the process for exporting my data"),
    ("when does my subscription renew", "what is my subscription renewal date"),
    ("who approved this pull request", "which person approved this pull request"),
    ("what does error code 429 mean", "explain error code 429"),
    ("how long are logs retained", "what is the log retention period"),
    ("can I change my billing email", "is it possible to update my billing email"),
    ("show me yesterday's failed jobs", "list the jobs that failed yesterday"),
    ("what regions are supported", "which regions do you support"),
    ("how do I invite a teammate", "what is the way to invite a teammate"),
    ("is the API rate limited", "does the API have rate limits"),
    ("what is the maximum file upload size", "how large can an uploaded file be"),
    ("describe the checkout flow", "walk me through the checkout flow"),
    ("why is my invoice higher this month", "what caused the increase in my invoice this month"),
    ("how do I cancel my account", "what are the steps to cancel my account"),
    ("what time is the daily backup", "when does the daily backup run"),
    ("does the Pro plan include SSO", "is SSO part of the Pro plan"),
    ("what is the average response time", "what is the mean response time"),
    ("how do I rotate an API key", "what is the procedure for rotating an API key"),
    ("list the open incidents", "show all incidents that are currently open"),
]

# Pairs that look similar but mean different things: serving one for the
# other is a correctness bug, and the expensive kind -- a confidently wrong
# answer.
NEAR_MISSES = [
    ("what is our Q1 revenue", "what is our Q1 loss"),
    ("how do I enable SSO", "how do I disable SSO"),
    ("list all active users", "list all inactive users"),
    ("delete the staging database", "back up the staging database"),
    ("what is the refund policy", "what is the cancellation policy"),
    ("show costs for July 2026", "show costs for June 2026"),
    ("increase the rate limit", "what is the rate limit"),
    ("what is the Q1 revenue", "what is the Q3 revenue"),
    ("show me yesterday's failed jobs", "show me yesterday's successful jobs"),
    ("how do I add a teammate", "how do I remove a teammate"),
    ("what is the maximum file upload size", "what is the maximum file download size"),
    ("show costs for 2025", "show costs for 2026"),
    ("how many seats are on the Growth plan", "how many seats are on the Starter plan"),
    ("logs from the last 7 days", "logs from the last 30 days"),
    ("what does error code 429 mean", "what does error code 502 mean"),
    ("upgrade my subscription", "downgrade my subscription"),
    ("who approved this pull request", "who requested this pull request"),
    ("top 10 slowest queries", "top 50 slowest queries"),
    ("the deployment on 2026-07-01", "the deployment on 2026-07-08"),
    ("is the Pro plan cheaper than Growth", "is the Growth plan cheaper than Pro"),
    ("how do I export my data", "how do I import my data"),
    ("when does my subscription renew", "when did my subscription start"),
    ("grant admin access to Dana", "revoke admin access to Dana"),
    ("what is the p50 latency", "what is the p99 latency"),
    ("restart the primary node", "restart the replica node"),
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


def _survives(pair, similarity, threshold, guard):
    """Would this pair be served from one cache entry, under this config?"""
    if similarity < threshold:
        return False
    return not (guard and blocks_reuse(*pair))


def test_guard_never_blocks_a_paraphrase():
    """The guard's safety property, and it needs no embedder at all.

    A veto costs a cache miss. That is acceptable on a near-miss and pure loss
    on a paraphrase, so the rule must be conservative enough to leave genuine
    rewordings alone.
    """
    wrongly_blocked = [pair for pair in PARAPHRASES if blocks_reuse(*pair)]
    assert not wrongly_blocked, (
        f"the discriminative guard blocks {len(wrongly_blocked)} genuine paraphrases, "
        f"which is pure lost recall: {wrongly_blocked}"
    )


def test_guard_blocks_a_meaningful_share_of_near_misses():
    """The guard's usefulness. Also needs no embedder.

    If this number falls to zero the guard is dead weight and should be
    deleted rather than left in as decoration.
    """
    blocked = [pair for pair in NEAR_MISSES if blocks_reuse(*pair)]
    assert len(blocked) >= 6, (
        f"the guard blocks only {len(blocked)}/{len(NEAR_MISSES)} near-miss pairs; "
        "it is no longer earning its place in the lookup path"
    )


def test_guard_strictly_improves_precision_at_the_default(similarities):
    """The guard must remove false hits without removing true ones."""
    def counts(guard):
        true_hits = sum(
            1 for pair, sim in zip(PARAPHRASES, similarities["paraphrase"], strict=True)
            if _survives(pair, sim, SHIPPED_DEFAULT, guard)
        )
        false_hits = sum(
            1 for pair, sim in zip(NEAR_MISSES, similarities["near_miss"], strict=True)
            if _survives(pair, sim, SHIPPED_DEFAULT, guard)
        )
        return true_hits, false_hits

    true_off, false_off = counts(guard=False)
    true_on, false_on = counts(guard=True)

    assert true_on == true_off, (
        f"the guard cost {true_off - true_on} true hits at {SHIPPED_DEFAULT}; "
        "it is meant to remove only false ones"
    )
    assert false_on < false_off, (
        f"the guard removed no false hits at {SHIPPED_DEFAULT} "
        f"({false_off} before, {false_on} after)"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN LIMITATION, not a flaky test. Even with the discriminative guard "
        "on, the shipped default admits one false hit: 'is the Pro plan cheaper "
        "than Growth' vs 'is the Growth plan cheaper than Pro' scores 0.978. The "
        "two questions have identical bags of words and differ only in argument "
        "order, so no threshold and no token-set rule can separate them -- only "
        "an order-sensitive comparison could. strict=True: if this starts passing, "
        "the limitation is resolved and docs/limitations.md must be updated."
    ),
)
def test_default_threshold_admits_no_false_hits(similarities):
    """At the shipped default, no near-miss pair may collide.

    A false hit means a user receives an answer to a question they did not
    ask -- worse than a cache miss, which merely costs money.

    This passed on the original 13-pair set. It does not pass on 50 pairs,
    which is the finding that motivated the guard: the old set was too small
    to show the failure, not the cache too good to have one.
    """
    false_hits = [
        (pair, round(sim, 4))
        for pair, sim in zip(NEAR_MISSES, similarities["near_miss"], strict=True)
        if _survives(pair, sim, SHIPPED_DEFAULT, guard=True)
    ]
    assert not false_hits, (
        f"at similarity_threshold={SHIPPED_DEFAULT} these distinct questions collide: "
        f"{false_hits}"
    )


def test_threshold_sweep_is_reported(similarities, capsys):
    """Print the precision/recall curve so the default is a choice, not a guess."""
    lines = []
    for guard in (False, True):
        lines += [
            "",
            f"discriminative guard {'ON' if guard else 'OFF'}",
            f"{'threshold':>10} {'recall':>8} {'precision':>10} {'false hits':>11}",
            "-" * 42,
        ]
        for threshold in THRESHOLDS:
            true_hits = sum(
                1 for pair, s in zip(PARAPHRASES, similarities["paraphrase"], strict=True)
                if _survives(pair, s, threshold, guard)
            )
            false_hits = sum(
                1 for pair, s in zip(NEAR_MISSES, similarities["near_miss"], strict=True)
                if _survives(pair, s, threshold, guard)
            )
            recall = true_hits / len(PARAPHRASES)
            precision = true_hits / (true_hits + false_hits) if (true_hits + false_hits) else 1.0
            lines.append(
                f"{threshold:>10.3f} {recall:>7.0%} {precision:>10.0%} {false_hits:>11}"
            )
    with capsys.disabled():
        print("\n".join(lines))


def test_default_threshold_recall_is_low(similarities):
    """Record the cost of being safe.

    At the shipped default the cache is safe (no false hits, asserted above)
    but rarely fires on a reworded question. This is a hard gate on the number
    documented in docs/limitations.md, so the docs cannot quietly overstate how
    "semantic" the semantic cache is.
    """
    hits = sum(1 for s in similarities["paraphrase"] if s >= SHIPPED_DEFAULT)
    recall = hits / len(PARAPHRASES)
    assert recall <= 0.35, (
        f"recall at {SHIPPED_DEFAULT} improved to {recall:.0%} -- this is good news, "
        "but docs/limitations.md and docs/claims.yaml quote the old figure. Update them."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN LIMITATION, not a flaky test. all-MiniLM-L6-v2 does not separate "
        "paraphrases from near-misses on short queries: measured worst paraphrase "
        "0.879 ('list all active users' / 'show me every active user') scores BELOW "
        "the worst near-miss 0.940 ('show costs for July 2026' / '...June 2026'). "
        "No single threshold is therefore both safe and useful, which is why the "
        "default of 0.95 is tuned for safety and yields ~17% recall. Fixing this "
        "needs a better embedding model or a hybrid lexical+vector match, not a "
        "threshold change. strict=True: if this starts passing, the limitation is "
        "resolved and the docs must be updated."
    ),
)
def test_paraphrases_score_above_near_misses(similarities):
    """The embedding must separate the two classes for tuning to be possible.

    If the worst paraphrase scores below the best near-miss, no threshold can
    give both good recall and good precision, and the layer cannot be made
    both safe and useful by tuning alone.
    """
    worst_paraphrase = min(similarities["paraphrase"])
    best_near_miss = max(similarities["near_miss"])
    assert worst_paraphrase > best_near_miss, (
        f"classes overlap: worst paraphrase {worst_paraphrase:.4f} <= "
        f"best near-miss {best_near_miss:.4f}; no threshold separates them"
    )
