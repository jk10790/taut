"""Lexical guards on semantic cache hits.

A vector similarity score answers "are these two questions about the same
thing?". It does not reliably answer "do they have the same answer". The
failure is concentrated in high-information tokens: sentence embeddings put
"show costs for July 2026" and "show costs for June 2026" at 0.94 similarity,
because one month token in nine carries almost no weight in the pooled vector
-- and yet it changes the answer completely.

So a semantic hit must clear two bars: the vector threshold, and an exact
match on the tokens whose *value* determines the answer. Numbers, years,
months and quarters are cheap to extract, deterministic, and measured (see
tests/benchmarks/test_cache_precision.py) to block 8 of 25 labelled near-miss
pairs while wrongly blocking 0 of 25 paraphrase pairs.

This guard can only ever turn a hit into a miss. A miss costs money; a false
hit serves a confidently wrong answer to the wrong question, which costs more.
"""
from __future__ import annotations

import re

# Month names and common abbreviations. "may" is deliberately included even
# though it is also a modal verb: a spurious veto costs one cache miss, while
# missing a month mismatch serves the wrong month's data.
MONTHS = frozenset({
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
})

QUARTERS = frozenset({"q1", "q2", "q3", "q4"})

# Tokens keep internal punctuation so that ISO dates (2026-07-01), versions
# (v2.1) and paths survive as single units rather than splitting into digits
# that would compare equal in a set.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9._/-]*")


def discriminative_tokens(text: str) -> frozenset[str]:
    """The tokens in `text` whose exact value changes the answer.

    Any token containing a digit qualifies. That one rule covers years, counts,
    error codes, percentiles (p50/p99), ISO dates, versions and IDs without
    needing a pattern for each. Month and quarter names are added because they
    are the digit-free way of writing a date.
    """
    tokens = _TOKEN.findall(text.lower())
    return frozenset(
        token for token in tokens
        if any(char.isdigit() for char in token)
        or token in MONTHS
        or token in QUARTERS
    )


def blocks_reuse(query: str, cached: str) -> bool:
    """True if `cached`'s answer must not be served for `query`.

    Compares sets, not sequences: "revenue in Q1" and "Q1 revenue" agree, while
    "July 2026" and "June 2026" do not.
    """
    return discriminative_tokens(query) != discriminative_tokens(cached)
