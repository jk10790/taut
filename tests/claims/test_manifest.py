"""Enforce docs/claims.yaml.

This is the mechanism that keeps documentation honest. It checks, in both
directions:

  * every "shipped" claim names a test that actually exists
  * every test in test_documented_claims.py is registered as a claim
  * every "unimplemented" claim is absent from the documentation prose
  * every number quoted in the README comes from the benchmark baseline

Without this, a future edit can reintroduce a promise the code does not keep --
which is exactly how the README came to advertise cost reporting that returned
$0.00 and a 60% RAG reduction with no implementation behind it.
"""
import pathlib
import re

import pytest

yaml = pytest.importorskip("yaml")

REPO = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = REPO / "docs" / "claims.yaml"
CLAIM_TESTS = REPO / "tests" / "claims" / "test_documented_claims.py"
DOC_FILES = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]


def _load():
    return yaml.safe_load(MANIFEST.read_text())["claims"]


def _test_names_in(path: pathlib.Path) -> set[str]:
    return set(re.findall(r"^(?:async )?def (test_\w+)", path.read_text(), re.MULTILINE))


def test_manifest_is_wellformed():
    claims = _load()
    assert claims, "manifest is empty"
    seen = set()
    for claim in claims:
        for field in ("id", "claim", "source", "test", "status"):
            assert field in claim, f"{claim.get('id', '?')} missing field {field!r}"
        assert claim["id"] not in seen, f"duplicate claim id {claim['id']}"
        seen.add(claim["id"])
        assert claim["status"] in ("shipped", "unimplemented", "unverified"), claim["id"]


def test_every_shipped_claim_names_a_real_test():
    """A claim whose test does not exist is a claim nothing is checking."""
    available = (
        _test_names_in(CLAIM_TESTS)
        | _test_names_in(REPO / "tests" / "claims" / "test_docs_examples.py")
        | _test_names_in(REPO / "tests" / "benchmarks" / "test_fidelity.py")
    )
    missing = []
    for claim in _load():
        if claim["status"] != "shipped":
            continue
        assert claim["test"], f"{claim['id']} is shipped but names no test"
        test_name = claim["test"].split("::")[-1]
        if test_name not in available:
            missing.append(f"{claim['id']} -> {claim['test']}")
    assert not missing, f"claims naming non-existent tests: {missing}"


def test_every_claim_test_is_registered():
    """The reverse direction: no orphan tests drifting out of the manifest."""
    registered = {c["test"].split("::")[-1] for c in _load() if c["test"]}
    orphans = sorted(_test_names_in(CLAIM_TESTS) - registered)
    assert not orphans, (
        f"tests in test_documented_claims.py with no manifest entry: {orphans}. "
        "Add them to docs/claims.yaml or move them to tests/unit/."
    )


def test_unimplemented_claims_are_not_asserted_in_the_docs():
    """The core guard: if we have not built it, we must not claim it."""
    docs_text = "\n".join(p.read_text() for p in DOC_FILES if p.exists())

    # Phrases that would constitute making an unimplemented claim.
    forbidden = {
        "fidelity.compression_preserves_answers": [
            r"without degrading",
            r"answers are unchanged",
            r"preserves the model's answer",
        ],
        "routing.by_health": [
            r"100%\s*CPU",
            r"\bby\s+health\b",
            r"health[- ]based\s+routing",
            r"overwhelmed.*automatically",
        ],
        "compress.rag_60pct": [
            r"up to 60%",
            r"RAG Optimization",
            r"[Cc]ompresses retrieved documents",
        ],
    }

    # Claims that are measured and shipped, but only within a stated scope.
    # The blanket phrasings stay banned regardless of status: fidelity holds
    # for retrieval, code and prose on both recorded models and for every task
    # on claude-haiku-4-5, but gpt-4o-mini misreads columnar aggregations. See
    # docs/limitations.md. Promoting the claim to "shipped" must not silently
    # unlock the slogan the measurement does not support.
    SCOPED = {"fidelity.compression_preserves_answers"}

    unproven = {c["id"] for c in _load() if c["status"] in ("unimplemented", "unverified")}
    violations = []
    for claim_id, patterns in forbidden.items():
        if claim_id not in unproven and claim_id not in SCOPED:
            continue
        for pattern in patterns:
            match = re.search(pattern, docs_text)
            if match:
                violations.append(f"{claim_id}: docs still say {match.group(0)!r}")
    assert not violations, (
        "documentation asserts capabilities that are not proven in "
        f"docs/claims.yaml: {violations}"
    )


def test_readme_numbers_come_from_the_benchmark_baseline():
    """Percentages in the README must match measured results.

    Hand-written numbers are how "40-80% reduction" survived in the docs while
    the benchmark file contained nothing but `pass`.
    """
    import json

    baseline_path = REPO / "tests" / "benchmarks" / "baseline.json"
    if not baseline_path.exists():
        pytest.skip("no baseline yet -- run `pytest tests/benchmarks -m bench`")

    baseline = json.loads(baseline_path.read_text())
    measured = {round(v["reduction_pct"]) for v in baseline["results"].values()}

    readme = (REPO / "README.md").read_text()
    # Only check percentages presented as compression results.
    quoted = {
        int(m) for m in re.findall(r"(\d{1,3})%\s*(?:reduction|smaller|saved)", readme)
    }
    unbacked = {q for q in quoted if q not in measured}
    assert not unbacked, (
        f"README quotes percentages not present in baseline.json: {sorted(unbacked)}. "
        f"Measured values are {sorted(measured)}."
    )


def test_fidelity_status_matches_recorded_cassettes():
    """An unrunnable test must not be quietly forgotten.

    The fidelity suite skips when no cassettes are committed. A skip reads as
    green, so without this check the claim could sit at "unverified" forever --
    or worse, cassettes could land and nobody would promote the claim or the
    docs. This ties the manifest to the facts on disk in both directions.
    """
    from tests.benchmarks.fidelity.runner import MODELS
    from tests.benchmarks.fidelity.cassettes import CassetteStore

    claim = next(
        c for c in _load() if c["id"] == "fidelity.compression_preserves_answers"
    )
    recorded = {m: CassetteStore(m).count() for m in MODELS}
    any_recorded = any(count > 0 for count in recorded.values())

    if any_recorded:
        assert claim["status"] == "shipped", (
            f"cassettes exist ({recorded}) so the fidelity suite now runs. "
            "Promote fidelity.compression_preserves_answers to status: shipped "
            "and state the result in the docs."
        )
    else:
        assert claim["status"] == "unverified", (
            "no cassettes are committed, so the fidelity suite skips. The claim "
            "must stay status: unverified until "
            "`python scripts/record_fidelity.py --record` has been run."
        )
