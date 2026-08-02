# Fidelity benchmarks

Compression benchmarks that measure only size are dangerous: deleting the whole
payload scores 100%. These tests measure whether compression changes the
**model's answer** — the product brief's actual north star.

## How it works

Compression is deterministic and tasks run at `temperature=0`, so each distinct
prompt only needs the model sampled **once**. Real responses are recorded to
cassettes, committed, and replayed forever.

```
record   real API ──► cassettes/<model>/<hash>.json    needs a key, run rarely
replay   cassettes ──► response                        no key, no network, free
```

Every PR — including PRs from forks — runs the full suite from cassettes. There
is no API key anywhere in CI.

A cassette **miss** is a hard failure, not a skip. If a prompt changes, the
recorded answer no longer applies and the suite says so.

## Recording (maintainers, local only)

Keys stay on your machine. Nothing about recording lives in GitHub Actions.

```bash
# See what it would cost — calls nothing
python scripts/record_fidelity.py --dry-run

# Record (~$0.09 for both providers)
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
python scripts/record_fidelity.py --record

# One provider only
python scripts/record_fidelity.py --record --model gpt-4o-mini

# Re-record everything, e.g. after a model version bump
python scripts/record_fidelity.py --refresh

# Has the provider drifted from what we committed?
python scripts/record_fidelity.py --verify
```

Keys may also go in a local `.env` (gitignored). Use a **restricted key with a
hard spend cap** — the worst case should be bounded by dollars, not trust.

Then review the cassette diff and commit it. Once cassettes exist,
`test_fidelity_status_matches_recorded_cassettes` requires
`fidelity.compression_preserves_answers` in `docs/claims.yaml` to be promoted
from `unverified` to `shipped`, so the result cannot land without the docs
being updated.

## What gets measured

14 tasks across every content type the compressors transform, each asked twice
— once against the raw payload, once compressed:

| Kind | Tasks | Exercises |
|---|---|---|
| `json` | 5 | Columnar transform stays readable to the model |
| `code` | 3 | AST stripping does not change behaviour |
| `prose` | 5 | Filler removal does not change meaning |
| `chat` | 1 | Transcript handling |

Expected answers for structured payloads are **computed from the corpus**, not
hand-written, so they cannot drift if the corpus is regenerated.

Everything except compression is disabled — cache, routing, prefix alignment
and restraint — so any difference in the answer is attributable to the
compressor alone.

## The metric: parity, not accuracy

The gate is `correct(compressed) >= correct(raw)`, per task.

A task the model gets wrong on the *uncompressed* payload is a model
limitation, not a taut defect. Asserting absolute accuracy would make the suite
fail for reasons outside this repo's control, and would tempt someone to weaken
the tasks until it passed. What must hold is that compression never turns a
right answer into a wrong one.

Two supporting gates stop a vacuous pass:

- `test_compression_actually_shrank_the_payload` — if compression saved
  nothing, "no regressions" is trivially true and proves nothing.
- `test_model_answers_most_tasks_correctly_on_raw_payload` — if the model
  cannot answer from the raw payload, the tasks are measuring the model rather
  than taut.

## Security

Cassettes are scrubbed on write: only model name, response content, token usage
and finish reason are stored. Never headers, auth material or organisation
identifiers.

`tests/unit/test_no_secrets_committed.py` scans every tracked file for
key-shaped strings on each CI run, asserts `.env` is gitignored and untracked,
and fails if any workflow ever adopts `pull_request_target` — the trigger that
would expose repository secrets to untrusted fork code.
