"""Fidelity task set: questions with known-correct answers.

Each task pairs a slice of the benchmark corpus with a question whose answer is
checkable in code. Where the corpus is structured, the expected answer is
*computed from the corpus* rather than hand-written, so it cannot drift out of
date if the corpus is regenerated.

Contexts are deliberately sliced. The full api_events.json is ~46k tokens; at
two calls per task per provider it would dominate recording cost for no extra
signal. Slices keep a full re-record in the low tens of cents.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from dataclasses import dataclass
from collections.abc import Callable

CORPUS = pathlib.Path(__file__).resolve().parents[1] / "corpus"

# Rows kept from each structured file. Changing these changes every prompt and
# therefore invalidates every cassette -- re-record after editing.
EVENT_ROWS = 100
INVENTORY_ITEMS = 80

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "twenty": 20, "twenty-five": 25,
    "thirty": 30, "thirty-five": 35, "sixty": 60,
}


# --------------------------------------------------------------------------
# Graders. All programmatic -- grading never needs an API key.
# --------------------------------------------------------------------------


def numeric(expected: int | float) -> Callable[[str], bool]:
    """True if the answer states `expected` as a number (or a number word)."""

    def grade(answer: str) -> bool:
        text = answer.lower().replace(",", "")
        for found in re.findall(r"-?\d+(?:\.\d+)?", text):
            value = float(found)
            if value == float(expected):
                return True
        for word, value in _WORD_NUMBERS.items():
            if value == expected and re.search(rf"\b{re.escape(word)}\b", text):
                return True
        return False

    grade.__doc__ = f"numeric == {expected}"
    return grade


def contains_any(*needles: str) -> Callable[[str], bool]:
    def grade(answer: str) -> bool:
        text = answer.lower()
        return any(needle.lower() in text for needle in needles)

    grade.__doc__ = f"contains any of {needles}"
    return grade


def yes_no(expected: bool) -> Callable[[str], bool]:
    """True if the answer affirms/denies as asked.

    A plain substring check is unsafe here: "no" appears inside "now", "not"
    and "cannot", so a wrong answer like "Yes, it is now available" would score
    as correct. Match on word boundaries and require the opposite word to be
    absent.
    """
    def grade(answer: str) -> bool:
        text = answer.lower()
        said_yes = bool(re.search(r"\byes\b", text))
        said_no = bool(re.search(r"\b(no|not)\b", text))
        return (said_yes and not said_no) if expected else (said_no and not said_yes)

    grade.__doc__ = f"answers {'yes' if expected else 'no'}"
    return grade


def only_of(expected: str, *alternatives: str) -> Callable[[str], bool]:
    """True if the answer names `expected` and none of the other valid values.

    Needed for lookup tasks over a categorical field: plain substring matching
    would score "not archived, it is active" as correct because "archived" is
    present. The alternatives are the field's other values in this corpus.
    """

    def grade(answer: str) -> bool:
        text = answer.lower()
        return expected.lower() in text and not any(alt.lower() in text for alt in alternatives)

    grade.__doc__ = f"names {expected!r} and none of {alternatives}"
    return grade


def exact(expected: str) -> Callable[[str], bool]:
    def grade(answer: str) -> bool:
        return answer.strip().strip(".").lower() == expected.strip().lower()

    grade.__doc__ = f"exact == {expected!r}"
    return grade


# --------------------------------------------------------------------------
# Corpus slices
# --------------------------------------------------------------------------


def _events() -> list[dict]:
    data = json.loads((CORPUS / "json_logs" / "api_events.json").read_text())
    return data[:EVENT_ROWS]


def _inventory() -> dict:
    data = json.loads((CORPUS / "json_logs" / "inventory.json").read_text())
    return {**data, "items": data["items"][:INVENTORY_ITEMS]}


def events_context() -> str:
    return json.dumps(_events(), indent=2)


def inventory_context() -> str:
    return json.dumps(_inventory(), indent=2)


def _read(*parts: str) -> str:
    return (CORPUS.joinpath(*parts)).read_text()


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FidelityTask:
    id: str
    context: str
    question: str
    grader: Callable[[str], bool]
    kind: str  # which compressor this exercises
    probe: str = ""  # experiment arm, for the JSON tasks -- see PROBES


# Arms of the aggregation experiment. `inventory.priciest_sku` regressed under
# compression with only a 0.2% gap between the right answer and the runner-up,
# which is consistent with two different stories:
#
#   knife-edge noise -- the model cannot reliably separate near-ties, and the
#     columnar payload is merely a different roll of the same dice; or
#   format damage    -- the columnar layout systematically hurts scanning a
#     column for an extremum.
#
# These arms separate them. `narrow` gaps are <2%, `wide` gaps are >35%, and
# `lookup` reads a single named row so it exercises retrieval without any
# scan-and-compare. If only `narrow` regresses, it is noise. If `wide`
# regresses too, or `lookup` stays clean while both aggregation arms break,
# the format is at fault.
PROBES = ("narrow", "wide", "lookup")


def build_tasks() -> list[FidelityTask]:
    events = _events()
    inventory = _inventory()

    five_hundreds = sum(1 for e in events if e["status_code"] == 500)
    top_region = Counter(e["region"] for e in events).most_common(1)[0][0]
    refunds = sum(1 for e in events if e["action"] == "refund")

    archived = sum(1 for i in inventory["items"] if i["status"] == "archived")
    items = inventory["items"]

    # Aggregation targets, all computed. Gaps in this corpus slice:
    #   priciest   499.60 vs 498.50   0.2%  narrow
    #   biggest_qty     96 vs 95      1.0%  narrow
    #   slowest     887.04 vs 880.55  0.7%  narrow
    #   cheapest     12.42 vs 20.60  39.7%  wide
    #   fastest       10.60 vs 16.52 35.8%  wide
    priciest = max(items, key=lambda i: i["unit_price"])["sku"]
    cheapest = min(items, key=lambda i: i["unit_price"])["sku"]
    biggest_qty = max(items, key=lambda i: i["qty"])["sku"]
    slowest_event = max(events, key=lambda e: e["latency_ms"])["event_id"]
    fastest_event = min(events, key=lambda e: e["latency_ms"])["event_id"]

    # Retrieval controls. Mid-corpus rows, so neither is findable by guessing
    # the first or last record.
    probe_item = next(i for i in items if i["sku"] == "SKU-00042")
    probe_event = next(e for e in events if e["event_id"] == 57)
    statuses = sorted({i["status"] for i in items})
    regions = sorted({e["region"] for e in events})

    brief = "Answer using only the provided context. Reply with the answer alone and nothing else."

    return [
        # ---- JSON: the columnar transform must stay readable to the model ----
        FidelityTask(
            id="events.count_500",
            context=events_context(),
            question=f"{brief} How many events have status_code 500?",
            grader=numeric(five_hundreds),
            kind="json",
        ),
        FidelityTask(
            id="events.top_region",
            context=events_context(),
            question=f"{brief} Which region appears most often?",
            grader=contains_any(top_region),
            kind="json",
        ),
        FidelityTask(
            id="events.count_refunds",
            context=events_context(),
            question=f'{brief} How many events have the action "refund"?',
            grader=numeric(refunds),
            kind="json",
        ),
        FidelityTask(
            id="inventory.count_archived",
            context=inventory_context(),
            question=f'{brief} How many items have status "archived"?',
            grader=numeric(archived),
            kind="json",
        ),
        FidelityTask(
            id="inventory.priciest_sku",
            context=inventory_context(),
            question=f"{brief} Which SKU has the highest unit_price?",
            grader=contains_any(priciest),
            kind="json",
            probe="narrow",
        ),
        # ---- Aggregation experiment: narrow gaps ----
        FidelityTask(
            id="inventory.biggest_qty_sku",
            context=inventory_context(),
            question=f"{brief} Which SKU has the highest qty?",
            grader=contains_any(biggest_qty),
            kind="json",
            probe="narrow",
        ),
        FidelityTask(
            id="events.slowest_event",
            context=events_context(),
            question=f"{brief} Which event_id has the highest latency_ms?",
            grader=numeric(slowest_event),
            kind="json",
            probe="narrow",
        ),
        # ---- Aggregation experiment: wide gaps ----
        FidelityTask(
            id="inventory.cheapest_sku",
            context=inventory_context(),
            question=f"{brief} Which SKU has the lowest unit_price?",
            grader=contains_any(cheapest),
            kind="json",
            probe="wide",
        ),
        FidelityTask(
            id="events.fastest_event",
            context=events_context(),
            question=f"{brief} Which event_id has the lowest latency_ms?",
            grader=numeric(fastest_event),
            kind="json",
            probe="wide",
        ),
        # ---- Retrieval controls: one named row, no scan-and-compare ----
        FidelityTask(
            id="inventory.lookup_status",
            context=inventory_context(),
            question=f'{brief} What is the status of {probe_item["sku"]}?',
            grader=only_of(
                probe_item["status"],
                *[s for s in statuses if s != probe_item["status"]],
            ),
            kind="json",
            probe="lookup",
        ),
        FidelityTask(
            id="inventory.lookup_price",
            context=inventory_context(),
            question=f'{brief} What is the unit_price of {probe_item["sku"]}?',
            grader=numeric(probe_item["unit_price"]),
            kind="json",
            probe="lookup",
        ),
        FidelityTask(
            id="events.lookup_region",
            context=events_context(),
            question=f'{brief} Which region is event_id {probe_event["event_id"]} in?',
            grader=only_of(
                probe_event["region"],
                *[r for r in regions if r != probe_event["region"]],
            ),
            kind="json",
            probe="lookup",
        ),
        FidelityTask(
            id="events.lookup_latency",
            context=events_context(),
            question=f'{brief} What is the latency_ms of event_id {probe_event["event_id"]}?',
            grader=numeric(probe_event["latency_ms"]),
            kind="json",
            probe="lookup",
        ),
        # ---- Python: stripping docstrings must not change behaviour ----
        FidelityTask(
            id="code.raises_on_short_seat",
            context=_read("python", "service.py"),
            question=(
                f"{brief} What exception class is raised when the seat identifier "
                "is too short?"
            ),
            grader=contains_any("ValueError"),
            kind="code",
        ),
        FidelityTask(
            id="code.default_retries",
            context=_read("python", "service.py"),
            question=f"{brief} What is the default value of max_retries?",
            grader=numeric(3),
            kind="code",
        ),
        FidelityTask(
            id="code.safe_load_failure",
            context=_read("python", "utils.py"),
            question=(
                f"{brief} What does safe_load return when the text is not valid JSON?"
            ),
            grader=contains_any("{}", "empty dict", "empty dictionary"),
            kind="code",
        ),
        # ---- Prose: filler removal must not change meaning ----
        FidelityTask(
            id="faq.starter_seats",
            context=_read("rag", "product_faq.md"),
            question=f"{brief} How many seats does the Starter plan include?",
            grader=numeric(5),
            kind="prose",
        ),
        FidelityTask(
            id="faq.scim_on_growth",
            context=_read("rag", "product_faq.md"),
            question=f"{brief} Is SCIM provisioning available on the Growth plan? Yes or no.",
            grader=yes_no(False),
            kind="prose",
        ),
        FidelityTask(
            id="faq.backup_purge_days",
            context=_read("rag", "product_faq.md"),
            question=(
                f"{brief} Within how many days are deleted records purged from backups?"
            ),
            grader=numeric(35),
            kind="prose",
        ),
        FidelityTask(
            id="postmortem.pool_size",
            context=_read("rag", "incident_postmortem.md"),
            question=f"{brief} What was the maximum connection pool size reduced to?",
            grader=numeric(60),
            kind="prose",
        ),
        FidelityTask(
            id="postmortem.rollback_time",
            context=_read("rag", "incident_postmortem.md"),
            question=f"{brief} At what time was the previous configuration reapplied?",
            grader=contains_any("09:44"),
            kind="prose",
        ),
        # ---- Chat transcript ----
        FidelityTask(
            id="chat.export_row_cap",
            context=_read("chat", "support_thread.md"),
            question=f"{brief} What is the row cap on a single asynchronous export?",
            # The source says "two million"; accept any faithful rendering.
            grader=contains_any("2,000,000", "2000000", "2 million", "two million"),
            kind="chat",
        ),
    ]
