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


def build_tasks() -> list[FidelityTask]:
    events = _events()
    inventory = _inventory()

    five_hundreds = sum(1 for e in events if e["status_code"] == 500)
    top_region = Counter(e["region"] for e in events).most_common(1)[0][0]
    refunds = sum(1 for e in events if e["action"] == "refund")

    archived = sum(1 for i in inventory["items"] if i["status"] == "archived")
    priciest = max(inventory["items"], key=lambda i: i["unit_price"])["sku"]

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
