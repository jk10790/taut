"""Record/replay of provider responses for fidelity testing.

Fidelity testing asks whether compression changes the model's answer. That
question only needs the model sampled *once* per distinct prompt: compression
is deterministic and the tasks run at temperature 0. So we record real
responses once, commit them, and replay them forever.

The payoff is that the full fidelity suite runs on every PR -- including PRs
from forks -- with no API key in scope, no network, and no spend. The key is
only needed when a maintainer deliberately re-records, which happens on their
own machine (see scripts/record_fidelity.py).

A cassette *miss* in replay mode is a hard error, never a silent pass: if the
prompt changed, the recorded answer no longer applies and the suite says so.

Security note: cassettes are scrubbed on write. Only the model name, response
content, token usage and finish reason are persisted -- never request headers,
auth material or organisation identifiers. tests/unit/test_no_secrets_committed.py
scans every tracked file for key-shaped strings on each CI run.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from taut.core.models import LLMRequest, LLMResponse, PipelineContext, TokenUsage
from taut.providers.base import BaseProvider

CASSETTE_ROOT = pathlib.Path(__file__).parent / "cassettes"

# How much of the prompt to keep alongside each cassette. Enough to review a
# diff and know what was asked; not so much that the corpus is duplicated into
# the cassette store.
PREVIEW_CHARS = 400


class CassetteMiss(RuntimeError):
    """Raised in replay mode when no recording matches the request."""


def _slug(model: str) -> str:
    """Filesystem-safe directory name for a model id."""
    return model.replace("/", "__").replace(":", "_")


def request_fingerprint(request: LLMRequest) -> dict[str, Any]:
    """The provider-visible parts of a request, in canonical form.

    Anything that can change the model's answer belongs here. Anything that
    cannot -- namespaces, metadata, taut-internal bookkeeping -- must not,
    or cassettes would miss for irrelevant reasons.
    """
    messages: list[dict[str, Any]] = []
    for msg in request.messages or []:
        content = msg.content
        if isinstance(content, list):
            content = json.dumps(content, sort_keys=True)
        messages.append({"role": msg.role, "content": content})

    return {
        "model": request.model,
        "system_prompt": request.system_prompt,
        "messages": messages,
        "context": request.context if isinstance(request.context, str) else None,
        "intent": request.intent,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
        "stop": request.stop,
        "response_format": request.response_format,
        "tools": request.tools,
    }


def request_key(request: LLMRequest) -> str:
    canonical = json.dumps(request_fingerprint(request), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _scrub(response: LLMResponse) -> dict[str, Any]:
    """Persist only what a replay needs. Never headers or auth material."""
    return {
        "content": response.content,
        "model": response.model,
        "finish_reason": response.finish_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


class CassetteStore:
    """Reads and writes cassettes for one model."""

    def __init__(self, model: str, root: pathlib.Path | None = None):
        self.model = model
        self.dir = (root or CASSETTE_ROOT) / _slug(model)

    def path_for(self, key: str) -> pathlib.Path:
        return self.dir / f"{key}.json"

    def exists(self, key: str) -> bool:
        return self.path_for(key).exists()

    def load(self, key: str) -> LLMResponse:
        data = json.loads(self.path_for(key).read_text())
        recorded = data["response"]
        return LLMResponse(
            content=recorded["content"],
            model=recorded["model"],
            finish_reason=recorded.get("finish_reason"),
            usage=TokenUsage(
                input_tokens=recorded["usage"]["input_tokens"],
                output_tokens=recorded["usage"]["output_tokens"],
            ),
        )

    def save(self, key: str, request: LLMRequest, response: LLMResponse, label: str = "") -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fingerprint = request_fingerprint(request)
        prompt_text = "\n".join(
            part for part in [fingerprint["system_prompt"] or "", fingerprint["intent"] or ""] if part
        )
        payload = {
            "note": "Recorded by scripts/record_fidelity.py. Do not hand-edit.",
            "label": label,
            "request_key": key,
            "request_model": fingerprint["model"],
            "request_preview": prompt_text[:PREVIEW_CHARS],
            "request_message_count": len(fingerprint["messages"]),
            "response": _scrub(response),
        }
        self.path_for(key).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def count(self) -> int:
        return len(list(self.dir.glob("*.json"))) if self.dir.exists() else 0


class CassetteProvider(BaseProvider):
    """Provider that replays recorded responses, or records new ones.

    Wraps a real provider only in record mode; in replay mode `inner` is never
    touched, so no credentials need to exist at all.
    """

    def __init__(
        self,
        model: str,
        mode: str = "replay",
        inner: BaseProvider | None = None,
        root: pathlib.Path | None = None,
        label: str = "",
    ):
        if mode not in ("replay", "record", "refresh"):
            raise ValueError(f"unknown cassette mode {mode!r}")
        if mode in ("record", "refresh") and inner is None:
            raise ValueError(f"{mode} mode needs a real provider to record from")
        self.model = model
        self.mode = mode
        self.inner = inner
        self.label = label
        self.store = CassetteStore(model, root=root)
        self.hits = 0
        self.recordings = 0

    async def complete(self, request: LLMRequest, context: PipelineContext) -> LLMResponse:
        key = request_key(request)

        if self.mode != "refresh" and self.store.exists(key):
            self.hits += 1
            return self.store.load(key)

        if self.mode == "replay":
            raise CassetteMiss(
                f"No cassette for {self.model} (key {key}).\n"
                f"  label: {self.label or '<unlabelled>'}\n"
                "The prompt changed, or this task is new. Re-record with:\n"
                "  python scripts/record_fidelity.py --record\n"
                "Recording needs an API key and runs on your machine only."
            )

        response = await self.inner.complete(request, context)
        self.store.save(key, request, response, label=self.label)
        self.recordings += 1
        return response

    async def complete_stream(self, request: LLMRequest, context: PipelineContext):
        raise NotImplementedError("fidelity tests do not stream")
        yield ""  # pragma: no cover - makes this an async generator
