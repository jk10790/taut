"""Fail the build if anything key-shaped is committed.

The fidelity workflow keeps API keys on the maintainer's machine and never in
GitHub, but "never commit the key" is a rule that needs enforcing rather than
remembering -- particularly in a public repository, where a leaked key is
scraped within minutes.

This scans every tracked file on every CI run.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# Provider key formats, plus a generic assignment of a long opaque value.
SECRET_PATTERNS = {
    "OpenAI API key": re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    "Anthropic API key": re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}"),
    "AWS access key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "private key block": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "hardcoded credential": re.compile(
        r"""(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*"""
        r"""["'][A-Za-z0-9+/_-]{24,}["']"""
    ),
}

# Files that legitimately describe key formats.
ALLOWLIST = {
    "tests/unit/test_no_secrets_committed.py",
}


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line]


def test_no_secret_shaped_strings_in_tracked_files():
    findings = []
    for relative in _tracked_files():
        if relative in ALLOWLIST:
            continue
        path = REPO / relative
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary or vanished; nothing text-shaped to leak
        for label, pattern in SECRET_PATTERNS.items():
            match = pattern.search(content)
            if match:
                # Report the location, never the value.
                line = content[: match.start()].count("\n") + 1
                findings.append(f"{relative}:{line} looks like a {label}")
    assert not findings, "possible secrets committed:\n  " + "\n  ".join(findings)


def test_dotenv_is_not_tracked():
    """A local .env is how keys are supplied for recording. It must stay local."""
    tracked = _tracked_files()
    leaked = [f for f in tracked if pathlib.PurePath(f).name in (".env", ".env.local")]
    assert not leaked, f"environment files must not be committed: {leaked}"


def test_dotenv_is_gitignored():
    ignore = (REPO / ".gitignore").read_text()
    assert re.search(r"^\.env$", ignore, re.MULTILINE), ".gitignore must exclude .env"


def test_no_recording_workflow_exposes_secrets():
    """Guard the public-repo footgun.

    `pull_request_target` runs with repository secrets available while checking
    out the pull request's code, so any fork could exfiltrate them. Fidelity
    recording is deliberately local-only, and no workflow here should ever need
    that trigger.
    """
    workflows = list((REPO / ".github" / "workflows").glob("*.yml"))
    assert workflows, "no workflows found -- has CI moved?"
    for workflow in workflows:
        text = workflow.read_text()
        assert "pull_request_target" not in text, (
            f"{workflow.name} uses pull_request_target, which exposes secrets to "
            "untrusted fork code"
        )


@pytest.mark.parametrize(
    "sample",
    [
        "sk-" + "a" * 32,
        "sk-ant-" + "b" * 40,
        "AKIA" + "A" * 16,
        'api_key = "' + "c" * 30 + '"',
    ],
)
def test_patterns_actually_detect_secrets(sample):
    """Negative control: a scanner that matches nothing would pass silently."""
    assert any(p.search(sample) for p in SECRET_PATTERNS.values()), (
        f"scanner failed to flag {sample[:12]}..."
    )


@pytest.mark.parametrize(
    "sample",
    [
        'api_key="your-api-key"',
        "export OPENAI_API_KEY=sk-...",
        'base_url="http://localhost:8000/v1"',
        "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}",
    ],
)
def test_patterns_do_not_flag_documentation(sample):
    """A scanner that cries wolf gets disabled, which is worse than no scanner."""
    hits = [name for name, p in SECRET_PATTERNS.items() if p.search(sample)]
    assert not hits, f"false positive ({hits}) on documentation sample: {sample}"
