"""Execute and validate the code in README.md, docs/ and examples/.

The documented SDK example used to raise AttributeError on four symbols and
ValidationError on a required field -- nobody noticed because no test ever ran
it. These tests make the documentation executable.
"""
import ast
import pathlib
import re
import textwrap
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
DOC_FILES = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _python_blocks() -> list[tuple[str, str]]:
    blocks = []
    for path in DOC_FILES:
        if not path.exists():
            continue
        for i, code in enumerate(PYTHON_BLOCK.findall(path.read_text())):
            blocks.append((f"{path.relative_to(REPO)}#block{i}", textwrap.dedent(code)))
    return blocks


def test_docs_contain_python_examples():
    assert _python_blocks(), "no python examples found -- the checks below would be vacuous"


@pytest.mark.parametrize("label,code", _python_blocks(), ids=lambda v: v if isinstance(v, str) else "")
def test_documented_python_blocks_parse(label, code):
    """Every documented snippet must at least be valid Python."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        pytest.fail(f"{label} is not valid Python: {exc}")


def test_documented_symbols_all_exist():
    """Every `taut.X` and `from taut import X` in the docs must resolve.

    This is the check that would have caught ContextBlock, QueryBlock,
    SemanticCacheConfig and CompressionConfig being documented but not
    exported.
    """
    import taut

    referenced: dict[str, set[str]] = {}
    sources = list(_python_blocks())
    for example in sorted((REPO / "examples").glob("*.py")):
        sources.append((str(example.relative_to(REPO)), example.read_text()))

    for label, code in sources:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue  # reported by test_documented_python_blocks_parse
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "taut":
                for alias in node.names:
                    referenced.setdefault(alias.name, set()).add(label)
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "taut"
            ):
                referenced.setdefault(node.attr, set()).add(label)

    missing = {
        name: sorted(where)
        for name, where in referenced.items()
        if not hasattr(taut, name)
    }
    assert not missing, f"documented symbols that do not exist on `taut`: {missing}"


def test_documented_install_extras_are_sufficient():
    """The install command in the README must cover the command that follows it.

    The README said `pip install "taut[cache]"` and then started the proxy --
    but the proxy needs FastAPI and uvicorn, which live in the `proxy` extra.
    """
    import tomllib

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    readme = (REPO / "README.md").read_text()

    # Matches both `pip install "taut[a,b]"` and `pip install -e ".[a,b]"`.
    install_re = re.compile(r'pip install\s+(?:-e\s+)?"?(?:taut|\.)\[([^\]]+)\]"?')
    installed: set[str] = set()
    for block in BASH_BLOCK.findall(readme):
        for match in install_re.findall(block):
            installed.update(part.strip() for part in match.split(","))

    starts_proxy = any(
        "taut.proxy.server" in block or "taut serve" in block
        for block in BASH_BLOCK.findall(readme)
    )
    if not starts_proxy:
        pytest.skip("README does not start the proxy")

    assert installed, "README starts the proxy but documents no install command"
    resolved: set[str] = set()
    for extra in installed:
        resolved.add(extra)
        for dep in extras.get(extra, []):
            for nested in re.findall(r"taut\[([^\]]+)\]", dep):
                resolved.update(p.strip() for p in nested.split(","))

    assert "proxy" in resolved, (
        f"README starts the proxy after installing taut{sorted(installed)}, "
        f"which does not provide FastAPI/uvicorn. Needs the 'proxy' extra."
    )


def test_quickstart_example_runs():
    """examples/quickstart.py must run to completion against a stub provider."""
    import taut
    from taut.core.models import LLMResponse, TokenUsage
    from taut.providers.base import BaseProvider

    class _Stub(BaseProvider):
        async def complete(self, request, context):
            return LLMResponse(
                content="Two items are active.",
                model=request.model or "stub",
                usage=TokenUsage(input_tokens=200, output_tokens=20),
            )

        async def complete_stream(self, request, context):
            yield "ok"

    class _Embedder:
        def embed(self, text):
            return [1.0] + [0.0] * 383

        def dimension(self):
            return 384

    real_create = taut.create_pipeline

    def _patched(config=None, **kwargs):
        pipeline = real_create(config, **kwargs)
        pipeline._provider = _Stub()
        for mw in pipeline._middlewares:
            if mw.name == "semantic_cache":
                mw._embedder = _Embedder()
        return pipeline

    script = (REPO / "examples" / "quickstart.py").read_text()
    module = types.ModuleType("quickstart_under_test")
    module.__file__ = str(REPO / "examples" / "quickstart.py")
    module.__name__ = "__main__"

    import os

    prior_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "stub-key-for-tests"
    taut.create_pipeline = _patched
    try:
        exec(compile(script, module.__file__, "exec"), module.__dict__)
    finally:
        taut.create_pipeline = real_create
        if prior_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = prior_key
