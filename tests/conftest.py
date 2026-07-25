"""Shared pytest configuration.

Optional-dependency tests skip rather than error, so a contributor with a
minimal install still gets a meaningful green/red signal.
"""
import importlib.util

import pytest


def _missing(*modules: str) -> list[str]:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests whose optional backing dependency is not installed."""
    requirements = {
        "tests/unit/test_redis.py": ("redis",),
        "tests/unit/test_cache.py": ("faiss", "numpy"),
        "tests/unit/test_proxy.py": ("fastapi",),
        "tests/e2e/test_network_e2e.py": ("fastapi", "uvicorn", "httpx"),
    }
    for item in items:
        for path_fragment, modules in requirements.items():
            if path_fragment in str(item.fspath).replace("\\", "/"):
                missing = _missing(*modules)
                if missing:
                    item.add_marker(
                        pytest.mark.skip(reason=f"optional dependency not installed: {', '.join(missing)}")
                    )
        if "tests/e2e/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.e2e)
