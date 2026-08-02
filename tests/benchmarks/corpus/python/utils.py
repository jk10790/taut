import json
from typing import Any, Iterable


def chunk(items: Iterable[Any], size: int = 100) -> list[list[Any]]:
    """Split an iterable into fixed-size chunks.

    Args:
        items: Any iterable.
        size: Maximum length of each chunk.

    Returns:
        A list of lists, each at most `size` long.
    """
    out: list[list[Any]] = []
    current: list[Any] = []
    for item in items:
        current.append(item)
        if len(current) >= size:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def safe_load(raw: str) -> dict[str, Any]:
    """Parse JSON, returning an empty dict rather than raising.

    Args:
        raw: The JSON text.

    Returns:
        The parsed object, or {} when the text is not valid JSON.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Callers treat {} as "nothing usable here".
        return {}
