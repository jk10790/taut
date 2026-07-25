"""Registry for extensible compression plugins."""
from typing import Any
from collections.abc import Callable

class CompressionRegistry:
    """Registry to store user-defined compression plugins based on MIME type or heuristic matching."""
    
    _plugins: list[dict[str, Any]] = []
    
    @classmethod
    def register(cls, mime_type: str, matcher: Callable[[str], bool] | None = None) -> Callable:
        """Decorator to register a custom compressor function.
        
        Args:
            mime_type: The identifier for this compressor.
            matcher: Optional function that takes text and returns True if it matches this type.
            
        Example:
            @CompressionRegistry.register("application/xml", matcher=lambda t: t.strip().startswith("<"))
            def compress_xml(data: str) -> str:
                return optimized_data
        """
        def decorator(func: Callable[[str], str]) -> Callable[[str], str]:
            # Remove existing plugin with same mime_type if it exists
            cls._plugins = [p for p in cls._plugins if p["mime_type"] != mime_type]
            
            cls._plugins.append({
                "mime_type": mime_type,
                "matcher": matcher,
                "func": func
            })
            return func
        return decorator
    
    @classmethod
    def get_compressor(cls, mime_type: str) -> Callable[[str], str] | None:
        """Retrieve a registered compressor by MIME type."""
        for p in cls._plugins:
            if p["mime_type"] == mime_type:
                return p["func"]
        return None
        
    @classmethod
    def detect_and_compress(cls, text: str) -> str | None:
        """Run through registered matchers and apply the first compressor that matches."""
        for p in cls._plugins:
            if p["matcher"] and p["matcher"](text):
                try:
                    return p["func"](text)
                except Exception as e:
                    import logging
                    logging.getLogger("taut.compression").warning(f"Plugin {p['mime_type']} failed: {e}")
        return None

    @classmethod
    def clear(cls) -> None:
        """Clear all registered plugins."""
        cls._plugins.clear()

def register_compressor(mime_type: str, matcher: Callable[[str], bool] | None = None) -> Callable:
    """Convenience alias for CompressionRegistry.register."""
    return CompressionRegistry.register(mime_type, matcher=matcher)
