from __future__ import annotations
import asyncio
import logging
import threading
from collections import defaultdict
from typing import Any
from collections.abc import Callable

logger = logging.getLogger("taut.events")

class EventBus:
    """Simple event bus for pipeline lifecycle events."""
    KNOWN_EVENTS = {"cache_hit", "cache_miss", "compression_complete", "model_routed", "request_complete", "error", "stream_complete"}
    
    def __init__(self, strict: bool = False):
        self._listeners: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self.strict = strict
    
    def on(self, event: str, callback: Callable) -> None:
        if self.strict and event not in self.KNOWN_EVENTS:
            logger.warning(f"Registering listener for unknown event: {event}")
        with self._lock:
            self._listeners[event].append(callback)
            
    def off(self, event: str, callback: Callable) -> None:
        with self._lock:
            if event in self._listeners:
                try:
                    self._listeners[event].remove(callback)
                except ValueError:
                    pass
    
    async def emit(self, event: str, **kwargs: Any) -> None:
        if self.strict and event not in self.KNOWN_EVENTS:
            logger.warning(f"Emitting unknown event: {event}")
            
        with self._lock:
            callbacks = list(self._listeners.get(event, []))
            
        for callback in callbacks:
            try:
                result = callback(**kwargs)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Event callback error for '{event}': {e}")
