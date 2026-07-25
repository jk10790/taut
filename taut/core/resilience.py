"""Resilience primitives for taut."""
import asyncio
import random
import time
import logging
from functools import wraps

logger = logging.getLogger("taut.resilience")

class CapacityExceededError(Exception):
    """Raised when rate limits or backpressure thresholds are exceeded."""
    pass

class ProviderBusyException(Exception):
    """Raised when the upstream provider is overwhelmed (e.g. 429 or 503)."""
    pass

class FallbackExhaustedError(Exception):
    """Raised when all fallbacks have been attempted and failed."""
    pass

class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open and the call was not attempted."""
    pass


class CircuitBreaker:
    """Per-model circuit breaker.

    A model that is failing repeatedly is skipped for `reset_timeout` seconds
    rather than retried on every request, so a dead upstream costs one probe
    per window instead of a full retry budget on each call.

    States: closed (normal) -> open (after `failure_threshold` consecutive
    failures) -> half-open (one probe allowed after `reset_timeout`) -> closed
    on success, or back to open on failure.
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}
        self._half_open: set[str] = set()
        self._lock = asyncio.Lock()

    async def allows(self, key: str) -> bool:
        """True if a call to `key` may be attempted now."""
        async with self._lock:
            opened = self._opened_at.get(key)
            if opened is None:
                return True
            if (time.monotonic() - opened) >= self.reset_timeout:
                # Let exactly one probe through.
                self._half_open.add(key)
                del self._opened_at[key]
                logger.info("Circuit for %s entering half-open", key)
                return True
            return False

    async def record_success(self, key: str) -> None:
        async with self._lock:
            if self._failures.pop(key, None) or key in self._half_open:
                logger.info("Circuit for %s closed", key)
            self._opened_at.pop(key, None)
            self._half_open.discard(key)

    async def record_failure(self, key: str) -> None:
        async with self._lock:
            if key in self._half_open:
                # Probe failed: straight back to open.
                self._half_open.discard(key)
                self._opened_at[key] = time.monotonic()
                logger.warning("Circuit for %s re-opened after failed probe", key)
                return
            count = self._failures.get(key, 0) + 1
            self._failures[key] = count
            if count >= self.failure_threshold:
                self._opened_at[key] = time.monotonic()
                logger.warning(
                    "Circuit for %s opened after %d consecutive failures", key, count
                )

    def state(self, key: str) -> str:
        """Return 'closed', 'open', or 'half_open' for observability."""
        if key in self._half_open:
            return "half_open"
        if key in self._opened_at:
            return "open"
        return "closed"


class RateLimiter:
    """Lightweight in-memory Token Bucket rate limiter."""
    
    def __init__(self, tokens_per_second: float, capacity: float):
        self.rate = tokens_per_second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
        
    async def acquire(self, tokens: float = 1.0, timeout: float | None = None):
        """Acquire tokens from the bucket, waiting if necessary."""
        start_time = time.monotonic()
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
            
            if timeout is not None and (time.monotonic() - start_time) > timeout:
                raise CapacityExceededError("Rate limiter timeout exceeded.")
                
            await asyncio.sleep(0.1)

def with_retries(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0, exceptions=(Exception,)):
    """Decorator for exponential backoff with jitter."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded.")
                        raise
                    
                    delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = delay + jitter
                    
                    logger.warning(f"Request failed with {e}. Retrying in {sleep_time:.2f}s (Attempt {retries}/{max_retries})")
                    await asyncio.sleep(sleep_time)
        return wrapper
    return decorator
