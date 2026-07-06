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
