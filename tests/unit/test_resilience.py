import asyncio
import time

import pytest

from taut.core.config import ResilienceConfig, TautConfig
from taut.core.models import LLMRequest, LLMResponse, TokenUsage
from taut.core.pipeline import create_pipeline
from taut.core.resilience import (
    CapacityExceededError,
    CircuitBreaker,
    FallbackExhaustedError,
    ProviderBusyException,
    RateLimiter,
    with_retries,
)
from taut.providers.base import BaseProvider

@pytest.mark.asyncio
async def test_rate_limiter():
    limiter = RateLimiter(tokens_per_second=10, capacity=5)
    
    # Should acquire 5 tokens immediately
    acquired = await limiter.acquire(5)
    assert acquired is True
    assert limiter.tokens < 1.0
    
    # Next acquire should block or timeout
    start_time = time.monotonic()
    try:
        await limiter.acquire(5, timeout=0.1)
        pytest.fail("Should have timed out")
    except CapacityExceededError:
        pass
    elapsed = time.monotonic() - start_time
    assert elapsed >= 0.1

    # Wait for tokens to replenish
    await asyncio.sleep(0.5)
    acquired = await limiter.acquire(3, timeout=0.1)
    assert acquired is True

@pytest.mark.asyncio
async def test_with_retries_success():
    call_count = 0
    
    @with_retries(max_retries=3, base_delay=0.01)
    async def flaky_call():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ProviderBusyException("Busy")
        return "Success"
        
    result = await flaky_call()
    assert result == "Success"
    assert call_count == 3

@pytest.mark.asyncio
async def test_with_retries_failure():
    call_count = 0
    
    @with_retries(max_retries=2, base_delay=0.01)
    async def failing_call():
        nonlocal call_count
        call_count += 1
        raise ProviderBusyException("Busy")
        
    with pytest.raises(ProviderBusyException):
        await failing_call()
    
    assert call_count == 3 # Initial call + 2 retries


# --------------------------------------------------------------------------
# Circuit breaker + backpressure wiring.
#
# Both primitives previously existed but were never called by the pipeline:
# RateLimiter had passing unit tests and zero call sites, and no circuit
# breaker existed at all despite docs/capabilities.md advertising one.
# --------------------------------------------------------------------------
class _DeadProvider(BaseProvider):
    def __init__(self):
        self.calls = 0
        self.num_retries = 0

    async def complete(self, request, context):
        self.calls += 1
        raise ProviderBusyException("503 upstream down")

    async def complete_stream(self, request, context):
        yield ""


class _OkProvider(BaseProvider):
    async def complete(self, request, context):
        return LLMResponse(content="ok", model=request.model or "m", usage=TokenUsage())

    async def complete_stream(self, request, context):
        yield "ok"


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=60)
    assert await breaker.allows("m")
    await breaker.record_failure("m")
    assert await breaker.allows("m")
    await breaker.record_failure("m")
    assert not await breaker.allows("m")
    assert breaker.state("m") == "open"


@pytest.mark.asyncio
async def test_circuit_breaker_half_opens_then_closes_on_success():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)
    await breaker.record_failure("m")
    assert not await breaker.allows("m")

    await asyncio.sleep(0.06)
    assert await breaker.allows("m")          # one probe permitted
    assert breaker.state("m") == "half_open"

    await breaker.record_success("m")
    assert breaker.state("m") == "closed"


@pytest.mark.asyncio
async def test_circuit_breaker_reopens_when_probe_fails():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)
    await breaker.record_failure("m")
    await asyncio.sleep(0.06)
    assert await breaker.allows("m")
    await breaker.record_failure("m")
    assert breaker.state("m") == "open"
    assert not await breaker.allows("m")


@pytest.mark.asyncio
async def test_pipeline_stops_calling_a_dead_provider():
    config = TautConfig(
        cache=None, compression=None, prefix=None, restraint=None,
        resilience=ResilienceConfig(circuit_failure_threshold=2, circuit_reset_timeout=60),
    )
    pipeline = create_pipeline(config)
    provider = _DeadProvider()
    pipeline._provider = provider

    for _ in range(5):
        with pytest.raises(FallbackExhaustedError):
            await pipeline.run(LLMRequest(intent="x", model="dead-model"))

    # Without the breaker this would be 5. The breaker opens after 2 and the
    # remaining requests fail without touching the network.
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_rate_limiter_raises_capacity_exceeded_through_the_pipeline():
    """docs/limitations.md tells callers to catch CapacityExceededError and
    queue the job. That contract needs the limiter actually wired in."""
    config = TautConfig(
        cache=None, compression=None, prefix=None, restraint=None,
        resilience=ResilienceConfig(requests_per_second=0.001, burst=1, acquire_timeout=0.05),
    )
    pipeline = create_pipeline(config)
    pipeline._provider = _OkProvider()

    await pipeline.run(LLMRequest(intent="first", model="m"))
    with pytest.raises(CapacityExceededError):
        await pipeline.run(LLMRequest(intent="second", model="m"))


@pytest.mark.asyncio
async def test_rate_limiting_is_off_by_default():
    pipeline = create_pipeline(
        TautConfig(cache=None, compression=None, prefix=None, restraint=None)
    )
    pipeline._provider = _OkProvider()
    for _ in range(5):
        assert (await pipeline.run(LLMRequest(intent="x", model="m"))).content == "ok"
