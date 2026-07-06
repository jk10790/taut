import pytest
import asyncio
import time
from taut.core.resilience import RateLimiter, with_retries, ProviderBusyException, CapacityExceededError

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
