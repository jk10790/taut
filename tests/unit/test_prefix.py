from taut.core.models import Message
from taut.layers.prefix.analyzer import PrefixAnalyzer
from taut.layers.prefix.providers.openai import OpenAIPrefixStrategy

def test_prefix_analyzer_detects_cache_breakers():
    analyzer = PrefixAnalyzer()
    
    text = (
        "Here is some text with a uuid: 123e4567-e89b-12d3-a456-426614174000. "
        "And a timestamp: 2026-07-05T13:49:21. "
        "And an iso_date: 2026-07-05."
    )
    
    breakers = analyzer.scan(text)
    
    assert len(breakers) == 3
    
    breaker_types = {b["type"] for b in breakers}
    assert breaker_types == {"uuid", "timestamp", "iso_date"}
    
    for breaker in breakers:
        if breaker["type"] == "uuid":
            assert breaker["value"] == "123e4567-e89b-12d3-a456-426614174000"
        elif breaker["type"] == "timestamp":
            assert breaker["value"] == "2026-07-05T13:49:21"
        elif breaker["type"] == "iso_date":
            assert breaker["value"] == "2026-07-05"

def test_prefix_analyzer_no_breakers():
    analyzer = PrefixAnalyzer()
    text = "This is a static prompt with no dynamic cache breakers."
    
    breakers = analyzer.scan(text)
    assert len(breakers) == 0

def test_openai_prefix_strategy_alignment():
    strategy = OpenAIPrefixStrategy()
    
    messages = [
        Message(role="user", content="Hello"),
        Message(role="system", content="You are a helpful assistant."),
        Message(role="assistant", content="Hi!"),
        Message(role="system", content="Always be polite.")
    ]
    
    aligned = strategy.align(messages)
    
    # Check that system messages are moved to the front
    assert len(aligned) == 4
    assert aligned[0].role == "system"
    assert aligned[0].content == "You are a helpful assistant."
    assert aligned[1].role == "system"
    assert aligned[1].content == "Always be polite."
    assert aligned[2].role == "user"
    assert aligned[2].content == "Hello"
    assert aligned[3].role == "assistant"
    assert aligned[3].content == "Hi!"
