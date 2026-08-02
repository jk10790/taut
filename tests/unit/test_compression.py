import pytest
import json
from unittest.mock import AsyncMock
from taut.core.models import LLMRequest, PipelineContext
from taut.core.config import CompressionConfig
from taut.layers.compression.middleware import CompressionMiddleware
from taut.layers.compression.strategies.json_crusher import SmartCrusher
from taut.layers.compression.strategies.code_compressor import CodeCompressor
from taut.layers.compression.detector import ContentDetector

pytestmark = pytest.mark.asyncio

def test_smart_crusher_valid_json():
    crusher = SmartCrusher()
    json_data = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"a": 5, "b": 6}
    ]
    text = json.dumps(json_data)
    
    result = crusher.compress(text)
    
    assert "COLS: a | b" in result.compressed_text
    assert result.compressed_size < result.original_size
    assert result.metadata["strategy"] == "json_crusher"

def test_smart_crusher_invalid_json():
    crusher = SmartCrusher()
    text = "Not a JSON string at all"
    
    result = crusher.compress(text)
    
    assert result.compressed_text == text
    assert result.compressed_size == result.original_size
    assert result.metadata.get("error") or result.metadata.get("strategy") != "json_crusher"

def test_smart_crusher_heterogeneous_json():
    crusher = SmartCrusher()
    json_data = [
        {"a": 1, "b": 2},
        {"a": 3, "c": 4} # Different keys
    ]
    text = json.dumps(json_data)
    
    result = crusher.compress(text)
    
    assert result.compressed_text == text
    assert result.compressed_size == result.original_size

def test_smart_crusher_delimiter_characters():
    # Edge case: JSON with delimiter characters in values
    crusher = SmartCrusher()
    json_data = [
        {"a": "val1, with comma", "b": "val2|with pipe"},
        {"a": "val3\x1fwith US", "b": "val4\x1ewith RS"}
    ]
    text = json.dumps(json_data)
    result = crusher.compress(text)
    # Just ensure it doesn't crash and returns validly handled text
    assert result.compressed_text is not None

def test_code_compressor_valid_python():
    compressor = CodeCompressor()
    
    code = '''
def add(a: int, b: int) -> int:
    """This function adds two numbers."""
    return a + b

class MyClass:
    """This is a test class."""
    def __init__(self):
        self.val: int = 10
'''
    
    result = compressor.compress(code)
    compressed = result.compressed_text
    
    assert "def add(a, b):" in compressed or "def add(a: int, b: int) -> int:" in compressed
    assert "This function adds two numbers." not in compressed
    assert "This is a test class." not in compressed
    assert "self.val" in compressed
    assert result.compressed_size < result.original_size
    assert result.metadata["strategy"] == "code_compressor"

def test_code_compressor_regex_fallback():
    compressor = CodeCompressor()
    
    code = '''
def func( # incomplete
# some comment
'''
    
    result = compressor.compress(code)
    compressed = result.compressed_text
    
    assert "def func(" in compressed
    assert "some comment" not in compressed
    assert "incomplete" not in compressed

def test_code_compressor_docstring_only_function():
    # Edge case: Function with only a docstring should become def foo(): pass
    compressor = CodeCompressor()
    
    code = '''
def empty_func():
    """This function does nothing but has a docstring."""
'''
    
    result = compressor.compress(code)
    compressed = result.compressed_text
    
    assert "pass" in compressed
    assert "This function does nothing" not in compressed

def test_content_detector_prose_with_keywords():
    # Edge case: Prose containing Python keywords should be detected as PROSE, not CODE
    detector = ContentDetector()
    text = "If you def a class with import and return, it might look like code."
    content_type = detector.detect(text)
    # Depending on detector enum/return, checking string repr
    assert str(content_type).lower() == "prose"

async def test_compression_middleware_process():
    config = CompressionConfig(
        enabled=True,
        skip_for_simple_tier=True,
        min_compress_tokens=10
    )
    middleware = CompressionMiddleware(config=config)
    
    context = PipelineContext()
    context.selected_tier = "standard"
    
    # Needs a long enough string to pass min_compress_tokens
    code_text = "def foo():\n    '''docstring here'''\n    pass\n" * 10
    request = LLMRequest(intent="Compress this", context=code_text)
    
    next_handler = AsyncMock()
    next_handler.return_value = "Success"
    
    result = await middleware.process(request, context, next_handler)
    
    assert result == "Success"
    assert context.compression_applied is True
    # The request context should be compressed
    assert "docstring here" not in request.context
    next_handler.assert_called_once_with(request, context)

async def test_compression_middleware_skips_simple_tier():
    config = CompressionConfig(
        enabled=True,
        skip_for_simple_tier=True,
        min_compress_tokens=10
    )
    middleware = CompressionMiddleware(config=config)
    
    context = PipelineContext()
    context.selected_tier = "simple"
    
    code_text = "def foo():\n    '''docstring here'''\n    pass\n" * 10
    request = LLMRequest(intent="Compress this", context=code_text)
    
    next_handler = AsyncMock()
    next_handler.return_value = "Success"
    
    await middleware.process(request, context, next_handler)
    
    assert context.compression_applied is False
    assert "docstring here" in request.context # unmodified
