import pytest
from taut.layers.compression.registry import CompressionRegistry, register_compressor
from taut.layers.compression.detector import ContentDetector
from taut.core.config import CompressionConfig
from taut.layers.compression.middleware import CompressionMiddleware
from taut.core.models import PipelineContext

@pytest.fixture(autouse=True)
def reset_registry():
    CompressionRegistry.clear()
    yield
    CompressionRegistry.clear()

def test_registry_registration():
    @register_compressor("application/xml", matcher=lambda t: t.strip().startswith("<xml>"))
    def compress_xml(text):
        return "<compressed/>"
        
    func = CompressionRegistry.get_compressor("application/xml")
    assert func is not None
    assert func("<xml>test</xml>") == "<compressed/>"

def test_detector_with_registry():
    @register_compressor("custom/cypher", matcher=lambda t: "MATCH" in t)
    def compress_cypher(text):
        return "compressed_cypher"
        
    detector = ContentDetector()
    mime_type = detector.detect("MATCH (n) RETURN n")
    assert mime_type == "custom/cypher"

def test_middleware_with_registry():
    @register_compressor("application/xml", matcher=lambda t: t.strip().startswith("<xml>"))
    def compress_xml(text):
        return "<compressed/>"
        
    config = CompressionConfig(enabled=True)
    middleware = CompressionMiddleware(config)
    context = PipelineContext()
    
    result = middleware._apply_strategy("<xml>data</xml>", "application/xml", context)
    assert result == "<compressed/>"
