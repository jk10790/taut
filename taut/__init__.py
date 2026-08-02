"""taut — AI Efficiency Middleware. Zero Waste Compute for LLM applications."""

__version__ = "0.1.0"

from taut.core.models import (
    LLMRequest,
    LLMResponse,
    PipelineContext,
    TokenUsage,
    PipelineMetrics,
    Message,
    ContentBlock,
    LayerMetrics,
)
from taut.core.config import (
    CompressionConfig,
    OutputRestraintConfig,
    PrefixAlignmentConfig,
    SemanticCacheConfig,
    TautConfig,
    TieredRoutingConfig,
)
from taut.core.prompt_blocks import (
    ContextBlock,
    PromptBlock,
    QueryBlock,
    SystemBlock,
    ToolsBlock,
)
from taut.core.resilience import CapacityExceededError
from taut.core.middleware import Middleware
from taut.providers.base import BaseProvider

# Short aliases for the documented `SemanticCache(...)` spelling. Both the
# canonical *Config names and these aliases are exported, because the docs and
# examples used the canonical names while only the aliases were importable.
SemanticCache = SemanticCacheConfig
Compression = CompressionConfig
PrefixAlignment = PrefixAlignmentConfig
TieredRouting = TieredRoutingConfig
OutputRestraint = OutputRestraintConfig

def __getattr__(name: str):
    if name in ("Pipeline", "create_pipeline"):
        from taut.core.pipeline import Pipeline, create_pipeline
        globals()["Pipeline"] = Pipeline
        globals()["create_pipeline"] = create_pipeline
        return globals()[name]
    
    if name == "CacheBackend":
        try:
            from taut.layers.cache.backends.base import CacheBackend
            globals()["CacheBackend"] = CacheBackend
            return CacheBackend
        except ImportError:
            pass
            
    if name == "Embedder":
        try:
            from taut.layers.cache.embedder import Embedder
            globals()["Embedder"] = Embedder
            return Embedder
        except ImportError:
            pass
            
    if name == "register_compressor":
        try:
            from taut.layers.compression.registry import register_compressor
            globals()["register_compressor"] = register_compressor
            return register_compressor
        except ImportError:
            pass

    raise AttributeError(f"module 'taut' has no attribute {name}")

__all__ = [
    "__version__",
    "LLMRequest", "LLMResponse", "PipelineContext", "TokenUsage",
    "PipelineMetrics", "Message", "ContentBlock", "LayerMetrics",
    "SemanticCache", "Compression", "PrefixAlignment", "TieredRouting",
    "OutputRestraint",
    "SemanticCacheConfig", "CompressionConfig", "PrefixAlignmentConfig",
    "TieredRoutingConfig", "OutputRestraintConfig", "TautConfig",
    "PromptBlock", "SystemBlock", "ToolsBlock", "ContextBlock", "QueryBlock",
    "CapacityExceededError",
    "Pipeline", "create_pipeline",
    "Middleware", "BaseProvider",
    "CacheBackend", "Embedder",
    "register_compressor",
]
