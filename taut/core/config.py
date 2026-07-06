"""Configuration data classes for the taut pipeline."""
from __future__ import annotations
import os
from typing import Literal, Any
from pydantic import BaseModel, Field, ConfigDict


class SemanticCacheConfig(BaseModel):
    """Configuration for Layer 1: Semantic Cache."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    backend: Literal["memory", "redis"] = "memory"
    similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    ttl_seconds: int = Field(default=600, ge=0)
    max_entries: int = Field(default=10000, ge=1)
    embedding_model: str = "Xenova/all-MiniLM-L6-v2"
    redis_url: str | None = None
    embedder: Any = None


class CompressionConfig(BaseModel):
    """Configuration for Layer 2: Payload & Prose Compression."""
    model_config = ConfigDict(populate_by_name=True)
    json_enabled: bool = Field(default=True, alias="json")
    code_enabled: bool = Field(default=True, alias="code")
    prose: Literal["off", "rules"] = "rules"
    min_tokens_to_compress: int = Field(default=100, ge=0)
    llmlingua_target_ratio: float = Field(default=0.5, ge=0.1, le=0.9)
    skip_for_simple_tier: bool = True
    min_compress_tokens: int = 500


class PrefixAlignmentConfig(BaseModel):
    """Configuration for Layer 3: Prefix Cache Alignment."""
    enabled: bool = True
    provider_hints: Literal["auto", "openai", "anthropic", "google"] = "auto"
    static_blocks: list[str] = Field(default_factory=list)
    detect_cache_breakers: bool = True


class TieredRoutingConfig(BaseModel):
    """Configuration for Layer 4: Tiered Routing."""
    tiers: dict[str, list[str]] = Field(default_factory=lambda: {
        "simple": ["gpt-4o-mini"],
        "standard": ["gpt-4o"],
        "complex": ["o3"],
    })
    default_tier: str = "standard"
    classifier: Literal["heuristic"] = "heuristic"
    complexity_keywords: list[str] = Field(default_factory=list)
    complexity_thresholds: dict[str, float] = Field(default_factory=lambda: {
        "simple": 0.3,
        "standard": 0.7,
    })


class OutputRestraintConfig(BaseModel):
    """Configuration for Layer 5: Output Restraint."""
    policy: Literal["off", "yagni", "terse", "structured", "adaptive"] = "yagni"
    max_tokens_cap: int | None = None
    custom_constraints: list[str] = Field(default_factory=list)


class TautConfig(BaseModel):
    """Top-level configuration for the taut pipeline."""
    provider: str = "litellm"
    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    num_retries: int = 2
    timeout: float = 60.0
    fallback_models: list[str] | None = None
    cache: SemanticCacheConfig | None = Field(default_factory=SemanticCacheConfig)
    compression: CompressionConfig | None = Field(default_factory=CompressionConfig)
    prefix: PrefixAlignmentConfig | None = Field(default_factory=PrefixAlignmentConfig)
    routing: TieredRoutingConfig | None = None
    restraint: OutputRestraintConfig | None = Field(default_factory=OutputRestraintConfig)
    
    @classmethod
    def from_env(cls) -> TautConfig:
        """Load configuration from environment variables."""
        kwargs = {
            "provider": os.getenv("TAUT_PROVIDER", "litellm"),
        }
        if os.getenv("TAUT_API_KEY"): kwargs["api_key"] = os.getenv("TAUT_API_KEY")
        if os.getenv("TAUT_BASE_URL"): kwargs["base_url"] = os.getenv("TAUT_BASE_URL")
        if os.getenv("TAUT_DEFAULT_MODEL"): kwargs["default_model"] = os.getenv("TAUT_DEFAULT_MODEL")
        
        cache_config = SemanticCacheConfig()
        if os.getenv("TAUT_CACHE_BACKEND"): cache_config.backend = os.getenv("TAUT_CACHE_BACKEND")  # type: ignore
        if os.getenv("TAUT_EMBEDDING_MODEL"): cache_config.embedding_model = os.getenv("TAUT_EMBEDDING_MODEL")  # type: ignore
        kwargs["cache"] = cache_config
        
        kwargs["compression"] = CompressionConfig()
        kwargs["prefix"] = PrefixAlignmentConfig()
        
        routing_kwargs = {}
        routing_tiers_env = os.getenv("TAUT_ROUTING_TIERS")
        if routing_tiers_env:
            import json
            import logging
            try:
                routing_kwargs["tiers"] = json.loads(routing_tiers_env)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse TAUT_ROUTING_TIERS JSON: {e}")
        kwargs["routing"] = TieredRoutingConfig(**routing_kwargs)
        
        kwargs["restraint"] = OutputRestraintConfig()
        
        return cls(**kwargs)
