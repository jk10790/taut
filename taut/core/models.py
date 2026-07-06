"""Core data models for the taut pipeline."""
from __future__ import annotations
import time
from typing import Any, Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import Self

from taut.core.prompt_blocks import PromptBlock

class Message(BaseModel):
    """A single message in an LLM conversation."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class ContentBlock(BaseModel):
    """A typed block of content within a request context."""
    type: Literal["text", "json", "code", "image"]
    content: str
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class LLMRequest(BaseModel):
    """Canonical request model for the taut pipeline."""
    intent: str
    context: str | list[ContentBlock] | None = None
    messages: list[Message] | None = None
    blocks: list[PromptBlock] | None = None
    model: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    namespace: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    _original_token_count: int | None = PrivateAttr(default=None)

class TokenUsage(BaseModel):
    """Token usage statistics.
    
    Note: cached_tokens is typically a subset of input_tokens.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int | None = 0
    total_tokens: int = 0
    @model_validator(mode="after")
    def _compute_total(self) -> Self:
        self.total_tokens = self.input_tokens + self.output_tokens
        return self

class LayerMetrics(BaseModel):
    """Metrics for a single middleware layer."""
    layer_name: str
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    latency_ms: float = 0.0
    applied: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    @model_validator(mode="after")
    def _compute_saved(self) -> Self:
        self.tokens_saved = max(0, self.tokens_before - self.tokens_after)
        return self

class PipelineMetrics(BaseModel):
    """Aggregated metrics across all pipeline layers."""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    total_tokens_saved: int = 0
    estimated_cost_without_taut: float = 0.0
    estimated_cost_with_taut: float = 0.0
    estimated_cost_saved: float = 0.0
    total_latency_ms: float = 0.0
    layers: list[LayerMetrics] = Field(default_factory=list)
    model_used: str | None = None
    cache_hit: bool = False

class LLMResponse(BaseModel):
    """Canonical response model."""
    content: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    metrics: PipelineMetrics = Field(default_factory=PipelineMetrics)
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    raw_response: dict[str, Any] | None = None

class PipelineContext(BaseModel):
    """Shared mutable state carried through the middleware chain."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    start_time: float = Field(default_factory=time.time)
    metrics: PipelineMetrics = Field(default_factory=PipelineMetrics)
    provider_name: str | None = None
    selected_model: str | None = None
    selected_tier: str | None = None
    content_types_detected: dict[str, str] = Field(default_factory=dict)
    compression_applied: bool = False
    compression_ratio: float = 0.0
    routing_reason: str | None = None
    original_request: LLMRequest | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
