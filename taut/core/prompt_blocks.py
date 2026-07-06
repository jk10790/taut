"""PromptBlock API for structured prompt construction."""
from dataclasses import dataclass
from typing import Literal, Any

@dataclass
class PromptBlock:
    content: str
    stability: Literal["immutable", "stable", "semi_stable", "dynamic"] = "dynamic"
    cache_eligible: bool = False

@dataclass
class SystemBlock(PromptBlock):
    stability: Literal["immutable", "stable", "semi_stable", "dynamic"] = "immutable"
    cache_eligible: bool = True

@dataclass
class ToolsBlock(PromptBlock):
    stability: Literal["immutable", "stable", "semi_stable", "dynamic"] = "stable"
    cache_eligible: bool = True
    tools: list[dict[str, Any]] | None = None

@dataclass
class ContextBlock(PromptBlock):
    stability: Literal["immutable", "stable", "semi_stable", "dynamic"] = "semi_stable"

@dataclass
class QueryBlock(PromptBlock):
    stability: Literal["immutable", "stable", "semi_stable", "dynamic"] = "dynamic"
