"""Structured JSON output restraint policy."""
from typing import Any
from .base import RestraintPolicy

class StructuredPolicy(RestraintPolicy):
    """Forces json_object format."""
    
    @property
    def name(self) -> str:
        return "structured"
        
    def get_instructions(self, context: 'PipelineContext | None' = None) -> str:
        return (
            "CRITICAL OUTPUT RESTRAINT: You must output strictly valid JSON. "
            "Do not include markdown code blocks, explanations, or any text outside the JSON object."
        )
        
    def get_response_format(self, provider: str | None = None, context: 'PipelineContext | None' = None) -> dict[str, Any] | None:
        if provider == "openai":
            return {"type": "json_object"}
        return None
