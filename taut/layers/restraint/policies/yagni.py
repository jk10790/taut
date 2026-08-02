"""YAGNI (You Aren't Gonna Need It) output restraint policy."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from taut.core.models import PipelineContext
from .base import RestraintPolicy

class YAGNIPolicy(RestraintPolicy):
    """Strict instructions against filler, caveats, and unnecessary code."""
    
    @property
    def name(self) -> str:
        return "yagni"
        
    def get_instructions(self, context: 'PipelineContext | None' = None) -> str:
        return (
            "CRITICAL OUTPUT RESTRAINT: You must follow YAGNI (You Aren't Gonna Need It). "
            "Output ONLY what is explicitly requested. Do not include filler, caveats, "
            "explanatory text, or unnecessary code. Avoid premature generalization."
        )
        
    def get_max_tokens(self, context: 'PipelineContext | None' = None) -> int | None:
        return self.config.max_tokens_cap or 4096
