"""Terse output restraint policy."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from taut.core.models import PipelineContext
from .base import RestraintPolicy

class TersePolicy(RestraintPolicy):
    """Telegraphic style instructions."""
    
    @property
    def name(self) -> str:
        return "terse"
        
    def get_instructions(self, context: 'PipelineContext | None' = None) -> str:
        return (
            "CRITICAL OUTPUT RESTRAINT: Be incredibly terse. Use telegraphic style. "
            "Skip conversational openings/closings. Answer in as few words as possible."
        )
        
    def get_max_tokens(self, context: 'PipelineContext | None' = None) -> int | None:
        return min(256, self.config.max_tokens_cap or 4096)
