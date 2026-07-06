"""Adaptive output restraint policy."""
from typing import Any
from taut.core.models import PipelineContext
from .base import RestraintPolicy
from .terse import TersePolicy
from .yagni import YAGNIPolicy

class AdaptivePolicy(RestraintPolicy):
    """Adapts restraint policy based on routing tier."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.terse = TersePolicy(config)
        self.yagni = YAGNIPolicy(config)
        
    @property
    def name(self) -> str:
        return "adaptive"
        
    def _get_active_policy(self, context: 'PipelineContext | None') -> RestraintPolicy | None:
        if not context:
            return self.yagni
            
        tier = getattr(context, 'selected_tier', None)
        if tier == "simple":
            return self.terse
        elif tier == "standard":
            return self.yagni
        else: # complex or unknown
            return None
            
    def get_instructions(self, context: 'PipelineContext | None' = None) -> str:
        policy = self._get_active_policy(context)
        if policy:
            return policy.get_instructions(context)
        return ""
        
    def get_max_tokens(self, context: 'PipelineContext | None' = None) -> int | None:
        policy = self._get_active_policy(context)
        if policy:
            return policy.get_max_tokens(context)
        return None
        
    def get_response_format(self, provider: str | None = None, context: 'PipelineContext | None' = None) -> dict[str, Any] | None:
        policy = self._get_active_policy(context)
        if policy:
            return policy.get_response_format(provider, context)
        return None
