"""Anthropic-specific prefix alignment strategy."""
from taut.core.models import Message

class AnthropicPrefixStrategy:
    """Strategy for aligning prefixes for Anthropic models.
    
    Anthropic allows up to 4 explicit cache breakpoints via:
    `cache_control: {"type": "ephemeral"}`
    """
    
    def align(self, messages: list[Message]) -> list[Message]:
        """Inject cache_control breakpoints into messages.
        
        Args:
            messages: Original messages.
            
        Returns:
            Messages with cache_control injected.
        """
        aligned_messages = []
        breakpoints_injected = 0
        MAX_BREAKPOINTS = 4
        
        for msg in messages:
            new_msg = msg.model_copy(deep=True)
            
            if msg.role == "system" and breakpoints_injected < MAX_BREAKPOINTS:
                if isinstance(new_msg.content, list):
                    if len(new_msg.content) > 0 and isinstance(new_msg.content[-1], dict):
                        new_msg.content[-1]["cache_control"] = {"type": "ephemeral"}
                        breakpoints_injected += 1
                elif isinstance(new_msg.content, str):
                    new_msg.metadata = new_msg.metadata or {}
                    new_msg.metadata["cache_control"] = {"type": "ephemeral"}
                    breakpoints_injected += 1
                    
            aligned_messages.append(new_msg)
            
        return aligned_messages
