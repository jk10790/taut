"""OpenAI-specific prefix alignment strategy."""
from taut.core.models import Message

class OpenAIPrefixStrategy:
    """Strategy for aligning prefixes for OpenAI models.
    
    OpenAI requires a contiguous static prefix of at least 1024 tokens to cache.
    This strategy moves dynamic elements out of the prefix.
    """
    
    def align(self, messages: list[Message]) -> list[Message]:
        """Align messages for optimal caching.
        
        Args:
            messages: Original messages.
            
        Returns:
            Reordered messages.
        """
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]
        
        return system_msgs + other_msgs
