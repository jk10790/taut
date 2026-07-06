"""Google-specific prefix alignment strategy."""
from taut.core.models import Message

class GooglePrefixStrategy:
    """Strategy for aligning prefixes for Google models.
    
    Google caches based on large contiguous prefixes.
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
