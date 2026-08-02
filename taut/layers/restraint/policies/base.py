"""Base class for output restraint policies."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from taut.core.models import PipelineContext
from abc import ABC, abstractmethod
from typing import Any
from taut.core.config import OutputRestraintConfig

class RestraintPolicy(ABC):
    """Abstract base class for output restraint policies."""
    
    def __init__(self, config: OutputRestraintConfig | None = None):
        self.config = config or OutputRestraintConfig()
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the policy."""
        ...
        
    @abstractmethod
    def get_instructions(self, context: 'PipelineContext | None' = None) -> str:
        """Get the policy instructions to inject."""
        ...
        
    def get_max_tokens(self, context: 'PipelineContext | None' = None) -> int | None:
        """Get the maximum tokens allowed for this policy."""
        return None
        
    def get_response_format(self, provider: str | None = None, context: 'PipelineContext | None' = None) -> dict[str, Any] | None:
        """Get the required response format."""
        return None
