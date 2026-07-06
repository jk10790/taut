"""Classifies requests by complexity."""
from taut.core.models import LLMRequest
import litellm
from taut.layers.compression.detector import ContentDetector
from pydantic import BaseModel

class ClassificationResult(BaseModel):
    score: float
    tier: str
    reason: str

class HeuristicClassifier:
    """Calculates complexity score based on heuristics."""
    
    def __init__(self, keywords: list[str] | None = None):
        self.keywords = keywords or ["analyze", "evaluate", "synthesize", "architect", "regex", "recursive", "algorithm"]
        self.detector = ContentDetector()
        
    def calculate_score(self, request: LLMRequest, thresholds: dict[str, float] | None = None) -> ClassificationResult:
        """Calculate a complexity score between 0.0 and 1.0."""
        score = 0.0
        reason_parts = []
        
        # 1. Tool count factor (up to 0.3)
        tool_count = len(request.tools) if request.tools else 0
        if tool_count > 0:
            tool_score = min(0.3, tool_count * 0.05)
            score += tool_score
            reason_parts.append(f"tools ({tool_count})")
        
        # 2. Text complexity factor
        text_content = ""
        
        if request.system_prompt:
            text_content += request.system_prompt
            
        if request.messages:
            for msg in request.messages:
                if isinstance(msg.content, str):
                    text_content += "\n" + msg.content
                    
        # Token estimation (using fallback model)
        try:
            # We assume a 40% token reduction if content is code or json
            content_type = self.detector.detect(text_content)
            raw_tokens = litellm.token_counter(model="gpt-3.5-turbo", text=text_content)
            if content_type in ["JSON", "CODE"]:
                estimated_tokens = int(raw_tokens * 0.6)
            else:
                estimated_tokens = raw_tokens
        except Exception:
            estimated_tokens = len(text_content) // 4
            
        # Length factor (up to 0.3)
        length_score = min(0.3, estimated_tokens / 4000.0)
        score += length_score
        if length_score > 0.1:
            reason_parts.append(f"length (~{estimated_tokens} tokens)")
        
        # Keyword factor (up to 0.4)
        text_lower = text_content.lower()
        keyword_hits = sum(1 for kw in self.keywords if kw in text_lower)
        if keyword_hits > 0:
            kw_score = min(0.4, keyword_hits * 0.1)
            score += kw_score
            reason_parts.append(f"keywords ({keyword_hits} found)")
            
        score = min(1.0, score)
        
        if not reason_parts:
            reason_parts.append("simple request")
            
        reason = "Classification based on: " + ", ".join(reason_parts)
        
        # Determine tier
        thresholds = thresholds or {"simple": 0.3, "standard": 0.7}
        
        if score < thresholds.get("simple", 0.3):
            tier = "simple"
        elif score < thresholds.get("standard", 0.7):
            tier = "standard"
        else:
            tier = "complex"
            
        return ClassificationResult(score=score, tier=tier, reason=reason)
