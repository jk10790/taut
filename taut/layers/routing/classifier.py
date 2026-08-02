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

        # request.intent is the user's actual question. Omitting it meant the
        # complexity keywords ("analyze", "synthesize", "architect", ...) were
        # scored against everything *except* the sentence most likely to
        # contain them, so reasoning-heavy asks under-scored and were routed to
        # a cheap model.
        if request.intent:
            text_content += request.intent

        if request.system_prompt:
            text_content += "\n" + request.system_prompt

        if request.messages:
            for msg in request.messages:
                if isinstance(msg.content, str):
                    text_content += "\n" + msg.content

        # request.context carries the bulk payload (log dumps, JSON arrays,
        # repository slices) and must be scored. Omitting it meant a 20KB
        # context scored 0.0 and routed to the cheapest tier -- which in turn
        # tripped skip_for_simple_tier and disabled compression, so the two
        # savings layers were both bypassed for precisely the large-payload
        # requests taut exists to optimise.
        if request.context:
            if isinstance(request.context, str):
                text_content += "\n" + request.context
            else:
                text_content += "\n" + "\n".join(
                    getattr(block, "content", "") for block in request.context
                )

        # Token estimation (using fallback model)
        try:
            # Code and JSON compress hard downstream, so their raw token count
            # overstates what will actually be sent. Discount accordingly.
            # (Compared lowercase: the detector returns 'json'/'code', so the
            # previous uppercase comparison never matched.)
            content_type = self.detector.detect(text_content)
            raw_tokens = litellm.token_counter(model="gpt-3.5-turbo", text=text_content)
            if content_type in ("json", "code"):
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
