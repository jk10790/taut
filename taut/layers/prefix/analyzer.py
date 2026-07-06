"""Analyzes prompts for cache breakers."""
import re
from typing import Any

class PrefixAnalyzer:
    """Scans system prompts and early messages for CacheBreakers."""
    
    # Common dynamic patterns that break caching
    CACHE_BREAKERS = {
        "uuid": re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I),
        "timestamp": re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),
        "iso_date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    }
    
    def scan(self, text: str) -> list[dict[str, Any]]:
        """Scan text for cache breakers.
        
        Args:
            text: The text to scan.
            
        Returns:
            List of detected cache breakers.
        """
        breakers = []
        for name, pattern in self.CACHE_BREAKERS.items():
            for match in pattern.finditer(text):
                breakers.append({
                    "type": name,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })
        return breakers
