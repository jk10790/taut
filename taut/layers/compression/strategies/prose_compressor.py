import re

from .base import CompressionResult, CompressionStrategy

class ProseCompressor(CompressionStrategy):
    """
    Rule-based filler word removal.
    """
    
    FILLER_PATTERNS = [
        (r'\bin order to\b', 'to'),
        (r'\bfor the purpose of\b', 'for'),
        (r'\bdue to the fact that\b', 'because'),
        (r'\bin the event that\b', 'if'),
        (r'\bwith the exception of\b', 'except'),
        (r'\bat this point in time\b', 'now'),
        (r'\bdespite the fact that\b', 'although'),
        (r'\bhas the ability to\b', 'can'),
        (r'\bbasically\b', ''),
        (r'\bactually\b', ''),
        (r'\bliterally\b', ''),
    ]

    def __init__(self):
        pass

    def compress(self, text: str) -> CompressionResult:
        original_size = len(text)
        compressed_text = text

        # Rule-based filler removal
        for pattern, replacement in self.FILLER_PATTERNS:
            compressed_text = re.sub(pattern, replacement, compressed_text, flags=re.IGNORECASE)
        
        # Cleanup extra spaces left by removals
        compressed_text = re.sub(r'\s{2,}', ' ', compressed_text)
            
        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text.strip(),
            original_size=original_size,
            compressed_size=len(compressed_text.strip()),
            metadata={"strategy": "prose_compressor"}
        )
