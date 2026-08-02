import json
import ast
import re
from .registry import CompressionRegistry

class ContentDetector:
    """Detects content types: custom plugins, JSON, CODE, PROSE, MIXED."""
    
    @staticmethod
    def is_json(text: str) -> bool:
        text = text.strip()
        if not (text.startswith('{') and text.endswith('}')) and not (text.startswith('[') and text.endswith(']')):
            return False
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def is_code(text: str) -> bool:
        # Require BOTH structural markers and successful ast.parse()
        has_structural_markers = (
            re.search(r'\b(def|class)\b\s+[\w]+\s*\(.*?\)\s*:', text, flags=re.DOTALL) or 
            re.search(r'^\s+(?:if|for|while|with|try|except|return|yield|pass)\b', text, flags=re.MULTILINE)
        )
        if not has_structural_markers:
            return False
            
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            return False
            
    @staticmethod
    def is_prose(text: str) -> bool:
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        return len(words) > 5 and len(sentences) > 1

    @staticmethod
    def detect(text: str) -> str:
        """
        Detects the predominant content type.
        Returns the registered MIME type, or one of 'json', 'code', 'prose', 'mixed'.
        """
        # 1. Check custom registries first
        for p in CompressionRegistry._plugins:
            if p["matcher"] and p["matcher"](text):
                return p["mime_type"]
                
        # 2. Built-in heuristics.
        #
        # Order matters. is_code() is the strict test -- it requires BOTH
        # structural markers (a def/class signature, or an indented control
        # keyword) AND a successful ast.parse(). is_prose() is the loose test:
        # ">5 words over >1 sentence" matches almost anything, including source
        # code, because attribute access (`item.get`) reads as a sentence
        # boundary. Running the loose test first sent essentially all bare
        # source code down the prose path, leaving the AST compressor
        # unreachable outside fenced blocks. Strict before loose.
        if ContentDetector.is_json(text):
            return "json"

        if re.search(r'```', text):
            return "mixed"

        if ContentDetector.is_code(text):
            return "code"

        if ContentDetector.is_prose(text):
            return "prose"

        return "prose"
