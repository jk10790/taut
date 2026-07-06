import json
import ast
import re

class ContentDetector:
    """Detects content types: JSON, CODE, PROSE, MIXED."""
    
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
        Returns one of 'json', 'code', 'prose', 'mixed'.
        """
        if ContentDetector.is_json(text):
            return "json"
            
        if re.search(r'```', text):
            return "mixed"
            
        if ContentDetector.is_prose(text):
            return "prose"
            
        if ContentDetector.is_code(text):
            return "code"
            
        return "prose"
