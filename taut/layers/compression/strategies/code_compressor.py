import ast
import re
from typing import Any

from .base import CompressionResult, CompressionStrategy

class CodeCompressor(CompressionStrategy):
    """
    AST-aware Python code compression. Removes docstrings, type annotations.
    Regex fallback for comments and blank lines.
    """

    def compress(self, text: str) -> CompressionResult:
        original_size = len(text)
        compressed_text = text
        
        try:
            # Try to parse as Python AST
            tree = ast.parse(text)
            
            class ASTCompressor(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    # Remove docstrings
                    if ast.get_docstring(node):
                        node.body = node.body[1:]
                    if not node.body:
                        pass_node = ast.Pass()
                        ast.copy_location(pass_node, node)
                        node.body = [pass_node]
                    # Remove type annotations
                    node.returns = None
                    for arg in node.args.args + getattr(node.args, 'kwonlyargs', []):
                        arg.annotation = None
                    if getattr(node.args, 'vararg', None):
                        node.args.vararg.annotation = None
                    if getattr(node.args, 'kwarg', None):
                        node.args.kwarg.annotation = None
                    return self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    # Remove docstrings
                    if ast.get_docstring(node):
                        node.body = node.body[1:]
                    if not node.body:
                        pass_node = ast.Pass()
                        ast.copy_location(pass_node, node)
                        node.body = [pass_node]
                    return self.generic_visit(node)
                
                def visit_AnnAssign(self, node):
                    # Convert annotated assignments to regular assignments where possible
                    if node.value is not None:
                        new_node = ast.Assign(targets=[node.target], value=node.value)
                        return ast.copy_location(new_node, node)
                    # If no value (e.g. x: int), we remove it entirely
                    return None

                def visit_Module(self, node):
                    # Remove docstrings
                    if ast.get_docstring(node):
                        node.body = node.body[1:]
                    return self.generic_visit(node)
            
            transformer = ASTCompressor()
            new_tree = transformer.visit(tree)
            ast.fix_missing_locations(new_tree)
            compressed_text = ast.unparse(new_tree)
            
        except SyntaxError:
            # Fallback to regex if not valid Python or unparse fails
            # Remove single line comments
            compressed_text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
            
        # Squeeze multiple newlines and empty spaces
        compressed_text = re.sub(r'^\s*$', '', compressed_text, flags=re.MULTILINE)
        compressed_text = re.sub(r'\n{2,}', '\n', compressed_text)
            
        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text.strip(),
            original_size=original_size,
            compressed_size=len(compressed_text.strip()),
            metadata={"strategy": "code_compressor"}
        )
