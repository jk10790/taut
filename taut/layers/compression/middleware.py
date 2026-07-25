import re
from collections.abc import Callable

from taut.core.middleware import Middleware
from taut.core.models import LLMRequest, LLMResponse, PipelineContext, LayerMetrics
from taut.core.config import CompressionConfig

from .detector import ContentDetector
from .registry import CompressionRegistry
from .strategies.json_crusher import SmartCrusher
from .strategies.code_compressor import CodeCompressor
from .strategies.prose_compressor import ProseCompressor

class CompressionMiddleware(Middleware):
    """
    Middleware that applies JSON, Code, Prose compressors sequentially based on detected type.
    """
    name = "compression"

    def __init__(self, config: CompressionConfig):
        self._config = config
        self._detector = ContentDetector()
        self._json_compressor = SmartCrusher()
        self._code_compressor = CodeCompressor()
        self._prose_compressor = ProseCompressor()

    def _estimate_tokens(self, request: LLMRequest) -> int:
        # Simple heuristic for metrics: 4 chars per token
        total = 0
        if getattr(request, 'context', None) and isinstance(request.context, str):
            total += len(request.context) // 4
        if getattr(request, 'messages', None):
            for msg in request.messages:
                if isinstance(msg.content, str):
                    total += len(msg.content) // 4
        return total

    def _apply_strategy(self, text: str, content_type: str, context: PipelineContext) -> str:
        # Check custom registry first
        custom_compressor = CompressionRegistry.get_compressor(content_type)
        if custom_compressor:
            try:
                return custom_compressor(text)
            except Exception as e:
                import logging
                logging.getLogger("taut.compression").warning(f"Custom compressor for {content_type} failed: {e}")
                # Fall back to default logic based on what happens next
                # Here we just return text if custom fails
                return text
                
        if content_type == "json" and self._config.json_enabled:
            return self._json_compressor.compress(text).compressed_text
        elif content_type == "code" and self._config.code_enabled:
            return self._code_compressor.compress(text).compressed_text
        elif content_type == "prose" and self._config.prose != "off":
            return self._prose_compressor.compress(text).compressed_text
        elif content_type == "mixed":
            return self._compress_mixed(text, context)
        return text

    def _compress_mixed(self, text: str, context: PipelineContext) -> str:
        # Split by markdown fences
        parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
        compressed_parts = []
        for part in parts:
            if part.startswith('```') and part.endswith('```'):
                lines = part.split('\n')
                if len(lines) >= 2:
                    fence_start = lines[0]
                    fence_end = lines[-1]
                    inner_code = '\n'.join(lines[1:-1])
                    if inner_code.strip():
                        sub_type = self._detector.detect(inner_code)
                        # Avoid recursing into mixed again to prevent bugs
                        if sub_type != "mixed":
                            inner_code = self._apply_strategy(inner_code, sub_type, context)
                        compressed_parts.append(f"{fence_start}\n{inner_code}\n{fence_end}")
                    else:
                        compressed_parts.append(part)
                else:
                    compressed_parts.append(part)
            else:
                if part.strip():
                    if self._config.prose != "off":
                        compressed_parts.append(self._prose_compressor.compress(part).compressed_text)
                    else:
                        compressed_parts.append(part)
                else:
                    compressed_parts.append(part)
                    
        return "".join(compressed_parts)

    def _inject_schema_instructions(self, request: LLMRequest, context: PipelineContext):
        # Look if JSON compression was applied or if JSON was detected
        if context.content_types_detected.get("context") == "json" or "json" in context.content_types_detected.values():
            schema_note = "Note: Some JSON data has been converted to a columnar table format (COLS: ... ROWS: ...). Please interpret it accordingly."
            if getattr(request, 'messages', None):
                for msg in request.messages:
                    if msg.role == "system" and isinstance(msg.content, str):
                        msg.content += f"\n\n{schema_note}"
                        return
                from taut.core.models import Message
                request.messages.insert(0, Message(role="system", content=schema_note))
            else:
                request.system_prompt = (request.system_prompt or "") + f"\n\n{schema_note}"

    def _record_metrics(self, context: PipelineContext, original_tokens: int, compressed_tokens: int):
        saved = max(0, original_tokens - compressed_tokens)
        context.metrics.layers.append(
            LayerMetrics(
                layer_name=self.name,
                applied=True if saved > 0 else False,
                tokens_before=original_tokens,
                tokens_after=compressed_tokens,
                tokens_saved=saved
            )
        )

    async def before_request(self, request: LLMRequest, context: PipelineContext) -> LLMRequest:
        skip = getattr(self._config, 'skip_for_simple_tier', True)
        if context.selected_tier == "simple" and skip:
            context.compression_applied = False
            return request

        self._transform_request(request, context)
        return request

    async def process(self, request: LLMRequest, context: PipelineContext, next_handler: Callable) -> LLMResponse:
        skip = getattr(self._config, 'skip_for_simple_tier', True)
        if context.selected_tier == "simple" and skip:
            context.compression_applied = False
            return await next_handler(request, context)

        original_tokens = self._estimate_tokens(request)
        
        self._transform_request(request, context)
        
        compressed_tokens = self._estimate_tokens(request)
        context.compression_applied = True
        context.compression_ratio = 1.0 - (compressed_tokens / max(original_tokens, 1))

        self._record_metrics(context, original_tokens, compressed_tokens)
        return await next_handler(request, context)

    def _transform_request(self, request: LLMRequest, context: PipelineContext):
        # Compress context
        if getattr(request, 'context', None) and isinstance(request.context, str):
            if len(request.context) // 4 > getattr(self._config, 'min_compress_tokens', 500):
                ct = self._detector.detect(request.context)
                context.content_types_detected["context"] = ct
                request.context = self._apply_strategy(request.context, ct, context)

        # Compress messages
        if getattr(request, 'messages', None):
            for i, msg in enumerate(request.messages):
                if isinstance(msg.content, str) and (len(msg.content) // 4) > getattr(self._config, 'min_compress_tokens', 500):
                    ct = self._detector.detect(msg.content)
                    context.content_types_detected[f"msg_{i}"] = ct
                    msg.content = self._apply_strategy(msg.content, ct, context)

        if context.selected_tier != "simple":
            self._inject_schema_instructions(request, context)
