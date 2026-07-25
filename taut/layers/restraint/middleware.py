"""Middleware for output restraint."""
import time
from collections.abc import Awaitable, Callable
from taut.core.middleware import Middleware
from taut.core.models import LLMRequest, LLMResponse, PipelineContext, LayerMetrics, Message
from taut.core.config import OutputRestraintConfig
import litellm

class OutputRestraintMiddleware(Middleware):
    """Injects policy instructions and applies max_tokens/format caps."""
    
    @property
    def name(self) -> str:
        return "output_restraint"
        
    def __init__(self, config: OutputRestraintConfig) -> None:
        self.config = config
        from .policies.adaptive import AdaptivePolicy
        from .policies.structured import StructuredPolicy
        from .policies.terse import TersePolicy
        from .policies.yagni import YAGNIPolicy

        self.policies = {
            "yagni": YAGNIPolicy(config),
            "terse": TersePolicy(config),
            "structured": StructuredPolicy(config),
            "adaptive": AdaptivePolicy(config),
        }
        # "off" is a valid policy value and must genuinely disable the layer.
        # Previously both "off" and "adaptive" fell through the dict lookup's
        # default and silently became YAGNI -- so restraint could not be turned
        # off, and the tier-aware policy was unreachable.
        self.enabled = config.policy != "off"
        if config.policy not in self.policies and self.enabled:
            raise ValueError(
                f"Unknown output restraint policy {config.policy!r}. "
                f"Expected one of: off, {', '.join(sorted(self.policies))}"
            )
        self.default_policy = self.policies.get(config.policy) if self.enabled else None
        
    async def process(
        self,
        request: LLMRequest,
        context: PipelineContext,
        next_handler: Callable[[LLMRequest, PipelineContext], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        start_time = time.time()

        if not self.enabled:
            context.metrics.layers.append(
                LayerMetrics(
                    layer_name=self.name,
                    latency_ms=(time.time() - start_time) * 1000,
                    applied=False,
                    details={"policy": "off"},
                )
            )
            return await next_handler(request, context)

        # Decide which policy to apply
        policy_name = request.metadata.get("restraint_policy")
        policy = self.default_policy
        if policy_name:
            policy = self.policies.get(policy_name, self.default_policy)

        # Context is passed through so tier-aware policies (adaptive) can see
        # which routing tier was selected upstream.
        instructions = policy.get_instructions(context)
        max_tokens = policy.get_max_tokens(context)

        for constraint in self.config.custom_constraints:
            instructions = f"{instructions}\n{constraint}" if instructions else constraint

        provider = context.provider_name
        if not provider and (context.selected_model or request.model):
            try:
                # litellm.get_llm_provider returns a tuple
                provider_info = litellm.get_llm_provider(context.selected_model or request.model)
                if provider_info and len(provider_info) > 1:
                    provider = provider_info[1]
            except Exception:
                provider = "openai"

        response_format = policy.get_response_format(provider, context)

        applied = False

        # Inject instructions. An empty instruction string is a deliberate
        # signal from a policy (adaptive returns "" for the complex tier, where
        # terseness would cost more in reasoning quality than it saves).
        if not instructions:
            pass
        elif request.system_prompt:
            request.system_prompt += "\n\n" + instructions
            applied = True
        elif request.messages:
            system_msg = next((m for m in request.messages if m.role == "system"), None)
            if system_msg and isinstance(system_msg.content, str):
                system_msg.content += "\n\n" + instructions
                applied = True
            else:
                request.messages.insert(0, Message(role="system", content=instructions))
                applied = True
        else:
            request.system_prompt = instructions
            applied = True
            
        # Apply token caps
        if max_tokens:
            if request.max_tokens is None or request.max_tokens > max_tokens:
                request.max_tokens = max_tokens
                applied = True
                
        # Apply format
        if response_format and request.response_format is None:
            request.response_format = response_format
            applied = True
            
        latency = (time.time() - start_time) * 1000
        
        metrics = LayerMetrics(
            layer_name=self.name,
            latency_ms=latency,
            applied=applied,
            details={"policy": policy.name}
        )
        context.metrics.layers.append(metrics)
        
        return await next_handler(request, context)
