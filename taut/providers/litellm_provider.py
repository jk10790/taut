import litellm
from typing import AsyncIterator, Any
from taut.providers.base import BaseProvider
from taut.core.models import LLMRequest, LLMResponse, PipelineContext, TokenUsage

class LiteLLMProvider(BaseProvider):
    """LiteLLM integration for unified model access."""
    
    def __init__(self, default_model: str = "gpt-3.5-turbo", api_key: str | None = None, base_url: str | None = None, num_retries: int = 2, timeout: float = 60.0, fallback_models: list[str] | None = None):
        self.default_model = default_model
        self.api_key = api_key
        self.base_url = base_url
        self.num_retries = num_retries
        self.timeout = timeout
        self.fallback_models = fallback_models
    
    async def complete(self, request: LLMRequest, context: PipelineContext) -> LLMResponse:
        """Execute a completion request using litellm."""
        messages = self._build_messages(request)
        
        # Determine model
        model = request.model or self.default_model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "num_retries": self.num_retries,
            "timeout": self.timeout,
        }
        if self.fallback_models:
            kwargs["fallbacks"] = [{"model": m} for m in self.fallback_models]

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if getattr(request, "tools", None) is not None:
            kwargs["tools"] = request.tools
        if getattr(request, "tool_choice", None) is not None:
            kwargs["tool_choice"] = request.tool_choice
        if getattr(request, "response_format", None) is not None:
            kwargs["response_format"] = request.response_format
        if getattr(request, "top_p", None) is not None:
            kwargs["top_p"] = request.top_p
        if getattr(request, "stop", None) is not None:
            kwargs["stop"] = request.stop
            
        try:
            if self.fallback_models:
                base_params = {"api_key": self.api_key} if self.api_key else {}
                if self.base_url: base_params["api_base"] = self.base_url
                model_list = [{"model_name": model, "litellm_params": {"model": model, **base_params}}]
                model_list += [{"model_name": m, "litellm_params": {"model": m, **base_params}} for m in self.fallback_models]
                router = litellm.Router(model_list=model_list)
                response = await router.acompletion(**kwargs)
            else:
                response = await litellm.acompletion(**kwargs)
        except Exception as e:
            raise RuntimeError(f"litellm.acompletion failed: {e}") from e
        
        content = response.choices[0].message.content or ""
        usage = response.usage
        
        # Use litellm token counter
        try:
            prompt_tokens = litellm.token_counter(model=model, messages=messages)
        except Exception:
            prompt_tokens = usage.prompt_tokens if usage else 0
            
        completion_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            content=content,
            model=response.model if hasattr(response, "model") else model,
            usage=TokenUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cached_tokens=(getattr(usage.prompt_tokens_details, 'cached_tokens', 0) if not isinstance(getattr(usage, 'prompt_tokens_details', None), dict) else getattr(usage, 'prompt_tokens_details', {}).get('cached_tokens', 0)) if getattr(usage, 'prompt_tokens_details', None) else 0,
            ),
            finish_reason=response.choices[0].finish_reason if response.choices else None,
            tool_calls=response.choices[0].message.tool_calls if response.choices else None,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
        )

    async def complete_stream(self, request: LLMRequest, context: PipelineContext) -> AsyncIterator[str]:
        """Execute a streaming completion request using litellm."""
        messages = self._build_messages(request)
        
        model = request.model or self.default_model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "num_retries": self.num_retries,
            "timeout": self.timeout,
        }
        if self.fallback_models:
            kwargs["fallbacks"] = [{"model": m} for m in self.fallback_models]

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if getattr(request, "tools", None) is not None:
            kwargs["tools"] = request.tools
        if getattr(request, "tool_choice", None) is not None:
            kwargs["tool_choice"] = request.tool_choice
        if getattr(request, "response_format", None) is not None:
            kwargs["response_format"] = request.response_format
        if getattr(request, "top_p", None) is not None:
            kwargs["top_p"] = request.top_p
        if getattr(request, "stop", None) is not None:
            kwargs["stop"] = request.stop

        if self.fallback_models:
            base_params = {"api_key": self.api_key} if self.api_key else {}
            if self.base_url: base_params["api_base"] = self.base_url
            model_list = [{"model_name": model, "litellm_params": {"model": model, **base_params}}]
            model_list += [{"model_name": m, "litellm_params": {"model": m, **base_params}} for m in self.fallback_models]
            router = litellm.Router(model_list=model_list)
            response = await router.acompletion(**kwargs)
        else:
            response = await litellm.acompletion(**kwargs)
        
        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content is not None:
                    yield delta.content

    def _build_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        """Construct the message list for the API call."""
        litellm_messages: list[dict[str, str]] = []
        
        if request.system_prompt:
            litellm_messages.append({"role": "system", "content": request.system_prompt})
            
        if request.messages:
            for msg in request.messages:
                msg_dict = {}
                if isinstance(msg, dict):
                    msg_dict = msg.copy()
                else:
                    msg_dict = {"role": getattr(msg, "role", "user"), "content": getattr(msg, "content", "")}
                    if getattr(msg, "name", None):
                        msg_dict["name"] = msg.name
                    if getattr(msg, "tool_call_id", None):
                        msg_dict["tool_call_id"] = msg.tool_call_id
                    if getattr(msg, "tool_calls", None):
                        msg_dict["tool_calls"] = msg.tool_calls
                litellm_messages.append(msg_dict)
        else:
            user_content = request.intent
            if request.context:
                user_content += f"\n\nContext:\n{request.context}"
            litellm_messages.append({"role": "user", "content": user_content})
            
        return litellm_messages
