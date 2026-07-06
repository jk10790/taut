"""API routes for the taut proxy."""
import time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from taut.core.models import LLMRequest, Message, ContentBlock

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    try:
        body = await request.json()
        
        messages = body.get("messages", [])
        model = body.get("model", "gpt-4o")
        stream = body.get("stream", False)
        
        taut_messages = []
        system_prompt = None
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system" and not system_prompt:
                system_prompt = content if isinstance(content, str) else str(content)
            else:
                taut_messages.append(Message(
                    role=role, 
                    content=content,
                    name=msg.get("name"),
                ))
        
        intent = "chat"
        if taut_messages:
            last_content = taut_messages[-1].content
            if isinstance(last_content, list):
                # Extract all text parts
                text_parts = [part.get("text", "") for part in last_content if isinstance(part, dict) and part.get("type") == "text"]
                intent = "\n".join(text_parts) if text_parts else "chat"
            elif isinstance(last_content, str):
                intent = last_content
                
        req = LLMRequest(
            intent=intent,
            messages=taut_messages,
            model=model,
            system_prompt=system_prompt,
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            response_format=body.get("response_format"),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            top_p=body.get("top_p"),
            stop=body.get("stop"),
            namespace=request.headers.get("X-Taut-Namespace"),
        )
        
        pipeline = request.app.state.pipeline
        
        if stream:
            return StreamingResponse(
                _stream_generator(pipeline, req, model),
                media_type="text/event-stream"
            )
        else:
            response = await pipeline.run(req)
            
            message_obj = {"role": "assistant"}
            if response.content is not None:
                message_obj["content"] = response.content
            if getattr(response, "tool_calls", None):
                message_obj["tool_calls"] = response.tool_calls
                
            return {
                "id": f"chatcmpl-{getattr(response.metrics, 'request_id', 'req')}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": response.model,
                "choices": [{
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": getattr(response, "finish_reason", None) or "stop"
                }],
                "usage": {
                    "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                    "completion_tokens": getattr(response.usage, "output_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                },
                "taut_metrics": response.metrics.model_dump() if hasattr(response.metrics, "model_dump") else {}
            }
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "api_error"}}
        )

async def _stream_generator(pipeline, req: LLMRequest, original_model: str):
    """Generator for streaming responses in OpenAI SSE format."""
    import json
    
    # We call pipeline.stream() which yields chunks
    async for chunk in pipeline.stream(req):
        data = {
            "id": f"chatcmpl-{req.metadata.get('request_id', 'stream')}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": original_model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(data)}\n\n"
        
    yield "data: [DONE]\n\n"

@router.get("/v1/taut/metrics")
async def get_metrics(request: Request):
    """Endpoint to fetch taut observability metrics."""
    return request.app.state.pipeline.metrics.to_dict()
