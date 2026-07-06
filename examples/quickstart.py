import asyncio
import os
import sys

# Add the parent directory to the Python path so we can import taut
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import taut

async def main():
    print("🚀 Initializing taut pipeline...")
    
    # Create a pipeline with explicit layer configurations
    config = taut.TautConfig(
        provider="litellm",
        cache=taut.SemanticCacheConfig(
            backend="memory",
            similarity_threshold=0.9
        ),
        routing=taut.TieredRoutingConfig(),
        compression=taut.CompressionConfig(
            json=True,
            code=True,
            skip_for_simple_tier=False # Force compression for the demo
        )
    )
    pipeline = taut.create_pipeline(config)
    
    # Set up blocks to demonstrate Prefix Alignment and Compression
    blocks = [
        taut.SystemBlock(content="You are a helpful assistant."),
        taut.ContextBlock(content='[{"id": 1, "status": "active"}, {"id": 2, "status": "inactive"}, {"id": 3, "status": "active"}]'),
        taut.ContextBlock(content='''
def calculate_metrics(data):
    """
    This is a long docstring that the LLM probably doesn't need to read.
    It takes up tokens for no reason.
    """
    total = 0
    # Loop through the data
    for item in data:
        if item.get("status") == "active":
            total += 1
    return total
'''),
        taut.QueryBlock(content="Summarize the active items and analyze the code.")
    ]
    
    request = taut.LLMRequest(
        blocks=blocks,
        model="gpt-4o-mini",
        temperature=0.0,
    )
    
    print("\n📝 Sending first request (Cache Miss)...")
    response1 = await pipeline.run(request)
    print(f"Response: {response1.content[:100]}...")
    if hasattr(response1, "metrics") and response1.metrics:
        print(f"Latency: {getattr(response1.metrics, 'total_latency_ms', 0):.2f}ms")
    
    print("\n📝 Sending identical request (Cache Hit)...")
    response2 = await pipeline.run(request)
    print(f"Response: {response2.content[:100]}...")
    if hasattr(response2, "metrics") and response2.metrics:
        print(f"Latency: {getattr(response2.metrics, 'total_latency_ms', 0):.2f}ms")
    
    print("\n📊 Pipeline Metrics Summary:")
    print(pipeline.metrics.summary())

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY environment variable not set. Provider calls will fail.")
    else:
        asyncio.run(main())
