"""FastAPI server for taut Proxy Mode."""
from contextlib import asynccontextmanager
import logging
import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import taut
from taut.proxy.routes import router

logger = logging.getLogger("taut.proxy")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the global taut pipeline
    if getattr(app.state, "pipeline", None) is None:
        logger.info("Initializing taut pipeline for Proxy Mode...")
        config = taut.TautConfig.from_env()
        app.state.pipeline = taut.create_pipeline(config)
    yield
    logger.info("Shutting down taut proxy...")

app = FastAPI(
    title="taut Proxy",
    description="OpenAI-compatible AI Efficiency Middleware Proxy",
    version=taut.__version__,
    lifespan=lifespan,
)

cors_origins_env = os.getenv("TAUT_PROXY_CORS_ORIGINS", "*")
cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
if not cors_origins:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Proxy error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "server_error"}},
    )

def start_server(port: int = 8000):
    """Start the proxy server."""
    import uvicorn
    host = os.getenv("TAUT_PROXY_HOST", "127.0.0.1")
    uvicorn.run("taut.proxy.server:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="taut Proxy Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the proxy on")
    args = parser.parse_args()
    start_server(port=args.port)
