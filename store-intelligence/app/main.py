"""
main.py — Store Intelligence API entrypoint.

FastAPI application serving the full analytics surface:
  POST /events/ingest         — Accepts up to 500 events per batch
  GET  /stores/{id}/metrics   — Real-time store metrics
  GET  /stores/{id}/funnel    — Conversion funnel
  GET  /stores/{id}/heatmap   — Zone heatmap data
  GET  /stores/{id}/anomalies — Active anomaly detection
  GET  /health                — Service status

Production features:
  - Structured JSON logging with trace_id on every request
  - Graceful degradation on DB unavailability (503 with structured body)
  - Idempotent event ingestion (event_id deduplication)
  - No raw stack traces in error responses
"""

import uuid
import time
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from database import init_db, get_db_health
from ingestion import router as ingest_router
from metrics import router as metrics_router
from funnel import router as funnel_router
from heatmap import router as heatmap_router
from anomalies import router as anomaly_router
from health import router as health_router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
logger = logging.getLogger("store_intelligence")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info('"Starting Store Intelligence API"')
    await init_db()
    logger.info('"Database initialised"')
    yield
    logger.info('"Shutting down Store Intelligence API"')


app = FastAPI(
    title="Store Intelligence API",
    description="Real-time retail analytics from CCTV event streams",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    """Log every request with trace_id, latency, store_id, status_code."""
    trace_id = str(uuid.uuid4())[:8]
    request.state.trace_id = trace_id
    start = time.time()

    # Extract store_id from path if present
    path_parts = request.url.path.split("/")
    store_id = None
    if "stores" in path_parts:
        idx = path_parts.index("stores")
        if idx + 1 < len(path_parts):
            store_id = path_parts[idx + 1]

    try:
        response = await call_next(request)
        latency_ms = int((time.time() - start) * 1000)
        logger.info(
            f'{{"trace_id":"{trace_id}","method":"{request.method}",'
            f'"path":"{request.url.path}","store_id":{repr(store_id)},'
            f'"status_code":{response.status_code},"latency_ms":{latency_ms}}}'
        )
        response.headers["X-Trace-Id"] = trace_id
        return response
    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000)
        logger.error(
            f'{{"trace_id":"{trace_id}","path":"{request.url.path}",'
            f'"error":"{type(exc).__name__}","latency_ms":{latency_ms}}}'
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "trace_id": trace_id,
            },
            headers={"X-Trace-Id": trace_id},
        )


# Mount all routers
app.include_router(ingest_router)
app.include_router(metrics_router)
app.include_router(funnel_router)
app.include_router(heatmap_router)
app.include_router(anomaly_router)
app.include_router(health_router)


@app.get("/")
async def root():
    return {
        "service": "Store Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
