"""
backend/server.py

FastAPI application.
- POST /api/runs          → start a benchmark run
- GET  /api/runs          → list all runs + results
- GET  /api/runs/{id}     → single run results
- POST /api/runs/{id}/cancel
- GET  /api/models        → supported model list
- GET  /api/health
- WS   /ws/{run_id}       → live event stream for a run
- GET  /metrics           → Prometheus text format
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from backend.engine import BenchmarkEngine, BenchmarkConfig
from backend.metrics import MetricsExporter


engine = MetricsExporter(BenchmarkEngine())


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="LLM Benchmark Dashboard",
    description="Real-time vllm bench serve orchestration and visualization",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schemas ───────────────────────────────────────────────────
class StartRunRequest(BaseModel):
    model: str = "openai/gpt-oss-120b"
    concurrency_levels: list[int] = Field(default=[8, 16, 32, 64, 128])
    input_len: int = Field(default=4096, ge=128, le=32768)
    output_len: int = Field(default=1024, ge=64, le=8192)
    backend_url: str = "http://localhost:8000"
    gpu_type: str = "MI300X"


# ── REST endpoints ────────────────────────────────────────────────────
@app.post("/api/runs", status_code=201)
async def start_run(req: StartRunRequest):
    cfg = BenchmarkConfig(**req.model_dump())
    run_id = await engine.start(cfg)
    return {
        "run_id": run_id,
        "status": "started",
        "model": cfg.model,
        "concurrency_levels": cfg.concurrency_levels,
        "ws_url": f"/ws/{run_id}",
    }


@app.get("/api/runs")
async def list_runs():
    return {
        "runs": [
            {
                "run_id": rid,
                "active": rid in engine.active_runs(),
                "result_count": len(engine.results(rid)),
                "results": engine.results(rid),
            }
            for rid in engine._results.keys()
        ]
    }


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    results = engine.results(run_id)
    if run_id not in engine._results:
        raise HTTPException(404, detail=f"Run {run_id!r} not found")
    return {
        "run_id": run_id,
        "active": run_id in engine.active_runs(),
        "results": results,
    }


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    if run_id not in engine._results:
        raise HTTPException(404, detail=f"Run {run_id!r} not found")
    engine.cancel(run_id)
    return {"run_id": run_id, "status": "cancelled"}


@app.get("/api/models")
async def list_models():
    return {
        "models": [
            {"id": "openai/gpt-oss-120b",       "params": "120B", "family": "GPT"},
            {"id": "meta-llama/Llama-3-70b",     "params": "70B",  "family": "Llama"},
            {"id": "meta-llama/Llama-3-8b",      "params": "8B",   "family": "Llama"},
            {"id": "mistralai/Mistral-7B-v0.3",  "params": "7B",   "family": "Mistral"},
            {"id": "microsoft/Phi-3-mini-4k",    "params": "3.8B", "family": "Phi"},
            {"id": "google/gemma-2-27b",         "params": "27B",  "family": "Gemma"},
        ]
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "active_runs": len(engine.active_runs()),
        "total_runs": len(engine._results),
        "real_vllm": False,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    return engine.generate_prometheus()


# ── WebSocket ─────────────────────────────────────────────────────────
@app.websocket("/ws/{run_id}")
async def run_websocket(websocket: WebSocket, run_id: str):
    await websocket.accept()

    # If run doesn't exist yet, wait briefly for it to be registered
    for _ in range(20):
        if run_id in engine._runs:
            break
        await asyncio.sleep(0.1)

    if run_id not in engine._runs:
        await websocket.send_json({"kind": "run_error", "run_id": run_id, "payload": {"error": "Run not found"}})
        await websocket.close()
        return

    try:
        async for event in engine.subscribe(run_id):
            await websocket.send_json({
                "kind": event.kind,
                "run_id": event.run_id,
                "payload": event.payload,
            })
            if event.kind in ("run_complete", "run_error", "run_cancelled"):
                break
    except WebSocketDisconnect:
        engine.cancel(run_id)
