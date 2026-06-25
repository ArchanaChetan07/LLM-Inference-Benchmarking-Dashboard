# LLM Inference Benchmarking Dashboard

> Real-time benchmarking dashboard for `vllm bench serve` — streams TTFT, TPOT, ITL, and E2EL live across multiple concurrency levels, plots results as they arrive, and exports to CSV/JSON.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![WebSocket](https://img.shields.io/badge/WebSocket-live%20stream-6366F1)](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)

---

## What it does

Running `vllm bench serve` at multiple concurrency levels is the standard way to characterise an LLM serving stack — but the output is raw terminal text and the data goes nowhere. This dashboard:

- **Orchestrates** `vllm bench serve` across any set of concurrency levels via an async subprocess pool
- **Streams** every result live over WebSocket the moment a level completes
- **Plots** TTFT / TPOT / ITL / E2EL / Throughput as animated multi-series line charts, building point-by-point
- **Compares** the current run against the previous one (overlaid in ghost white)
- **Exports** full results to CSV or JSON with one click
- **Exposes** Prometheus metrics for Grafana dashboards
- **Runs in simulation mode** with a realistic MI300X latency model — no vLLM installation required to try it

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  dashboard/index.html   (static — open directly in browser)      │
│  WebSocket client → receives RunEvents, updates charts live      │
└───────────────────────────┬──────────────────────────────────────┘
                            │ ws://localhost:8080/ws/{run_id}
┌───────────────────────────▼──────────────────────────────────────┐
│  FastAPI Server  (backend/server.py)                              │
│                                                                  │
│  POST /api/runs          → start benchmark run                   │
│  GET  /api/runs/{id}     → fetch results                         │
│  POST /api/runs/{id}/cancel                                      │
│  GET  /metrics           → Prometheus                            │
│  WS   /ws/{run_id}       → live RunEvent stream                  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  BenchmarkEngine  (backend/engine.py)                             │
│                                                                  │
│  asyncio.create_task(_orchestrate)                               │
│  └── for each concurrency level:                                 │
│       asyncio.create_subprocess_exec("vllm", "bench", "serve")  │
│       parse stdout → LevelResult                                 │
│       asyncio.Queue.put(RunEvent)                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option 1 — Open the dashboard directly (simulation mode)

```bash
git clone https://github.com/yourusername/llm-benchmark-dashboard
# Open dashboard/index.html in your browser — no server needed
```

The dashboard includes a built-in simulation engine using a realistic MI300X latency model. Select concurrency levels, click **Run Benchmark**, and watch results stream in.

### Option 2 — Full Python server

```bash
pip install -r requirements.txt

# Simulation mode (no vLLM required)
uvicorn backend.server:app --host 0.0.0.0 --port 8080

# Real vLLM mode — must have vLLM running on port 8000
VLLM_REAL=1 uvicorn backend.server:app --host 0.0.0.0 --port 8080
```

Open `dashboard/index.html` — it auto-connects to `localhost:8080`.

### Option 3 — Docker Compose (server + Prometheus + Grafana)

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| Benchmark API | http://localhost:8080 |
| Prometheus | http://localhost:9090 |
| Grafana (admin/admin) | http://localhost:3000 |

---

## Connecting to vLLM

Start your vLLM server first:

```bash
export MODEL_ID="openai/gpt-oss-120b"
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1

vllm serve $MODEL_ID \
    --port 8000 \
    --tensor-parallel 8 \
    --compilation-config '{"full_cuda_graph": true}'
```

Then start the dashboard with `VLLM_REAL=1`:

```bash
VLLM_REAL=1 uvicorn backend.server:app --port 8080
```

The engine will invoke `vllm bench serve` as a subprocess for each concurrency level and parse its stdout.

---

## Metrics (TTFT / TPOT / ITL / E2EL)

| Metric | Full Name | What it measures |
|---|---|---|
| **TTFT** | Time to First Token | Latency from request submission to first output token — the dominant factor in perceived responsiveness |
| **TPOT** | Time Per Output Token | Average time to generate each subsequent token — determines streaming speed |
| **ITL** | Inter-Token Latency | Time between consecutive tokens — variance here causes choppy streaming |
| **E2EL** | End-to-End Latency | Total time from HTTP request to last response byte — the SLA metric |

Each metric is measured at **p50**, **p90**, and **p99** across the `num_prompts = 10 × concurrency` requests per level.

---

## Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `bench_runs_total` | Counter | Total benchmark runs started |
| `bench_levels_total` | Counter | Total concurrency levels measured |
| `bench_inference_requests_total` | Counter | Inference requests sent (by model, status) |
| `bench_active_runs` | Gauge | Currently active runs |
| `bench_best_ttft_p50_ms` | Gauge | Best TTFT p50 seen per model |
| `bench_best_throughput_tok_s` | Gauge | Best token throughput seen per model |
| `bench_ttft_p50_ms` | Histogram | Distribution of TTFT p50 across levels |
| `bench_tpot_p50_ms` | Histogram | Distribution of TPOT p50 across levels |
| `bench_throughput_req_s` | Histogram | Distribution of request throughput |

---

## REST API

```bash
# Start a benchmark run
curl -X POST localhost:8080/api/runs \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-120b","concurrency_levels":[8,16,32,64,128],"input_len":4096,"output_len":1024}'

# Stream results via WebSocket
wscat -c ws://localhost:8080/ws/{run_id}

# Fetch completed results
curl localhost:8080/api/runs/{run_id}

# Cancel a running benchmark
curl -X POST localhost:8080/api/runs/{run_id}/cancel

# Prometheus metrics
curl localhost:8080/metrics
```

---

## Testing

```bash
pytest tests/ -v --asyncio-mode=auto
# 20+ tests: engine state, event sequence, result correctness,
# vLLM output parser, concurrent runs, p99 > p50 invariant
```

---

## Project Structure

```
llm-benchmark-dashboard/
├── backend/
│   ├── engine.py        # BenchmarkEngine — async orchestration, subprocess, parser
│   ├── server.py        # FastAPI + WebSocket server
│   └── metrics.py       # Prometheus exporter wrapping the engine
├── dashboard/
│   └── index.html       # Standalone live dashboard — no build step
├── dashboards/
│   └── grafana-benchmark.json   # Grafana dashboard — import directly
├── configs/
│   ├── prometheus.yml
│   ├── grafana-datasource.yml
│   └── grafana-dashboard.yml
├── tests/
│   └── test_engine.py   # 20+ async unit tests
├── .github/
│   └── workflows/ci.yml # CI: test × 3 Python versions + lint + Docker build
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Hardware Context

Built and tuned for AMD MI300X (192GB HBM3) with:
- `VLLM_ROCM_USE_AITER=1`
- `VLLM_USE_AITER_UNIFIED_ATTENTION=1`
- `--compilation-config '{"full_cuda_graph": true}'`

The simulation mode uses MI300X-calibrated latency coefficients. Works on any GPU supported by vLLM.
