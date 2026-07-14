# LLM Inference Benchmarking Dashboard

### Real-time control plane for **vLLM `bench serve`** concurrency sweeps — TTFT · TPOT · ITL · E2EL · throughput streamed over WebSockets, exported to Prometheus / Grafana

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-bench%20serve-111111?style=for-the-badge" />
  <img alt="Prometheus" src="https://img.shields.io/badge/Prometheus-v2.51-E6522C?style=for-the-badge&logo=prometheus&logoColor=white" />
  <img alt="Grafana" src="https://img.shields.io/badge/Grafana-10.4-F46800?style=for-the-badge&logo=grafana&logoColor=white" />
</p>

<p align="center">
  <img alt="Metrics" src="https://img.shields.io/badge/Latency-TTFT%20%7C%20TPOT%20%7C%20ITL%20%7C%20E2EL-0B3D91" />
  <img alt="Concurrency" src="https://img.shields.io/badge/Default%20sweep-8%2C16%2C32%2C64%2C128-6f42c1" />
  <img alt="IO" src="https://img.shields.io/badge/Default%20tokens-4096%20in%20%2F%201024%20out-2088FF" />
  <a href="tests/test_engine.py"><img alt="pytest" src="https://img.shields.io/badge/pytest-21%20tests-0A7A0A" /></a>
  <a href="docker-compose.yml"><img alt="Docker" src="https://img.shields.io/badge/Docker%20Compose-API%20%2B%20Prom%20%2B%20Grafana-2496ED?logo=docker&logoColor=white" /></a>
</p>

---

## Why this project

Measuring LLM serving quality under load usually means ad-hoc shell scripts and screenshots. This repo is a **portfolio-grade MLOps / inference engineering** stack that:

- Orchestrates **concurrency sweeps** against `vllm bench serve` (or a faithful local simulator)
- Streams **percentile latency + throughput** live to a browser dashboard via **WebSockets**
- Exposes **Prometheus** gauges/histograms and a provisioned **Grafana** board
- Ships **21 pytest** cases covering orchestration, scaling invariants, and stdout parsing

Ideal signal for roles in **ML Platform**, **Inference Engineering**, **AI Infra**, and **Observability**.

> Results provenance matters: simulator output and parser fixtures are useful for testing, but they are not hardware measurements.

---

## Results

### Real/live benchmark evidence

**No real GPU benchmark result is committed yet.** The tracked tree and full commit history contain no `results/` artifact, exported API response, benchmark log, screenshot, or other evidence of a completed dashboard-driven `VLLM_REAL=1` run. Consequently, this README does not claim measured latency or throughput for MI300X or for any catalog model.

The six entries returned by `/api/models` are selectable presets, not evidence that those models were benchmarked. Likewise, the default `openai/gpt-oss-120b` / MI300X labels describe configuration defaults, not a completed hardware run.

### Configurations exercised by committed tests

These runs execute the in-process **simulator** (`VLLM_REAL=0`, the default), not vLLM or a GPU:

| Test purpose | Model label | Concurrency | Input / output tokens | Evidence |
|--------------|-------------|-------------|-----------------------|----------|
| Full simulated run | `test/model-7b` | `4, 8` | `512 / 128` | `config` fixture and `TestSimulatedRun` |
| Scaling checks | `openai/gpt-oss-120b` | `8, 128` | `4096 / 1024` | TTFT and throughput invariant tests |
| Concurrent-run isolation | `model-A`, `model-B` | `4`, `8` | `4096 / 1024` | `TestConcurrentRuns` |
| Result isolation | `openai/gpt-oss-120b` | `4`, `128` | `4096 / 1024` | `test_results_not_cross_contaminated` |

The default sweep `[8, 16, 32, 64, 128]` is verified as a configuration value, but no committed test completes that full sweep. The cancellation test creates it and stops it before results are produced.

### Parser fixture — not a measured repository result

`tests/test_engine.py` includes a hand-authored sample of vLLM-like stdout to test parsing. Its exact values are:

| Fixture field | Value |
|---------------|-------|
| Successful requests | **80** |
| Benchmark duration | **12.34 s** |
| Request / output-token throughput | **6.48 req/s** / **6644.32 tok/s** |
| TTFT P50 / P90 / P99 | **82.10 / 141.00 / 201.55 ms** |
| TPOT P50 / P90 / P99 | **10.80 / 14.20 / 18.50 ms** |
| ITL P50 / P99 | **11.50 / 27.30 ms** |
| E2EL P50 / P99 | **11.18 / 18.44 s** |
| Input / output tokens | **327,680 / 81,920** |

These numbers validate `_parse_vllm_output`; the fixture records no model, GPU, server, command, timestamp, or raw artifact provenance and therefore must not be presented as a live benchmark.

### Producing publishable real results

1. Install vLLM separately, start an OpenAI-compatible serving endpoint, and install this repository's dependencies in an isolated environment.
2. Set `VLLM_REAL=1` **before** starting the API, then launch `uvicorn backend.server:app --host 0.0.0.0 --port 8080`.
3. Start a run from the dashboard or `POST /api/runs` with the actual model, endpoint, token lengths, concurrency levels, and GPU label.
4. After completion, export `GET /api/runs/{run_id}` to a timestamped JSON file. Record the vLLM version, model revision, GPU count/type, server flags, and command alongside it.
5. Commit that immutable artifact before adding its exact measured values here. Run history is currently memory-only, so it is lost when the API process exits.

### Stack footprint

| Fact | Value | Source |
|------|--------|--------|
| Tracked files | **18** | git tree |
| Languages (bytes) | HTML **52,232** · Python **31,968** · Dockerfile **539** | GitHub API |
| pytest cases | **21** | `tests/test_engine.py` |
| Grafana panels | **9** | `dashboards/grafana-benchmark.json` |
| Prometheus scrape | **every 5s** · retention **7d** | compose + `prometheus.yml` |
| Ports | API **8080** · Prometheus **9090** · Grafana **3000** | compose / Dockerfile |

```mermaid
%%{init: {'theme':'base'}}%%
pie showData title Language composition (bytes)
    "HTML" : 52232
    "Python" : 31968
    "Dockerfile" : 539
```

---

## Architecture

```mermaid
flowchart TB
  subgraph UI["Frontend"]
    HTML["dashboard/index.html<br/>Chart.js KPIs + live log"]
  end

  subgraph API["FastAPI · :8080"]
    REST["REST /api/runs · /api/models · /api/health"]
    WS["WebSocket /ws/{run_id}"]
    PROM["GET /metrics"]
  end

  subgraph Engine["BenchmarkEngine"]
    ORCH["Orchestrate concurrency loop"]
    SIM["Simulator VLLM_REAL=0"]
    REAL["subprocess: vllm bench serve<br/>VLLM_REAL=1"]
    PARSE["_parse_vllm_output()"]
  end

  subgraph Obs["Observability"]
    P["Prometheus :9090"]
    G["Grafana :3000<br/>9-panel board"]
  end

  HTML -->|POST start| REST
  REST --> ORCH
  ORCH --> SIM
  ORCH --> REAL --> PARSE
  ORCH -->|RunEvent stream| WS --> HTML
  ORCH --> PROM --> P --> G
```

```mermaid
sequenceDiagram
  participant UI as Dashboard
  participant API as FastAPI
  participant E as BenchmarkEngine
  participant V as vLLM bench serve
  participant Prom as Prometheus
  UI->>API: POST /api/runs {levels, in/out len, model}
  API->>E: start(config) → run_id
  API-->>UI: {run_id, ws_url}
  UI->>API: WS /ws/{run_id}
  loop each concurrency
    E-->>UI: level_started {num_prompts=c×10}
    alt VLLM_REAL=1
      E->>V: bench serve --max-concurrency c
      V-->>E: stdout percentiles
    else simulation
      E-->>E: MI300X latency model + sleep
    end
    E-->>UI: level_result {TTFT/TPOT/ITL/E2EL, tok/s}
  end
  E-->>UI: run_complete
  Prom->>API: scrape /metrics every 5s
```

```mermaid
flowchart LR
  subgraph LevelResult["Per-level metric schema"]
    A[ttft_p50/p90/p99]
    B[tpot_p50/p90/p99]
    C[itl_p50/p99]
    D[e2el_p50/p99]
    E[throughput_req_s / tok_s]
    F[success / fail counts]
  end
```

---

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/runs` | Start sweep → `{run_id, ws_url}` |
| `GET` | `/api/runs` | List runs + stored `LevelResult`s |
| `GET` | `/api/runs/{id}` | Fetch one run |
| `POST` | `/api/runs/{id}/cancel` | Cooperative cancel |
| `GET` | `/api/models` | 6 model presets |
| `GET` | `/api/health` | Liveness · version `1.0.0` |
| `GET` | `/metrics` | Prometheus text exposition |
| `WS` | `/ws/{run_id}` | Live `RunEvent` stream |

**WebSocket event kinds:** `run_started` · `level_started` · `level_result` · `run_complete` · `run_error` · `run_cancelled`

**Catalog models:** GPT-OSS 120B · Llama-3 70B · Llama-3 8B · Mistral-7B · Phi-3-mini · Gemma-2 27B

---

## Observability

Prometheus metrics (selected):

| Metric | Type | Role |
|--------|------|------|
| `bench_runs_total` | Counter | Runs started |
| `bench_levels_total` | Counter | Levels measured |
| `bench_inference_requests_total` | Counter | ok / failed |
| `bench_active_runs` | Gauge | In-flight runs |
| `bench_best_ttft_p50_ms` | Gauge | Best TTFT p50 by model |
| `bench_best_throughput_tok_s` | Gauge | Best tok/s by model |
| `bench_ttft_p50_ms` / `bench_tpot_p50_ms` / `bench_throughput_req_s` | Histograms | Distribution across levels |

Grafana dashboard `llm-bench-v1` (9 panels): Total Runs · Best TTFT p50 · Best Throughput · Active Runs · Total Inference Requests · TTFT timeseries · Throughput timeseries · TTFT heatmap · TPOT timeseries. Refresh **10s**. Tags: `vllm`, `llm`, `benchmark`, `inference`, `mi300x`.

---

## Repository layout

```text
LLM-Inference-Benchmarking-Dashboard/
├── backend/
│   ├── engine.py          # BenchmarkEngine · LevelResult · vLLM parse/sim
│   ├── metrics.py         # Prometheus MetricsExporter
│   └── server.py          # FastAPI REST + WebSocket
├── dashboard/index.html   # Live UI (Chart.js)
├── dashboards/grafana-benchmark.json
├── configs/               # prometheus.yml · grafana provisioning
├── tests/test_engine.py   # 21 pytest cases
├── Dockerfile             # python:3.11-slim · uvicorn :8080
├── docker-compose.yml     # api + prometheus + grafana
└── requirements.txt
```

---

## Tech stack & keywords

| Layer | Technology |
|-------|------------|
| API | **FastAPI 0.115**, Uvicorn, Pydantic v2, WebSockets |
| Inference bench | **vLLM** `bench serve` (`VLLM_REAL=1`) |
| Metrics | **prometheus-client**, Prometheus **v2.51**, Grafana **10.4** |
| Frontend | Static HTML dashboard · live charts |
| Runtime | **Docker** / Compose · Python **3.11** |
| Quality | **pytest** + **pytest-asyncio** · GitHub Actions |

**Keyword surface:** Python · FastAPI · WebSockets · vLLM · LLM inference · TTFT · TPOT · ITL · E2EL · throughput · concurrency sweep · Prometheus · Grafana · observability · MLOps · GPU serving · MI300X · Docker · pytest · CI/CD · system design

---

## Quickstart

```bash
git clone https://github.com/ArchanaChetan07/LLM-Inference-Benchmarking-Dashboard.git
cd LLM-Inference-Benchmarking-Dashboard

# Full stack (API + Prometheus + Grafana)
docker compose up --build -d
# API http://localhost:8080/api/health
# Grafana http://localhost:3000  (admin/admin)
# Prometheus http://localhost:9090
# Open dashboard/index.html in a browser

# Local API only (use an isolated environment)
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.server:app --host 0.0.0.0 --port 8080

# Real vLLM benches (requires vllm on PATH + reachable serve backend)
VLLM_REAL=1 uvicorn backend.server:app --port 8080

pytest tests/ -v --asyncio-mode=auto
```

Example start payload:

```bash
curl -X POST http://localhost:8080/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-120b","concurrency_levels":[8,16,32,64,128],"input_len":4096,"output_len":1024,"gpu_type":"MI300X"}'
```

---

## Testing

| Suite | Coverage |
|-------|----------|
| Config | Default levels `[8,16,32,64,128]`, 8-char `run_id` |
| Engine | Register / cancel / event ordering / multi-run isolation |
| Scaling invariants | TTFT↑ & throughput↑ vs concurrency; P99≥P50; success+fail=total; prompts=`c×10` |
| Parser | Fixture TTFT/TPOT/E2EL/throughput asserts; missing field → 0 |

```mermaid
flowchart LR
  Push[git push] --> GHA[GitHub Actions]
  GHA --> Py[Python + deps]
  Py --> Test[pytest --asyncio-mode=auto]
```

---

## Roadmap

- Check in a real-GPU `results/*.json` artifact from `VLLM_REAL=1` for portfolio latency tables  
- Surface DCGM / GPU utilization next to TTFT panels  
- Persist run history to SQLite/Postgres for multi-session compare  

---

<p align="center">
  <b>LLM Inference Benchmarking Dashboard</b><br/>
  <a href="https://github.com/ArchanaChetan07/LLM-Inference-Benchmarking-Dashboard">github.com/ArchanaChetan07/LLM-Inference-Benchmarking-Dashboard</a>
</p>
