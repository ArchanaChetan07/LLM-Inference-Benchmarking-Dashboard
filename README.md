<div align="center">

# LLM Inference Benchmarking Dashboard

**Real-time TTFT · TPOT · ITL · E2EL benchmarking for vLLM**

[![vLLM](https://img.shields.io/badge/vLLM-Inference%20Engine-FF6B35?style=flat-square)](https://github.com/vllm-project/vllm)
[![Prometheus](https://img.shields.io/badge/Prometheus-Live%20Metrics-E6522C?style=flat-square&logo=prometheus&logoColor=white)](https://prometheus.io)
[![Real-time](https://img.shields.io/badge/Streaming-Live%20Charts-2ea44f?style=flat-square)](https://github.com/ArchanaChetan07/LLM-Inference-Benchmarking-Dashboard)
[![GPU](https://img.shields.io/badge/NVIDIA-DCGM%20Metrics-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/dcgm)

*The four metrics that actually tell you if your LLM API is healthy — live, as requests happen.*

</div>

---

## The four metrics that matter

Most teams monitor their LLM API with generic HTTP metrics (response time, error rate). Those tell you *something is wrong*. These four tell you *what* is wrong and *where in the inference pipeline* the problem is.

| Metric | Full name | What it measures | When it degrades |
|---|---|---|---|
| **TTFT** | Time to First Token | How long users wait before seeing any output | Prefill queue backed up; model loading; GPU contention |
| **TPOT** | Time Per Output Token | Speed of the generation stream | GPU compute-bound; context too long; batch too large |
| **ITL** | Inter-Token Latency | Consistency of the token stream | Memory bandwidth; KV cache pressure; batch size variance |
| **E2EL** | End-to-End Latency | Total request time from send to last token | Combination of TTFT + (tokens × TPOT) |

---

## Dashboard layout

```
┌─────────────────────────────────────────────────────┐
│  TTFT P50/P95/P99    │  TPOT P50/P95/P99            │
│  Live line chart     │  Live line chart              │
├──────────────────────┼──────────────────────────────┤
│  ITL distribution    │  E2EL histogram               │
│  Animated bar chart  │  Percentile breakdown         │
├──────────────────────┴──────────────────────────────┤
│  Token throughput (prompt + generation tokens/sec)   │
│  Queue depth (num_requests_running / waiting)        │
├─────────────────────────────────────────────────────┤
│  GPU utilization %   │  GPU memory used/free         │
│  DCGM_FI_DEV_GPU_UTIL│  DCGM_FI_DEV_FB_USED         │
└─────────────────────────────────────────────────────┘
```

---

## Metrics source

All metrics are scraped from vLLM's `/metrics` Prometheus endpoint and NVIDIA DCGM:

```
vLLM metrics (via /metrics)
├── vllm:time_to_first_token_seconds_bucket    → TTFT histogram
├── vllm:time_per_output_token_seconds_bucket  → TPOT histogram
├── vllm:inter_token_latency_seconds_bucket    → ITL histogram
├── vllm:e2e_request_latency_seconds_bucket    → E2EL histogram
├── vllm:generation_tokens_total               → generation throughput
├── vllm:prompt_tokens_total                   → prompt throughput
├── vllm:num_requests_running                  → active requests
└── vllm:num_requests_waiting                  → queue depth

NVIDIA DCGM (via dcgm-exporter)
├── DCGM_FI_DEV_GPU_UTIL      → GPU utilization %
├── DCGM_FI_DEV_FB_USED       → GPU memory used (MiB)
├── DCGM_FI_DEV_FB_FREE       → GPU memory free (MiB)
└── DCGM_FI_DEV_GPU_TEMP      → GPU temperature (°C)
```

---

## How to read the dashboard

**TTFT spiking while TPOT is stable** → prefill bottleneck. Too many long prompts queuing up. Reduce `--max-num-seqs` or scale out engine replicas.

**TPOT degrading while TTFT is stable** → generation bottleneck. GPU compute-bound during decoding. Check GPU utilization — if < 80%, look at batch size and `--max-num-seqs`.

**ITL variance high** → inconsistent generation speed. Usually caused by KV cache pressure (check `vllm:gpu_cache_usage_perc`) or competing requests with very different context lengths.

**E2EL growing linearly with requests** → queue saturation. `num_requests_waiting` will confirm. HPA should scale out — check if it's hitting `maxReplicas`.

---

## Quick start

```bash
git clone https://github.com/ArchanaChetan07/LLM-Inference-Benchmarking-Dashboard
cd LLM-Inference-Benchmarking-Dashboard

# Configure your vLLM endpoint
export VLLM_METRICS_URL=http://localhost:8000/metrics
export PROMETHEUS_URL=http://localhost:9090

# Open dashboard
open index.html
```

For full metrics (DCGM), deploy NVIDIA DCGM Exporter via the GPU Operator or standalone:
```bash
helm install dcgm-exporter nvidia/dcgm-exporter -n monitoring
```

---

## Related projects

Part of a complete vLLM observability suite:

- **[KubeInfer](https://github.com/ArchanaChetan07/KubeInfer)** — production K8s deployment platform (Helm · HPA · RBAC · CI/CD)
- **[KV Cache Profiler](https://github.com/ArchanaChetan07/KV-Cache-Profiler-)** — deep-dive KV cache hit rate and eviction analysis
- **LLM Inference Benchmarking Dashboard** ← you are here

---

## Author

**Archana Suresh Patil** — MLOps & AI Infrastructure Engineer  
MS Data Science · University of San Diego · GPA 3.9  
📬 apatil@sandiego.edu · [LinkedIn](https://linkedin.com/in/archana-suresh-patil-792213245) · [GitHub](https://github.com/ArchanaChetan07)
