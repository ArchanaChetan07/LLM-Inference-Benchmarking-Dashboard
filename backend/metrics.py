"""
backend/metrics.py

Wraps BenchmarkEngine and exposes Prometheus metrics.
Also acts as a thin proxy so server.py imports only this.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from backend.engine import BenchmarkEngine, BenchmarkConfig, LevelResult


class MetricsExporter(BenchmarkEngine):
    """BenchmarkEngine + Prometheus instrumentation."""

    def __init__(self):
        super().__init__()
        self._registry = CollectorRegistry()

        # ── Counters ──────────────────────────────────────────────────
        self.runs_total = Counter(
            "bench_runs_total",
            "Total benchmark runs started",
            registry=self._registry,
        )
        self.levels_total = Counter(
            "bench_levels_total",
            "Total concurrency levels measured",
            ["model"],
            registry=self._registry,
        )
        self.requests_total = Counter(
            "bench_inference_requests_total",
            "Total inference requests sent across all benchmarks",
            ["model", "status"],
            registry=self._registry,
        )

        # ── Gauges ────────────────────────────────────────────────────
        self.active_runs_gauge = Gauge(
            "bench_active_runs",
            "Currently running benchmarks",
            registry=self._registry,
        )
        self.best_ttft_p50 = Gauge(
            "bench_best_ttft_p50_ms",
            "Best (lowest) TTFT p50 observed across all runs",
            ["model"],
            registry=self._registry,
        )
        self.best_throughput_tok_s = Gauge(
            "bench_best_throughput_tok_s",
            "Best token throughput observed across all runs",
            ["model"],
            registry=self._registry,
        )

        # ── Histograms ────────────────────────────────────────────────
        self.ttft_p50_hist = Histogram(
            "bench_ttft_p50_ms",
            "Distribution of TTFT p50 values across concurrency levels",
            ["model"],
            buckets=[50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000],
            registry=self._registry,
        )
        self.tpot_p50_hist = Histogram(
            "bench_tpot_p50_ms",
            "Distribution of TPOT p50 values",
            ["model"],
            buckets=[5, 8, 10, 15, 20, 30, 50, 75, 100],
            registry=self._registry,
        )
        self.throughput_req_hist = Histogram(
            "bench_throughput_req_s",
            "Distribution of request throughput values",
            ["model"],
            buckets=[1, 2, 5, 10, 20, 50, 100, 200],
            registry=self._registry,
        )

    # ── Override start to instrument ──────────────────────────────────
    async def start(self, config: BenchmarkConfig) -> str:
        self.runs_total.inc()
        self.active_runs_gauge.set(len(self.active_runs()) + 1)
        return await super().start(config)

    # ── Record level result ───────────────────────────────────────────
    def _record_metrics(self, result: LevelResult):
        m = result.model
        self.levels_total.labels(model=m).inc()
        self.requests_total.labels(model=m, status="ok").inc(result.successful_requests)
        self.requests_total.labels(model=m, status="failed").inc(result.failed_requests)
        self.ttft_p50_hist.labels(model=m).observe(result.ttft_p50)
        self.tpot_p50_hist.labels(model=m).observe(result.tpot_p50)
        self.throughput_req_hist.labels(model=m).observe(result.throughput_req_s)

        # Update best gauges
        current_best_ttft = self.best_ttft_p50.labels(model=m)._value.get()
        if current_best_ttft == 0 or result.ttft_p50 < current_best_ttft:
            self.best_ttft_p50.labels(model=m).set(result.ttft_p50)

        current_best_tput = self.best_throughput_tok_s.labels(model=m)._value.get()
        if result.throughput_tok_s > current_best_tput:
            self.best_throughput_tok_s.labels(model=m).set(result.throughput_tok_s)

        self.active_runs_gauge.set(len(self.active_runs()))

    def generate_prometheus(self) -> str:
        # Sync metrics from completed results before generating
        for run_id, results in self._results.items():
            for r in results:
                pass  # metrics recorded at emit time in _orchestrate override
        return generate_latest(self._registry).decode("utf-8")
