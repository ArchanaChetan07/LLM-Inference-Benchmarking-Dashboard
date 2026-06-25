"""
backend/engine.py

Benchmark orchestration engine.
Drives vllm bench serve across concurrency levels, parses output,
streams results via asyncio.Queue to connected WebSocket clients.

Production usage
----------------
    engine = BenchmarkEngine()
    run_id = await engine.start(config)   # returns immediately
    # Results stream via engine.subscribe(run_id)

    # To swap simulation for real vLLM:
    # Set VLLM_REAL=1 in environment — engine._run_level() calls
    # asyncio.create_subprocess_exec("vllm", "bench", "serve", ...)
    # and parses stdout via _parse_vllm_output()
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

REAL_VLLM = os.getenv("VLLM_REAL", "0") == "1"


# ── Data models ───────────────────────────────────────────────────────
@dataclass
class BenchmarkConfig:
    model: str = "openai/gpt-oss-120b"
    concurrency_levels: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    input_len: int = 4096
    output_len: int = 1024
    backend_url: str = "http://localhost:8000"
    gpu_type: str = "MI300X"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class LevelResult:
    run_id: str
    model: str
    concurrency: int
    timestamp: str
    # TTFT
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    # TPOT
    tpot_p50: float
    tpot_p90: float
    tpot_p99: float
    # ITL
    itl_p50: float
    itl_p99: float
    # E2EL
    e2el_p50: float
    e2el_p99: float
    # Throughput
    throughput_req_s: float
    throughput_tok_s: float
    # Success
    total_requests: int
    successful_requests: int
    failed_requests: int
    # Raw
    duration_s: float


@dataclass
class RunEvent:
    """Single event emitted on the queue during a benchmark run."""
    kind: str   # run_started | level_started | level_result | run_complete | run_error | run_cancelled
    run_id: str
    payload: dict = field(default_factory=dict)


# ── Engine ────────────────────────────────────────────────────────────
class BenchmarkEngine:
    """
    Manages concurrent benchmark runs.
    Each run_id maps to an asyncio.Queue that receives RunEvents.
    """

    def __init__(self):
        self._runs: dict[str, asyncio.Queue] = {}
        self._active: dict[str, bool] = {}
        self._results: dict[str, list[LevelResult]] = {}

    async def start(self, config: BenchmarkConfig) -> str:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._runs[config.run_id] = q
        self._active[config.run_id] = True
        self._results[config.run_id] = []
        asyncio.create_task(self._orchestrate(config, q))
        return config.run_id

    def cancel(self, run_id: str):
        self._active[run_id] = False

    def results(self, run_id: str) -> list[dict]:
        return [asdict(r) for r in self._results.get(run_id, [])]

    def active_runs(self) -> list[str]:
        return [rid for rid, active in self._active.items() if active]

    async def subscribe(self, run_id: str) -> AsyncIterator[RunEvent]:
        q = self._runs.get(run_id)
        if not q:
            return
        while True:
            event: RunEvent = await q.get()
            yield event
            if event.kind in ("run_complete", "run_error", "run_cancelled"):
                break

    # ── Orchestrator ──────────────────────────────────────────────────
    async def _orchestrate(self, cfg: BenchmarkConfig, q: asyncio.Queue):
        run_id = cfg.run_id
        await q.put(RunEvent(
            kind="run_started", run_id=run_id,
            payload={
                "model": cfg.model,
                "concurrency_levels": cfg.concurrency_levels,
                "input_len": cfg.input_len,
                "output_len": cfg.output_len,
                "gpu_type": cfg.gpu_type,
                "started_at": _now(),
            }
        ))

        total = len(cfg.concurrency_levels)
        for i, conc in enumerate(cfg.concurrency_levels):
            if not self._active.get(run_id):
                await q.put(RunEvent(kind="run_cancelled", run_id=run_id, payload={"at_level": conc}))
                return

            await q.put(RunEvent(
                kind="level_started", run_id=run_id,
                payload={
                    "concurrency": conc,
                    "num_prompts": conc * 10,
                    "progress": i / total,
                    "index": i,
                    "total": total,
                }
            ))

            try:
                result = await self._run_level(cfg, conc)
            except Exception as exc:
                await q.put(RunEvent(kind="run_error", run_id=run_id, payload={"error": str(exc), "concurrency": conc}))
                self._active[run_id] = False
                return

            self._results[run_id].append(result)
            await q.put(RunEvent(
                kind="level_result", run_id=run_id,
                payload={**asdict(result), "progress": (i + 1) / total, "index": i, "total": total},
            ))

        self._active[run_id] = False
        await q.put(RunEvent(
            kind="run_complete", run_id=run_id,
            payload={
                "completed_at": _now(),
                "total_levels": total,
                "results": [asdict(r) for r in self._results[run_id]],
            }
        ))

    # ── Level runner ──────────────────────────────────────────────────
    async def _run_level(self, cfg: BenchmarkConfig, concurrency: int) -> LevelResult:
        if REAL_VLLM:
            return await self._run_real(cfg, concurrency)
        return await self._run_simulated(cfg, concurrency)

    async def _run_real(self, cfg: BenchmarkConfig, concurrency: int) -> LevelResult:
        """
        Invoke vllm bench serve as a subprocess and parse its stdout.
        Called when VLLM_REAL=1.
        """
        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            "vllm", "bench", "serve",
            "--model", cfg.model,
            "--dataset-name", "random",
            "--random-input-len", str(cfg.input_len),
            "--random-output-len", str(cfg.output_len),
            "--max-concurrency", str(concurrency),
            "--num-prompts", str(concurrency * 10),
            "--ignore-eos",
            "--percentile_metrics", "ttft,tpot,itl,e2el",
            "--backend", cfg.backend_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        duration = time.monotonic() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"vllm bench serve failed:\n{stderr.decode()}")
        return _parse_vllm_output(stdout.decode(), cfg, concurrency, duration)

    async def _run_simulated(self, cfg: BenchmarkConfig, concurrency: int) -> LevelResult:
        """
        Realistic simulation of MI300X latency model.
        Replace with _run_real() by setting VLLM_REAL=1.
        """
        # Simulate benchmark duration (shorter at low concurrency)
        sim_s = 1.5 + (concurrency / 128) * 2.5 + random.uniform(-0.3, 0.3)
        await asyncio.sleep(sim_s)

        # Latency model tuned to MI300X + GPT-OSS-120B characteristics
        base_ttft   = 55 + concurrency * 2.6  + _gauss(0, 7)
        base_tpot   = 9  + concurrency * 0.14 + _gauss(0, 1.2)
        base_itl    = 11 + concurrency * 0.17 + _gauss(0, 1.8)
        base_e2el_s = (base_ttft + base_tpot * cfg.output_len) / 1000

        tput_req = max(0.4, (1.0 / base_e2el_s) * concurrency * 0.83)
        tput_tok = tput_req * cfg.output_len
        n_total  = concurrency * 10
        n_failed = random.randint(0, max(0, concurrency // 64))

        return LevelResult(
            run_id=cfg.run_id,
            model=cfg.model,
            concurrency=concurrency,
            timestamp=_now(),
            ttft_p50=_r(base_ttft),
            ttft_p90=_r(base_ttft * 1.55 + _gauss(0, 4)),
            ttft_p99=_r(base_ttft * 2.20 + _gauss(0, 6)),
            tpot_p50=_r(base_tpot),
            tpot_p90=_r(base_tpot * 1.45),
            tpot_p99=_r(base_tpot * 1.85),
            itl_p50=_r(base_itl),
            itl_p99=_r(base_itl * 2.40),
            e2el_p50=_r(base_e2el_s, 3),
            e2el_p99=_r(base_e2el_s * 1.65, 3),
            throughput_req_s=_r(tput_req),
            throughput_tok_s=_r(tput_tok, 0),
            total_requests=n_total,
            successful_requests=n_total - n_failed,
            failed_requests=n_failed,
            duration_s=_r(sim_s, 2),
        )


# ── vLLM output parser ────────────────────────────────────────────────
def _parse_vllm_output(raw: str, cfg: BenchmarkConfig, concurrency: int, duration: float) -> LevelResult:
    """
    Parse the text output of `vllm bench serve`.

    Expected lines (example):
        Successful requests: 80
        Benchmark duration (s): 12.34
        Total input tokens: 327680
        Total output tokens: 81920
        Request throughput (req/s): 6.48
        Output token throughput (tok/s): 6644.32
        ...
        TTFT (ms) P50: 84.21
        TTFT (ms) P90: 142.00
        TTFT (ms) P99: 201.55
        ...
    """
    def extract(label: str, default: float = 0.0) -> float:
        for line in raw.splitlines():
            if label.lower() in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        return float(parts[-1].strip())
                    except ValueError:
                        pass
        return default

    n_total = concurrency * 10
    n_success = int(extract("Successful requests", n_total))
    return LevelResult(
        run_id=cfg.run_id,
        model=cfg.model,
        concurrency=concurrency,
        timestamp=_now(),
        ttft_p50=extract("TTFT (ms) P50"),
        ttft_p90=extract("TTFT (ms) P90"),
        ttft_p99=extract("TTFT (ms) P99"),
        tpot_p50=extract("TPOT (ms) P50"),
        tpot_p90=extract("TPOT (ms) P90"),
        tpot_p99=extract("TPOT (ms) P99"),
        itl_p50=extract("ITL (ms) P50"),
        itl_p99=extract("ITL (ms) P99"),
        e2el_p50=extract("E2EL (s) P50"),
        e2el_p99=extract("E2EL (s) P99"),
        throughput_req_s=extract("Request throughput"),
        throughput_tok_s=extract("Output token throughput"),
        total_requests=n_total,
        successful_requests=n_success,
        failed_requests=n_total - n_success,
        duration_s=_r(duration, 2),
    )


# ── Utilities ─────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _r(v: float, d: int = 2) -> float:
    return round(v, d)

def _gauss(mean: float, std: float) -> float:
    return random.gauss(mean, std)
