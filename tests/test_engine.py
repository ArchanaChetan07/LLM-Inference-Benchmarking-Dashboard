"""
tests/test_engine.py

Unit + integration tests for the benchmark engine.

Run:  pytest tests/ -v --asyncio-mode=auto
"""

from __future__ import annotations

import asyncio
import pytest

from backend.engine import BenchmarkEngine, BenchmarkConfig, LevelResult, _parse_vllm_output


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def engine():
    return BenchmarkEngine()


@pytest.fixture
def config():
    return BenchmarkConfig(
        model="test/model-7b",
        concurrency_levels=[4, 8],
        input_len=512,
        output_len=128,
        run_id="test-run-001",
    )


# ── BenchmarkConfig ────────────────────────────────────────────────────
class TestBenchmarkConfig:
    def test_default_concurrency(self):
        cfg = BenchmarkConfig()
        assert cfg.concurrency_levels == [8, 16, 32, 64, 128]

    def test_run_id_auto_generated(self):
        cfg1 = BenchmarkConfig()
        cfg2 = BenchmarkConfig()
        assert cfg1.run_id != cfg2.run_id
        assert len(cfg1.run_id) == 8

    def test_custom_config(self, config):
        assert config.model == "test/model-7b"
        assert config.concurrency_levels == [4, 8]
        assert config.input_len == 512
        assert config.output_len == 128


# ── Engine state ────────────────────────────────────────────────────────
class TestEngineState:
    def test_starts_empty(self, engine):
        assert engine.results("nonexistent") == []
        assert engine.active_runs() == []

    @pytest.mark.asyncio
    async def test_run_registered_after_start(self, engine, config):
        await engine.start(config)
        assert config.run_id in engine._results
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_cancel_stops_run(self, engine):
        cfg = BenchmarkConfig(concurrency_levels=[8, 16, 32, 64, 128], run_id="cancel-test")
        await engine.start(cfg)
        engine.cancel(cfg.run_id)
        assert not engine._active.get(cfg.run_id)


# ── Full run (simulated) ────────────────────────────────────────────────
class TestSimulatedRun:
    @pytest.mark.asyncio
    async def test_run_emits_all_events(self, engine, config):
        await engine.start(config)
        events = []
        async for event in engine.subscribe(config.run_id):
            events.append(event.kind)
        assert events[0] == "run_started"
        assert events.count("level_started") == 2
        assert events.count("level_result") == 2
        assert events[-1] == "run_complete"

    @pytest.mark.asyncio
    async def test_results_stored_after_run(self, engine, config):
        await engine.start(config)
        async for _ in engine.subscribe(config.run_id):
            pass
        results = engine.results(config.run_id)
        assert len(results) == 2
        assert results[0]["concurrency"] == 4
        assert results[1]["concurrency"] == 8

    @pytest.mark.asyncio
    async def test_level_result_fields(self, engine, config):
        await engine.start(config)
        async for event in engine.subscribe(config.run_id):
            if event.kind == "level_result":
                p = event.payload
                assert "ttft_p50" in p
                assert "ttft_p99" in p
                assert "tpot_p50" in p
                assert "itl_p50" in p
                assert "e2el_p50" in p
                assert "throughput_req_s" in p
                assert "throughput_tok_s" in p
                assert p["concurrency"] in [4, 8]
                break

    @pytest.mark.asyncio
    async def test_ttft_increases_with_concurrency(self, engine):
        cfg = BenchmarkConfig(
            concurrency_levels=[8, 128],
            run_id="ttft-scaling-test",
        )
        await engine.start(cfg)
        async for _ in engine.subscribe(cfg.run_id):
            pass
        results = engine.results(cfg.run_id)
        low_conc  = next(r for r in results if r["concurrency"] == 8)
        high_conc = next(r for r in results if r["concurrency"] == 128)
        assert high_conc["ttft_p50"] > low_conc["ttft_p50"]

    @pytest.mark.asyncio
    async def test_throughput_increases_with_concurrency(self, engine):
        cfg = BenchmarkConfig(
            concurrency_levels=[8, 128],
            run_id="tput-scaling-test",
        )
        await engine.start(cfg)
        async for _ in engine.subscribe(cfg.run_id):
            pass
        results = engine.results(cfg.run_id)
        low  = next(r for r in results if r["concurrency"] == 8)
        high = next(r for r in results if r["concurrency"] == 128)
        assert high["throughput_req_s"] > low["throughput_req_s"]

    @pytest.mark.asyncio
    async def test_p99_greater_than_p50(self, engine, config):
        await engine.start(config)
        async for _ in engine.subscribe(config.run_id):
            pass
        for r in engine.results(config.run_id):
            assert r["ttft_p99"] >= r["ttft_p50"]
            assert r["tpot_p99"] >= r["tpot_p50"]

    @pytest.mark.asyncio
    async def test_successful_plus_failed_equals_total(self, engine, config):
        await engine.start(config)
        async for _ in engine.subscribe(config.run_id):
            pass
        for r in engine.results(config.run_id):
            assert r["successful_requests"] + r["failed_requests"] == r["total_requests"]

    @pytest.mark.asyncio
    async def test_num_prompts_ten_times_concurrency(self, engine, config):
        await engine.start(config)
        async for event in engine.subscribe(config.run_id):
            if event.kind == "level_started":
                conc = event.payload["concurrency"]
                assert event.payload["num_prompts"] == conc * 10


# ── vLLM output parser ─────────────────────────────────────────────────
class TestVllmParser:
    SAMPLE_OUTPUT = """
Successful requests:                   80
Benchmark duration (s):                12.34
Total input tokens:                    327680
Total output tokens:                   81920
Request throughput (req/s):            6.48
Output token throughput (tok/s):       6644.32
Mean TTFT (ms):                        84.21
Median TTFT (ms):                      82.10
P90 TTFT (ms):                         141.00
P99 TTFT (ms):                         201.55
Mean TPOT (ms):                        11.22
Median TPOT (ms):                      10.80
P99 TPOT (ms):                         18.50
TTFT (ms) P50:                         82.10
TTFT (ms) P90:                         141.00
TTFT (ms) P99:                         201.55
TPOT (ms) P50:                         10.80
TPOT (ms) P90:                         14.20
TPOT (ms) P99:                         18.50
ITL (ms) P50:                          11.50
ITL (ms) P99:                          27.30
E2EL (s) P50:                          11.18
E2EL (s) P99:                          18.44
"""

    def test_parse_successful_requests(self):
        cfg = BenchmarkConfig(run_id="parse-test")
        result = _parse_vllm_output(self.SAMPLE_OUTPUT, cfg, 8, 12.34)
        assert result.successful_requests == 80

    def test_parse_throughput(self):
        cfg = BenchmarkConfig(run_id="parse-test-2")
        result = _parse_vllm_output(self.SAMPLE_OUTPUT, cfg, 8, 12.34)
        assert result.throughput_req_s == 6.48
        assert result.throughput_tok_s == 6644.32

    def test_parse_ttft(self):
        cfg = BenchmarkConfig(run_id="parse-test-3")
        result = _parse_vllm_output(self.SAMPLE_OUTPUT, cfg, 8, 12.34)
        assert result.ttft_p50 == 82.10
        assert result.ttft_p99 == 201.55

    def test_parse_e2el(self):
        cfg = BenchmarkConfig(run_id="parse-test-4")
        result = _parse_vllm_output(self.SAMPLE_OUTPUT, cfg, 8, 12.34)
        assert result.e2el_p50 == 11.18
        assert result.e2el_p99 == 18.44

    def test_parse_missing_field_returns_zero(self):
        cfg = BenchmarkConfig(run_id="parse-test-5")
        result = _parse_vllm_output("Successful requests: 10", cfg, 8, 1.0)
        assert result.ttft_p50 == 0.0


# ── Concurrent runs ────────────────────────────────────────────────────
class TestConcurrentRuns:
    @pytest.mark.asyncio
    async def test_multiple_runs_isolated(self, engine):
        cfg1 = BenchmarkConfig(concurrency_levels=[4], run_id="run-A", model="model-A")
        cfg2 = BenchmarkConfig(concurrency_levels=[8], run_id="run-B", model="model-B")
        await engine.start(cfg1)
        await engine.start(cfg2)
        async for _ in engine.subscribe("run-A"):
            pass
        async for _ in engine.subscribe("run-B"):
            pass
        assert all(r["model"] == "model-A" for r in engine.results("run-A"))
        assert all(r["model"] == "model-B" for r in engine.results("run-B"))

    @pytest.mark.asyncio
    async def test_results_not_cross_contaminated(self, engine):
        cfg1 = BenchmarkConfig(concurrency_levels=[4], run_id="iso-1")
        cfg2 = BenchmarkConfig(concurrency_levels=[128], run_id="iso-2")
        await engine.start(cfg1)
        await engine.start(cfg2)
        async for _ in engine.subscribe("iso-1"):
            pass
        async for _ in engine.subscribe("iso-2"):
            pass
        assert engine.results("iso-1")[0]["concurrency"] == 4
        assert engine.results("iso-2")[0]["concurrency"] == 128
