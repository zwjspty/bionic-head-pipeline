from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_benchmark_emotalk_sidecar_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_emotalk_sidecar.py", "--help"],
        check=False,
        env={**os.environ, "PYTHONPATH": "src:."},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Benchmark old EmoTalk provider against persistent sidecar" in result.stdout


def test_build_summary_compares_old_cold_and_warm_sidecar() -> None:
    from scripts.benchmark_emotalk_sidecar import build_summary

    summary = build_summary(
        old_emotalk_ms=[8000.0, 16000.0],
        sidecar_cold_ms=7000.0,
        sidecar_warm_ms=[100.0, 200.0],
        old_shapes=[[111, 52], [99, 52]],
        sidecar_shapes=[[30, 52], [30, 52], [30, 52]],
        sidecar_breakdown=[
            {"provider_total_ms": 7000.0, "worker_total_ms": 6900.0},
            {"provider_total_ms": 100.0, "worker_total_ms": 90.0},
            {"provider_total_ms": 200.0, "worker_total_ms": 180.0},
        ],
        prewarm_ms=7350.120,
        first_real_after_prewarm_ms=440.231,
        second_real_after_prewarm_ms=432.884,
        prewarmed_ms=[440.231, 432.884],
        prewarm_breakdown={"worker_total_ms": 7340.0, "model_predict_ms": 400.0},
        prewarmed_breakdown=[
            {"provider_total_ms": 440.231, "worker_total_ms": 439.0},
            {"provider_total_ms": 432.884, "worker_total_ms": 431.0},
        ],
    )

    assert summary["old_emotalk_ms"] == [8000.0, 16000.0]
    assert summary["sidecar_cold_ms"] == 7000.0
    assert summary["cold_without_prewarm_ms"] == 7000.0
    assert summary["sidecar_warm_ms"] == [100.0, 200.0]
    assert summary["speedup_warm_vs_old"] == 80.0
    assert summary["old_shapes"] == [[111, 52], [99, 52]]
    assert summary["sidecar_shapes"] == [[30, 52], [30, 52], [30, 52]]
    assert summary["sidecar_breakdown"] == [
        {"provider_total_ms": 7000.0, "worker_total_ms": 6900.0},
        {"provider_total_ms": 100.0, "worker_total_ms": 90.0},
        {"provider_total_ms": 200.0, "worker_total_ms": 180.0},
    ]
    assert summary["warm_breakdown"] == [
        {"provider_total_ms": 100.0, "worker_total_ms": 90.0},
        {"provider_total_ms": 200.0, "worker_total_ms": 180.0},
    ]
    assert summary["prewarm_ms"] == 7350.12
    assert summary["first_real_after_prewarm_ms"] == 440.231
    assert summary["second_real_after_prewarm_ms"] == 432.884
    assert summary["prewarm_effective"] is True
    assert summary["prewarm_breakdown"] == {
        "worker_total_ms": 7340.0,
        "model_predict_ms": 400.0,
    }
    assert summary["prewarmed_breakdown"] == [
        {"provider_total_ms": 440.231, "worker_total_ms": 439.0},
        {"provider_total_ms": 432.884, "worker_total_ms": 431.0},
    ]


def test_write_benchmark_report_creates_parent_directory(tmp_path: Path) -> None:
    from scripts.benchmark_emotalk_sidecar import write_report

    output = tmp_path / "benchmarks" / "emotalk_sidecar.json"
    payload = {"old_emotalk_ms": [1.0], "sidecar_warm_ms": [0.1]}

    write_report(output, payload)

    assert json.loads(output.read_text(encoding="utf-8")) == payload
