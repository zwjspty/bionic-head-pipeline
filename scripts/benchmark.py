from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from time import perf_counter

from bionic_head.evaluation.latency import build_latency_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark real bionic-head pipeline latency")
    parser.add_argument("--mode", choices=["offline", "stream"], default="offline")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="HTTP API base URL")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8000/pipeline/stream", help="WebSocket stream URL")
    parser.add_argument("--wav", required=True, type=Path, help="Input Chinese WAV")
    parser.add_argument("--runs", type=int, default=1, help="Number of turns to run; use at least 10 for acceptance evidence")
    parser.add_argument("--output", required=True, type=Path, help="Output latency_report.json path")
    return parser


def run_offline_once(base_url: str, wav_path: Path) -> dict[str, object]:
    import httpx

    started = perf_counter()
    with httpx.Client(timeout=600.0) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/pipeline/audio",
            files={"audio": ("input.wav", wav_path.read_bytes(), "audio/wav")},
        )
    wall_ms = (perf_counter() - started) * 1000.0
    if response.status_code != 200:
        return {
            "success": False,
            "failure_code": _failure_code(response),
            "metrics": {"total_turn_duration_ms": wall_ms},
        }
    body = response.json()
    return {
        "success": True,
        "timeline": body.get("timeline", {}),
        "providers": _providers_from_timeline(body.get("timeline", {})),
        "metrics": {"total_turn_duration_ms": wall_ms},
    }


def run_stream_once(ws_url: str, wav_path: Path, output_dir: Path) -> dict[str, object]:
    from scripts.stream_client import run_client

    started = perf_counter()
    terminal = asyncio.run(run_client(ws_url, wav_path, output_dir, chunk_ms=40))
    wall_ms = (perf_counter() - started) * 1000.0
    return {
        "success": terminal == "server.pipeline.done",
        "failure_code": None if terminal == "server.pipeline.done" else terminal,
        "metrics": {"total_turn_duration_ms": wall_ms},
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")

    runs: list[dict[str, object]] = []
    if args.mode == "offline":
        for _ in range(args.runs):
            runs.append(run_offline_once(args.base_url, args.wav))
    else:
        with tempfile.TemporaryDirectory(prefix="bionic-benchmark-") as tmp:
            root = Path(tmp)
            for index in range(args.runs):
                runs.append(run_stream_once(args.ws_url, args.wav, root / f"run-{index:04d}"))

    report = build_latency_report(runs, source_wav=args.wav, mode=args.mode)
    report["acceptance_evidence"] = args.runs >= 10
    if args.runs < 10:
        report["acceptance_note"] = "Use at least 10 runs for deployment acceptance evidence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


def _failure_code(response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"http_{response.status_code}"
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return error["code"]
    return f"http_{response.status_code}"


def _providers_from_timeline(timeline: object) -> dict[str, str]:
    if not isinstance(timeline, dict):
        return {}
    stages = timeline.get("stages")
    if not isinstance(stages, list):
        return {}
    providers: dict[str, str] = {}
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        name = stage.get("name")
        provider = stage.get("provider")
        if isinstance(name, str) and isinstance(provider, str):
            providers[name] = provider
    return providers


if __name__ == "__main__":
    main()
