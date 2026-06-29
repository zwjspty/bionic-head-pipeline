from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bionic_head.client.demo_acceptance import (
    AcceptanceCheckResult,
    build_demo_acceptance_report,
    write_json,
)
from bionic_head.client.demo_artifacts import collect_latest_artifacts, http_get_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local demo acceptance checks.")
    parser.add_argument("--url", required=True, help="WebSocket URL, e.g. ws://127.0.0.1:8005/pipeline/stream")
    parser.add_argument("--http-base-url", required=True, help="HTTP base URL, e.g. http://127.0.0.1:8005")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["fake", "real"], default="fake")
    parser.add_argument("--audio-backend", choices=["null", "sounddevice"], default="null")
    parser.add_argument(
        "--playback-sync",
        nargs="+",
        choices=["immediate_audio", "wait_for_face"],
        default=["immediate_audio", "wait_for_face"],
    )
    parser.add_argument("--wait-for-face-timeout-ms", type=int, default=800)
    parser.add_argument("--history-turn1-wav", type=Path)
    parser.add_argument("--history-turn2-wav", type=Path)
    parser.add_argument("--expect", default="小张")
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--data-latest-dir", type=Path, default=Path("data/latest"))
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "real" and (args.history_turn1_wav is None or args.history_turn2_wav is None):
        raise SystemExit("real mode requires --history-turn1-wav and --history-turn2-wav")


async def run_demo_acceptance(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    server = await _check_server(args.http_base_url, timeout_sec=args.timeout_sec)
    checks: dict[str, AcceptanceCheckResult] = {}
    artifacts = collect_latest_artifacts(
        output_dir=args.output_dir,
        http_base_url=args.http_base_url,
        data_latest_dir=args.data_latest_dir,
        timeout_sec=args.timeout_sec,
    )
    report = build_demo_acceptance_report(
        mode=args.mode,
        server=server,
        checks=checks,
        artifacts=artifacts,
    )
    write_json(args.output_dir / "demo_acceptance_report.json", report)
    return report


async def _check_server(http_base_url: str, *, timeout_sec: float) -> dict[str, object]:
    health_ok, health_payload, health_error = http_get_json(
        http_base_url.rstrip("/") + "/health",
        timeout_sec=timeout_sec,
    )
    diagnostics_ok, diagnostics_payload, diagnostics_error = http_get_json(
        http_base_url.rstrip("/") + "/diagnostics",
        timeout_sec=timeout_sec,
    )
    return {
        "health_ok": bool(health_ok and isinstance(health_payload, dict) and health_payload.get("status") == "ok"),
        "health": health_payload,
        "health_error": health_error,
        "diagnostics_ok": diagnostics_ok,
        "diagnostics": diagnostics_payload,
        "diagnostics_error": diagnostics_error,
    }


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(run_demo_acceptance(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
