from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bionic_head.client.history_smoke import write_history_smoke_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a two-turn stream history smoke.")
    parser.add_argument("--url", required=True, help="WebSocket URL, e.g. ws://127.0.0.1:8005/pipeline/stream")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["mock", "real"], default="real")
    parser.add_argument("--turn1-wav", type=Path)
    parser.add_argument("--turn2-wav", type=Path)
    parser.add_argument("--expect", default="小张")
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    return parser


async def run_history_smoke(
    *,
    url: str,
    output_dir: Path,
    mode: str,
    turn1_wav: Path | None,
    turn2_wav: Path | None,
    expected_text: str,
    chunk_ms: int,
    timeout_sec: float,
):
    raise NotImplementedError("run_history_smoke is implemented in Task 3")


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "real" and (args.turn1_wav is None or args.turn2_wav is None):
        raise SystemExit("real mode requires --turn1-wav and --turn2-wav")

    report = asyncio.run(
        run_history_smoke(
            url=args.url,
            output_dir=args.output_dir,
            mode=args.mode,
            turn1_wav=args.turn1_wav,
            turn2_wav=args.turn2_wav,
            expected_text=args.expect,
            chunk_ms=args.chunk_ms,
            timeout_sec=args.timeout_sec,
        )
    )
    write_history_smoke_report(args.output_dir / "history_smoke_report.json", report)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if not report.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
