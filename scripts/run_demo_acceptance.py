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
    write_demo_input_wav,
    write_json,
)
from bionic_head.client.demo_artifacts import collect_existing_artifacts, collect_latest_artifacts, http_get_json
import scripts.history_smoke as history_smoke
import scripts.interactive_demo_client as interactive_demo_client
import scripts.local_demo_client as local_demo_client


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
    checks["scripted_interactive_smoke"] = await _run_scripted_interactive_check(args)
    checks["history_smoke"] = await _run_history_check(args)
    checks["playback_interrupt_smoke"] = await _run_playback_interrupt_check(args)
    for strategy in args.playback_sync:
        checks[f"av_sync_{strategy}"] = await _run_av_sync_check(args, playback_sync=strategy)
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
    ).to_dict()
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


async def _run_scripted_interactive_check(args: argparse.Namespace) -> AcceptanceCheckResult:
    output_dir = args.output_dir / "scripted_interactive_smoke"
    try:
        await interactive_demo_client.run_scripted_demo(
            url=args.url,
            output_dir=output_dir,
            scripted_turns=2,
            scripted_cancel_after_ms=300,
            chunk_ms=args.chunk_ms,
            sample_rate=16000,
            audio_backend=args.audio_backend,
            playback_sync="immediate_audio",
            wait_for_face_timeout_ms=args.wait_for_face_timeout_ms,
            wait_timeout_sec=args.timeout_sec,
        )
        report = _read_json(output_dir / "interaction_report.json")
        reasons: list[str] = []
        if not bool(report.get("success")):
            reasons.append("scripted_interactive_failed")
        if int(report.get("old_generation_audio_play_count", 0) or 0) != 0:
            reasons.append("old_generation_audio_played")
        if int(report.get("old_generation_face_display_count", 0) or 0) != 0:
            reasons.append("old_generation_face_displayed")
        return AcceptanceCheckResult(
            success=not reasons,
            failure_code=reasons[0] if reasons else None,
            failure_message="; ".join(reasons) if reasons else None,
            artifacts=collect_existing_artifacts(
                args.output_dir,
                {
                    "interaction_report": output_dir / "interaction_report.json",
                    "summary": output_dir / "summary.json",
                },
            ),
            metrics={key: report.get(key) for key in ("turn_count", "completed_turn_count", "cancelled_turn_count")},
        )
    except Exception as exc:  # noqa: BLE001
        return AcceptanceCheckResult(False, failure_code="scripted_interactive_exception", error_message=str(exc))


async def _run_history_check(args: argparse.Namespace) -> AcceptanceCheckResult:
    output_dir = args.output_dir / "history_smoke"
    try:
        report = await history_smoke.run_history_smoke(
            url=args.url,
            output_dir=output_dir,
            mode="mock" if args.mode == "fake" else "real",
            turn1_wav=args.history_turn1_wav,
            turn2_wav=args.history_turn2_wav,
            expected_text=args.expect,
            chunk_ms=args.chunk_ms,
            timeout_sec=args.timeout_sec,
        )
        return AcceptanceCheckResult(
            success=report.success,
            failure_code=report.failure_reasons[0] if report.failure_reasons else None,
            failure_message="; ".join(report.failure_reasons) if report.failure_reasons else None,
            artifacts=collect_existing_artifacts(
                args.output_dir,
                {
                    "history_smoke_report": output_dir / "history_smoke_report.json",
                    "summary": output_dir / "summary.json",
                    "events": output_dir / "events.jsonl",
                },
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return AcceptanceCheckResult(False, failure_code="history_smoke_exception", error_message=str(exc))


async def _run_playback_interrupt_check(args: argparse.Namespace) -> AcceptanceCheckResult:
    output_dir = args.output_dir / "playback_interrupt_smoke"
    wav_path = write_demo_input_wav(args.output_dir / "generated-input.wav")
    try:
        terminal = await local_demo_client.run_local_demo(
            url=args.url,
            wav_path=wav_path,
            output_dir=output_dir,
            chunk_ms=args.chunk_ms,
            play_audio=False,
            cancel_after_ms=300,
            playback_sync="immediate_audio",
            wait_for_face_timeout_ms=args.wait_for_face_timeout_ms,
        )
        summary = _read_json(output_dir / "summary.json")
        evidence = terminal == "server.turn.cancelled" or (
            int(summary.get("playback_stop_count", 0) or 0) >= 1
            and summary.get("client_interrupt_sent_ms") is not None
        )
        return AcceptanceCheckResult(
            success=evidence,
            failure_code=None if evidence else "playback_interrupt_failed",
            failure_message=None if evidence else "Playback interrupt did not produce cancel or playback-stop evidence.",
            artifacts=collect_existing_artifacts(
                args.output_dir,
                {
                    "summary": output_dir / "summary.json",
                    "client_playback_metrics": output_dir / "client_playback_metrics.json",
                },
            ),
            metrics={
                "terminal_event": terminal,
                "playback_stop_count": summary.get("playback_stop_count"),
                "client_interrupt_sent_ms": summary.get("client_interrupt_sent_ms"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return AcceptanceCheckResult(False, failure_code="playback_interrupt_exception", error_message=str(exc))


async def _run_av_sync_check(args: argparse.Namespace, *, playback_sync: str) -> AcceptanceCheckResult:
    output_dir = args.output_dir / f"av_sync_{playback_sync}"
    try:
        await interactive_demo_client.run_scripted_demo(
            url=args.url,
            output_dir=output_dir,
            scripted_turns=1,
            scripted_cancel_after_ms=0,
            chunk_ms=args.chunk_ms,
            sample_rate=16000,
            audio_backend=args.audio_backend,
            playback_sync=playback_sync,
            wait_for_face_timeout_ms=args.wait_for_face_timeout_ms,
            wait_timeout_sec=args.timeout_sec,
        )
        report = _read_json(output_dir / "interaction_report.json")
        reasons: list[str] = []
        if not bool(report.get("success")):
            reasons.append("av_sync_failed")
        if report.get("playback_sync_strategy") != playback_sync:
            reasons.append("av_sync_strategy_mismatch")
        if report.get("client_audio_face_offset_ms") is None:
            reasons.append("av_sync_offset_missing")
        if playback_sync == "wait_for_face":
            if report.get("client_audio_wait_for_face_ms") is None:
                reasons.append("av_sync_wait_missing")
            if report.get("client_audio_wait_for_face_timeout") is True:
                reasons.append("av_sync_wait_for_face_timeout")
        return AcceptanceCheckResult(
            success=not reasons,
            failure_code=reasons[0] if reasons else None,
            failure_message="; ".join(reasons) if reasons else None,
            artifacts=collect_existing_artifacts(
                args.output_dir,
                {
                    "interaction_report": output_dir / "interaction_report.json",
                    "summary": output_dir / "summary.json",
                },
            ),
            metrics={
                "playback_sync_strategy": report.get("playback_sync_strategy"),
                "client_audio_face_offset_ms": report.get("client_audio_face_offset_ms"),
                "client_audio_wait_for_face_ms": report.get("client_audio_wait_for_face_ms"),
                "client_audio_wait_for_face_timeout": report.get("client_audio_wait_for_face_timeout"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return AcceptanceCheckResult(False, failure_code=f"av_sync_{playback_sync}_exception", error_message=str(exc))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(run_demo_acceptance(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
