from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    summary = _read_stream_summary(output_dir)
    return {
        "success": terminal == "server.pipeline.done",
        "failure_code": None if terminal == "server.pipeline.done" else terminal,
        "metrics": stream_metrics_from_summary(summary, wall_ms=wall_ms),
    }


def stream_metrics_from_summary(summary: dict[str, object], *, wall_ms: float) -> dict[str, float]:
    metrics = {"total_turn_duration_ms": wall_ms}
    event_first_ms = summary.get("event_first_ms")
    if not isinstance(event_first_ms, dict):
        event_first_ms = {}

    first_tts = _float_or_none(summary.get("first_tts_binary_ms"))
    if first_tts is None:
        first_tts = _float_or_none(event_first_ms.get("server.tts.audio"))
    if first_tts is not None:
        metrics["tts_first_audio_ms"] = first_tts
        metrics["e2e_first_audible_ms"] = first_tts

    first_ue5 = _float_or_none(event_first_ms.get("server.ue5.frames"))
    first_face = _float_or_none(event_first_ms.get("server.face.frames"))
    if first_face is None:
        first_face = first_ue5
    if first_face is not None:
        metrics["face_first_chunk_ms"] = first_face

    if first_ue5 is not None:
        metrics["e2e_first_visible_face_ms"] = first_ue5

    playback_stop = _float_or_none(event_first_ms.get("server.playback.stop"))
    if playback_stop is not None:
        metrics["interrupt_to_playback_stop_ms"] = playback_stop

    first_segment = _first_stream_segment(summary)
    if first_segment is not None:
        tts_audio_ready = _float_or_none(first_segment.get("tts_audio_event_ms"))
        if tts_audio_ready is not None:
            metrics["tts_audio_ready_ms"] = tts_audio_ready
        face_total = _float_or_none(first_segment.get("face_total_ms"))
        if face_total is not None:
            metrics["face_total_ms"] = face_total
        first_frame_after_tts = _float_or_none(first_segment.get("ue5_first_frame_after_tts_ms"))
        if first_frame_after_tts is not None:
            metrics["ue5_first_frame_after_tts_ms"] = first_frame_after_tts
        first_visible = _float_or_none(first_segment.get("ue5_first_frame_ms"))
        if first_visible is None:
            first_visible = _float_or_none(first_segment.get("e2e_first_visible_face_ms"))
        if first_visible is not None:
            metrics["e2e_first_visible_face_ms"] = first_visible
        for key in (
            "face_stitch_overlap_frames",
            "face_stitch_actual_overlap_frames",
            "face_boundary_delta_before",
            "face_boundary_delta_after",
        ):
            value = _float_or_none(first_segment.get(key))
            if value is not None:
                metrics[key] = value
        applied = _bool_or_none(first_segment.get("face_stitch_applied"))
        if applied is not None:
            metrics["face_stitch_applied_count"] = 1.0 if applied else 0.0
        reset = _bool_or_none(first_segment.get("face_stitch_reset"))
        if reset is not None:
            metrics["face_stitch_reset_count"] = 1.0 if reset else 0.0

    segments = _stream_segments(summary)
    if segments:
        applied_values = [
            value
            for value in (_bool_or_none(segment.get("face_stitch_applied")) for segment in segments)
            if value is not None
        ]
        if applied_values:
            metrics["face_stitch_applied_count"] = float(sum(1 for value in applied_values if value))
        reset_values = [
            value
            for value in (_bool_or_none(segment.get("face_stitch_reset")) for segment in segments)
            if value is not None
        ]
        if reset_values:
            metrics["face_stitch_reset_count"] = float(sum(1 for value in reset_values if value))

    for key in ("old_turn_face_leak_count", "stale_face_drop_count"):
        value = _float_or_none(summary.get(key))
        if value is None and key == "stale_face_drop_count":
            value = _float_or_none(summary.get("stale_drop_count"))
        if value is not None:
            metrics[key] = value

    return metrics


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


def _read_stream_summary(output_dir: Path) -> dict[str, object]:
    path = output_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _first_stream_segment(summary: dict[str, object]) -> dict[str, object] | None:
    candidates = _stream_segments(summary)
    if not candidates:
        return None

    return min(candidates, key=_stream_segment_sort_key)


def _stream_segments(summary: dict[str, object]) -> list[dict[str, object]]:
    segments = summary.get("segments")
    if not isinstance(segments, dict):
        return []
    return [segment for segment in segments.values() if isinstance(segment, dict)]


def _stream_segment_sort_key(segment: dict[str, object]) -> tuple[float, str]:
    order = _float_or_none(segment.get("tts_audio_event_ms"))
    if order is None:
        order = _float_or_none(segment.get("ue5_first_frame_ms"))
    if order is None:
        order = float("inf")
    segment_id = str(segment.get("segment_id", segment.get("chunk_id", "")))
    return order, segment_id


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
