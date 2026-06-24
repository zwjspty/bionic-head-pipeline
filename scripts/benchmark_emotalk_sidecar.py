from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bionic_head.adapters.emotalk import EmoTalkAudio2FaceAdapter
from bionic_head.adapters.emotalk_sidecar import EmoTalkSidecarAudio2FaceAdapter
from bionic_head.config import load_settings
from bionic_head.core.cancellation import CancellationToken
from bionic_head.domain.models import AudioArtifact, Emotion, TurnContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark old EmoTalk provider against persistent sidecar",
    )
    parser.add_argument("--config", type=Path, default=Path("config/emotalk.example.json"))
    parser.add_argument("--wav", type=Path, required=True, help="Input WAV used for Audio2Face.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to data/benchmarks/emotalk_sidecar_<timestamp>.json",
    )
    parser.add_argument("--old-runs", type=int, default=1)
    parser.add_argument("--sidecar-runs", type=int, default=3)
    parser.add_argument("--prewarm-runs", type=int, default=2)
    parser.add_argument("--skip-old", action="store_true")
    parser.add_argument("--skip-sidecar", action="store_true")
    return parser


def build_summary(
    *,
    old_emotalk_ms: list[float],
    sidecar_cold_ms: float | None,
    sidecar_warm_ms: list[float],
    old_shapes: list[list[int]],
    sidecar_shapes: list[list[int]],
    sidecar_breakdown: list[dict[str, float]] | None = None,
    prewarm_ms: float | None = None,
    first_real_after_prewarm_ms: float | None = None,
    second_real_after_prewarm_ms: float | None = None,
    prewarmed_ms: list[float] | None = None,
    prewarm_breakdown: dict[str, float] | None = None,
    prewarmed_breakdown: list[dict[str, float]] | None = None,
) -> dict[str, object]:
    speedup = None
    if old_emotalk_ms and sidecar_warm_ms:
        warm_mean = statistics.fmean(sidecar_warm_ms)
        if warm_mean > 0:
            speedup = statistics.fmean(old_emotalk_ms) / warm_mean
    rounded_breakdown = [_round_metrics(metrics) for metrics in (sidecar_breakdown or [])]
    rounded_prewarmed_breakdown = [
        _round_metrics(metrics) for metrics in (prewarmed_breakdown or [])
    ]
    rounded_prewarmed_ms = [round(value, 3) for value in (prewarmed_ms or [])]
    first_after_prewarm = (
        first_real_after_prewarm_ms
        if first_real_after_prewarm_ms is not None
        else (prewarmed_ms or [None])[0]
    )
    second_after_prewarm = (
        second_real_after_prewarm_ms
        if second_real_after_prewarm_ms is not None
        else ((prewarmed_ms or [None, None])[1] if len(prewarmed_ms or []) > 1 else None)
    )
    return {
        "old_emotalk_ms": [round(value, 3) for value in old_emotalk_ms],
        "sidecar_cold_ms": None if sidecar_cold_ms is None else round(sidecar_cold_ms, 3),
        "cold_without_prewarm_ms": None if sidecar_cold_ms is None else round(sidecar_cold_ms, 3),
        "sidecar_warm_ms": [round(value, 3) for value in sidecar_warm_ms],
        "speedup_warm_vs_old": None if speedup is None else round(speedup, 3),
        "old_shapes": old_shapes,
        "sidecar_shapes": sidecar_shapes,
        "sidecar_breakdown": rounded_breakdown,
        "warm_breakdown": rounded_breakdown[1:],
        "prewarm_ms": None if prewarm_ms is None else round(prewarm_ms, 3),
        "first_real_after_prewarm_ms": None
        if first_after_prewarm is None
        else round(first_after_prewarm, 3),
        "second_real_after_prewarm_ms": None
        if second_after_prewarm is None
        else round(second_after_prewarm, 3),
        "prewarmed_ms": rounded_prewarmed_ms,
        "prewarm_effective": None if first_after_prewarm is None else first_after_prewarm < 1000.0,
        "prewarm_breakdown": _round_metrics(prewarm_breakdown or {}),
        "prewarmed_breakdown": rounded_prewarmed_breakdown,
    }


def write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.old_runs < 0 or args.sidecar_runs < 0 or args.prewarm_runs < 0:
        raise SystemExit("--old-runs, --sidecar-runs, and --prewarm-runs must be non-negative")
    if args.skip_old and args.skip_sidecar:
        raise SystemExit("nothing to benchmark: both --skip-old and --skip-sidecar were set")

    settings = load_settings(args.config)
    audio = _audio_artifact_from_wav(args.wav)
    run_root = Path("data/benchmarks") / f"emotalk_sidecar_{_timestamp()}"
    run_root.mkdir(parents=True, exist_ok=True)

    old_ms: list[float] = []
    old_shapes: list[list[int]] = []
    sidecar_ms: list[float] = []
    sidecar_shapes: list[list[int]] = []
    sidecar_breakdown: list[dict[str, float]] = []
    prewarm_ms: float | None = None
    prewarm_breakdown: dict[str, float] = {}
    prewarmed_ms: list[float] = []
    prewarmed_shapes: list[list[int]] = []
    prewarmed_breakdown: list[dict[str, float]] = []

    if not args.skip_old and args.old_runs > 0:
        old_adapter = EmoTalkAudio2FaceAdapter.from_settings(
            settings.providers.emotalk,
            grace_seconds=settings.limits.subprocess_terminate_grace_seconds,
        )
        old_ms, old_shapes, _old_breakdown = await _time_adapter(
            old_adapter,
            audio,
            run_root=run_root,
            label="old_emotalk",
            runs=args.old_runs,
        )

    sidecar_adapter: EmoTalkSidecarAudio2FaceAdapter | None = None
    try:
        if not args.skip_sidecar and args.sidecar_runs > 0:
            sidecar_adapter = EmoTalkSidecarAudio2FaceAdapter.from_settings(
                settings.providers.emotalk_sidecar,
            )
            sidecar_ms, sidecar_shapes, sidecar_breakdown = await _time_adapter(
                sidecar_adapter,
                audio,
                run_root=run_root,
                label="emotalk_sidecar",
                runs=args.sidecar_runs,
            )
    finally:
        if sidecar_adapter is not None:
            await sidecar_adapter.close()

    prewarmed_adapter: EmoTalkSidecarAudio2FaceAdapter | None = None
    try:
        if not args.skip_sidecar and args.prewarm_runs > 0:
            prewarmed_adapter = EmoTalkSidecarAudio2FaceAdapter.from_settings(
                settings.providers.emotalk_sidecar,
            )
            prewarm_started = perf_counter()
            await prewarmed_adapter.prewarm()
            prewarm_ms = (perf_counter() - prewarm_started) * 1000.0
            prewarm_breakdown = dict(prewarmed_adapter.last_prewarm_metrics)
            prewarmed_ms, prewarmed_shapes, prewarmed_breakdown = await _time_adapter(
                prewarmed_adapter,
                audio,
                run_root=run_root,
                label="emotalk_sidecar_after_prewarm",
                runs=args.prewarm_runs,
            )
    finally:
        if prewarmed_adapter is not None:
            await prewarmed_adapter.close()

    summary = build_summary(
        old_emotalk_ms=old_ms,
        sidecar_cold_ms=sidecar_ms[0] if sidecar_ms else None,
        sidecar_warm_ms=sidecar_ms[1:],
        old_shapes=old_shapes,
        sidecar_shapes=sidecar_shapes,
        sidecar_breakdown=sidecar_breakdown,
        prewarm_ms=prewarm_ms,
        prewarmed_ms=prewarmed_ms,
        prewarm_breakdown=prewarm_breakdown,
        prewarmed_breakdown=prewarmed_breakdown,
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "source_wav": str(args.wav),
        "audio": {
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
            "duration_seconds": audio.duration_seconds,
            "byte_length": audio.byte_length,
        },
        "sidecar_command": settings.providers.emotalk_sidecar.sidecar_command,
        "sidecar_cwd": None
        if settings.providers.emotalk_sidecar.sidecar_cwd is None
        else str(settings.providers.emotalk_sidecar.sidecar_cwd),
        "sidecar_env": settings.providers.emotalk_sidecar.sidecar_env,
        "prewarmed_shapes": prewarmed_shapes,
        **summary,
    }


async def _time_adapter(
    adapter,
    audio: AudioArtifact,
    *,
    run_root: Path,
    label: str,
    runs: int,
) -> tuple[list[float], list[list[int]], list[dict[str, float]]]:
    durations: list[float] = []
    shapes: list[list[int]] = []
    breakdowns: list[dict[str, float]] = []
    for index in range(runs):
        context = TurnContext(
            session_id=uuid4(),
            turn_id=uuid4(),
            artifact_dir=run_root / label / f"run-{index + 1:04d}",
            cancellation=CancellationToken(),
            generation_epoch=0,
        )
        started = perf_counter()
        face = await adapter.drive(audio, Emotion.NEUTRAL, 0.5, context)
        durations.append((perf_counter() - started) * 1000.0)
        shapes.append([face.frame_count, face.channel_count])
        breakdowns.append(_load_face_metrics(face.auxiliary_paths))
    return durations, shapes, breakdowns


def _load_face_metrics(paths: list[Path]) -> dict[str, float]:
    for path in paths:
        if path.name != "meta.json" or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            return {}
        numeric_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(key, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_metrics[key] = float(value)
        return numeric_metrics
    return {}


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 3) for key, value in metrics.items()}


def _audio_artifact_from_wav(path: Path) -> AudioArtifact:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
    return AudioArtifact(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        duration_seconds=frame_count / float(sample_rate),
        byte_length=path.stat().st_size,
    )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or Path("data/benchmarks") / f"emotalk_sidecar_{_timestamp()}.json"
    report = asyncio.run(run_benchmark(args))
    write_report(output, report)
    print(f"wrote {output}")
    print(json.dumps(build_summary(
        old_emotalk_ms=report["old_emotalk_ms"],  # type: ignore[arg-type]
        sidecar_cold_ms=report["sidecar_cold_ms"],  # type: ignore[arg-type]
        sidecar_warm_ms=report["sidecar_warm_ms"],  # type: ignore[arg-type]
        old_shapes=report["old_shapes"],  # type: ignore[arg-type]
        sidecar_shapes=report["sidecar_shapes"],  # type: ignore[arg-type]
        sidecar_breakdown=report["sidecar_breakdown"],  # type: ignore[arg-type]
        prewarm_ms=report["prewarm_ms"],  # type: ignore[arg-type]
        prewarmed_ms=report["prewarmed_ms"],  # type: ignore[arg-type]
        prewarm_breakdown=report["prewarm_breakdown"],  # type: ignore[arg-type]
        prewarmed_breakdown=report["prewarmed_breakdown"],  # type: ignore[arg-type]
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
