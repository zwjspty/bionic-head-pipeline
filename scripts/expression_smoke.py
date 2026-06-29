from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from bionic_head.domain.models import Emotion
from bionic_head.expression import ExpressionPostProcessor, load_expression_channel_mapping
from scripts.render_emotalk_grey_head import DEFAULT_EMOTALK_ROOT, render_grey_head


DEFAULT_EXPRESSION_PROFILES: dict[str, dict[str, float]] = {
    "happy": {"mouth_smile_left": 0.15, "mouth_smile_right": 0.15},
    "friendly": {"mouth_smile_left": 0.08, "mouth_smile_right": 0.08},
    "surprised": {"jaw_open": 0.18},
    "sad": {"mouth_frown_left": 0.1, "mouth_frown_right": 0.1},
    "thinking": {"brow_down_left": 0.08, "brow_down_right": 0.08},
}


def load_base_face(path: Path) -> list[list[float]]:
    try:
        frames = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load base face npy: {path}") from exc
    if frames.ndim != 2 or frames.shape[1] != 52:
        raise ValueError(f"Expected base face shaped [N, 52], got {frames.shape}")
    if frames.shape[0] <= 0:
        raise ValueError("Expected at least one base face frame")
    if not np.isfinite(frames).all():
        raise ValueError("Base face values must be finite")
    return frames.astype(np.float32).tolist()


def write_expression_variants(
    *,
    base_face: Path,
    output_dir: Path,
    emotions: Iterable[str],
    mapping_path: Path,
    profiles: dict[str, dict[str, float]],
    max_delta: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_base_face(base_face)
    mapping = load_expression_channel_mapping(mapping_path)
    processor = ExpressionPostProcessor(
        enabled=True,
        mapping=mapping,
        profiles=profiles,
        max_delta=max_delta,
    )

    variants: dict[str, dict[str, object]] = {}
    for emotion in emotions:
        if emotion == Emotion.NEUTRAL.value:
            variant_frames = [frame[:] for frame in frames]
            metrics = ExpressionPostProcessor(enabled=False).process(
                frames,
                emotion=emotion,
                intensity=0.0,
            )[1]
        else:
            variant_frames, metrics = processor.process(frames, emotion=emotion, intensity=1.0)
        face_npy = output_dir / f"{emotion}.npy"
        np.save(face_npy, np.asarray(variant_frames, dtype=np.float32))
        variants[emotion] = {
            "face_npy": str(face_npy),
            "metrics": metrics.to_timing_payload(),
        }

    report: dict[str, object] = {
        "base_face": str(base_face),
        "mapping_path": str(mapping_path),
        "max_delta": max_delta,
        "variants": variants,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run expression visual smoke for grey-head rendering.")
    parser.add_argument("--base-face", type=Path, required=True, help="Input [N,52] base face .npy.")
    parser.add_argument("--audio", type=Path, required=True, help="Input WAV to mux into rendered videos.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=["neutral", "happy", "surprised", "sad", "thinking"],
    )
    parser.add_argument("--mapping", type=Path, default=Path("config/expression_channels.example.json"))
    parser.add_argument("--max-delta", type=float, default=0.3)
    parser.add_argument("--resolution", default="720")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--emotalk-root", type=Path, default=DEFAULT_EMOTALK_ROOT)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/bionic-expression-smoke-render"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-render", action="store_true", help="Only write variant npy files and report.json.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = write_expression_variants(
        base_face=args.base_face,
        output_dir=args.output_dir,
        emotions=args.emotions,
        mapping_path=args.mapping,
        profiles=DEFAULT_EXPRESSION_PROFILES,
        max_delta=args.max_delta,
    )
    if not args.skip_render:
        variants = report["variants"]
        assert isinstance(variants, dict)
        for emotion, item in variants.items():
            assert isinstance(item, dict)
            render_grey_head(
                face_npy=Path(str(item["face_npy"])),
                audio_wav=args.audio,
                output_mp4=args.output_dir / f"{emotion}.mp4",
                emotalk_root=args.emotalk_root,
                work_dir=args.work_dir,
                name=str(emotion),
                threads=args.threads,
                fps=args.fps,
                resolution=args.resolution,
                blender=None,
                keep_frames=False,
            )
            item["video"] = str(args.output_dir / f"{emotion}.mp4")
        (args.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(args.output_dir / "report.json")


if __name__ == "__main__":
    main()
