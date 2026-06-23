from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_EMOTALK_ROOT = Path("/home/user/code/EmoTalk_release")
DEFAULT_FPS = 30
DEFAULT_RESOLUTION = "512x768"


@dataclass(frozen=True)
class StagedRenderInputs:
    root_arg: str
    name: str
    staged_npy: Path
    frame_dir: Path


def validate_face_npy(face_npy: Path) -> int:
    try:
        coeffs = np.load(face_npy, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load face npy: {face_npy}") from exc
    if coeffs.ndim != 2 or coeffs.shape[1] != 52:
        raise ValueError(f"Expected face coefficients shaped [N, 52], got {coeffs.shape}")
    if coeffs.shape[0] <= 0:
        raise ValueError("Expected at least one face frame")
    if not np.isfinite(coeffs).all():
        raise ValueError("Face coefficients must be finite")
    return int(coeffs.shape[0])


def prepare_render_inputs(face_npy: Path, work_dir: Path, *, name: str) -> StagedRenderInputs:
    work_dir.mkdir(parents=True, exist_ok=True)
    staged_npy = work_dir / f"{name}.npy"
    frame_dir = work_dir / name
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    shutil.copy2(face_npy, staged_npy)
    return StagedRenderInputs(
        root_arg=f"{work_dir.resolve()}/",
        name=name,
        staged_npy=staged_npy,
        frame_dir=frame_dir,
    )


def build_blender_command(
    *,
    blender: Path,
    render_blend: Path,
    render_py: Path,
    root_arg: str,
    name: str,
    threads: int,
) -> list[str]:
    return [
        str(blender),
        "-t",
        str(threads),
        "-b",
        str(render_blend),
        "-P",
        str(render_py),
        "--",
        root_arg,
        name,
    ]


def build_ffmpeg_command(
    *,
    frame_dir: Path,
    audio_wav: Path,
    output_mp4: Path,
    fps: int,
    resolution: str,
) -> list[str]:
    return [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-r",
        str(fps),
        "-i",
        str(frame_dir / "%d.png"),
        "-i",
        str(audio_wav),
        "-pix_fmt",
        "yuv420p",
        "-s",
        resolution,
        "-shortest",
        str(output_mp4),
    ]


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def render_grey_head(
    *,
    face_npy: Path,
    audio_wav: Path,
    output_mp4: Path,
    emotalk_root: Path,
    blender: Path | None = None,
    work_dir: Path,
    name: str,
    threads: int,
    fps: int,
    resolution: str,
    keep_frames: bool,
) -> None:
    require_file(face_npy, "face npy")
    require_file(audio_wav, "audio wav")
    frame_count = validate_face_npy(face_npy)

    blender_path = blender or (emotalk_root / "blender" / "blender")
    render_blend = emotalk_root / "render.blend"
    render_py = emotalk_root / "render.py"
    require_file(blender_path, "Blender executable")
    require_file(render_blend, "EmoTalk render.blend")
    require_file(render_py, "EmoTalk render.py")

    staged = prepare_render_inputs(face_npy, work_dir, name=name)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        build_blender_command(
            blender=blender_path,
            render_blend=render_blend,
            render_py=render_py,
            root_arg=staged.root_arg,
            name=staged.name,
            threads=threads,
        ),
        cwd=emotalk_root,
        check=True,
    )

    rendered_frames = len(list(staged.frame_dir.glob("*.png")))
    if rendered_frames != frame_count:
        raise RuntimeError(f"Rendered {rendered_frames} frames, expected {frame_count}")

    subprocess.run(
        build_ffmpeg_command(
            frame_dir=staged.frame_dir,
            audio_wav=audio_wav,
            output_mp4=output_mp4,
            fps=fps,
            resolution=resolution,
        ),
        check=True,
    )

    if not keep_frames:
        shutil.rmtree(staged.frame_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render EmoTalk 52D coefficients as a 3D grey-head MP4.")
    parser.add_argument("--face-npy", type=Path, required=True, help="Input [N, 52] blendshape .npy.")
    parser.add_argument("--audio-wav", type=Path, required=True, help="Input speech wav to mux into the MP4.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--emotalk-root", type=Path, default=DEFAULT_EMOTALK_ROOT)
    parser.add_argument("--blender", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/bionic-emotalk-grey-render"))
    parser.add_argument("--name", default="preview")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_grey_head(
        face_npy=args.face_npy,
        audio_wav=args.audio_wav,
        output_mp4=args.output,
        emotalk_root=args.emotalk_root,
        blender=args.blender,
        work_dir=args.work_dir,
        name=args.name,
        threads=args.threads,
        fps=args.fps,
        resolution=args.resolution,
        keep_frames=args.keep_frames,
    )
    print(args.output)


if __name__ == "__main__":
    main()
