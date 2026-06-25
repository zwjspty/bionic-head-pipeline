from pathlib import Path

import numpy as np
import pytest

from scripts.render_emotalk_grey_head import (
    build_blender_command,
    build_ffmpeg_command,
    prepare_render_inputs,
    validate_face_npy,
)


def test_validate_face_npy_accepts_n_by_52(tmp_path: Path) -> None:
    face_path = tmp_path / "face.npy"
    np.save(face_path, np.ones((4, 52), dtype=np.float32))

    assert validate_face_npy(face_path) == 4


def test_validate_face_npy_rejects_wrong_shape(tmp_path: Path) -> None:
    face_path = tmp_path / "face.npy"
    np.save(face_path, np.ones((4, 51), dtype=np.float32))

    with pytest.raises(ValueError, match=r"\[N, 52\]"):
        validate_face_npy(face_path)


def test_prepare_render_inputs_stages_npy_and_frame_directory(tmp_path: Path) -> None:
    face_path = tmp_path / "source.npy"
    work_dir = tmp_path / "work"
    np.save(face_path, np.arange(104, dtype=np.float32).reshape(2, 52))

    staged = prepare_render_inputs(face_path, work_dir, name="preview")

    assert staged.root_arg == f"{work_dir.resolve()}/"
    assert staged.name == "preview"
    assert staged.staged_npy == work_dir / "preview.npy"
    assert staged.frame_dir == work_dir / "preview"
    assert staged.frame_dir.is_dir()
    np.testing.assert_array_equal(np.load(staged.staged_npy), np.load(face_path))


def test_builds_blender_command_with_existing_emotalk_paths(tmp_path: Path) -> None:
    emotalk_root = tmp_path / "EmoTalk_release"
    blender = emotalk_root / "blender" / "blender"
    render_blend = emotalk_root / "render.blend"
    render_py = emotalk_root / "render.py"

    command = build_blender_command(
        blender=blender,
        render_blend=render_blend,
        render_py=render_py,
        root_arg="/tmp/render-root/",
        name="preview",
        threads=8,
    )

    assert command == [
        str(blender),
        "-t",
        "8",
        "-b",
        str(render_blend),
        "-P",
        str(render_py),
        "--",
        "/tmp/render-root/",
        "preview",
    ]


def test_builds_ffmpeg_command_for_numbered_png_frames(tmp_path: Path) -> None:
    command = build_ffmpeg_command(
        frame_dir=tmp_path / "preview",
        audio_wav=tmp_path / "reply.wav",
        output_mp4=tmp_path / "preview.mp4",
        fps=30,
        resolution="512x768",
    )

    assert command == [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-r",
        "30",
        "-i",
        str(tmp_path / "preview" / "%d.png"),
        "-i",
        str(tmp_path / "reply.wav"),
        "-pix_fmt",
        "yuv420p",
        "-s",
        "512x768",
        "-shortest",
        str(tmp_path / "preview.mp4"),
    ]


def test_builds_ffmpeg_command_accepts_square_resolution_shorthand(tmp_path: Path) -> None:
    command = build_ffmpeg_command(
        frame_dir=tmp_path / "preview",
        audio_wav=tmp_path / "reply.wav",
        output_mp4=tmp_path / "preview.mp4",
        fps=30,
        resolution="720",
    )

    assert command[command.index("-s") + 1] == "720x720"
