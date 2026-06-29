from pathlib import Path
import json
import subprocess
import sys
import wave

import numpy as np
import pytest

from scripts.probe_face_channels import build_manifest, build_probe_frames, write_silence_wav


def test_build_probe_frames_sets_only_selected_channel() -> None:
    frames = build_probe_frames(channel=12, value=1.0, frames=90)

    assert frames.shape == (90, 52)
    assert frames.dtype == np.float32
    assert np.all(frames[:, 12] == np.float32(1.0))
    assert np.count_nonzero(np.delete(frames, 12, axis=1)) == 0


def test_build_probe_frames_rejects_out_of_range_channel() -> None:
    with pytest.raises(ValueError, match=r"channel must be in \[0, 51\]"):
        build_probe_frames(channel=52, value=1.0, frames=90)


def test_write_silence_wav_writes_16k_mono_pcm16(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"

    write_silence_wav(path, seconds=1.0)

    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 16000


def test_build_manifest_records_probe_metadata() -> None:
    manifest = build_manifest(
        [{"channel": 0, "path": "channel-00.npy"}],
        fps=30,
        frames=90,
        value=1.0,
    )

    assert manifest["format"] == "morpheus_52_raw"
    assert manifest["channel_count"] == 52
    assert manifest["fps"] == 30
    assert manifest["frames"] == 90
    assert manifest["value"] == 1.0
    assert manifest["probes"] == [{"channel": 0, "path": "channel-00.npy"}]


def test_probe_cli_generates_single_channel_file(tmp_path: Path) -> None:
    output = tmp_path / "channel-12.npy"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/probe_face_channels.py",
            "--channel",
            "12",
            "--value",
            "1.0",
            "--frames",
            "90",
            "--fps",
            "30",
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    frames = np.load(output)
    assert frames.shape == (90, 52)
    assert np.all(frames[:, 12] == np.float32(1.0))


def test_probe_cli_generates_all_channels_manifest_and_silence(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/probe_face_channels.py",
            "--all",
            "--value",
            "1.0",
            "--frames",
            "30",
            "--fps",
            "30",
            "--output-dir",
            str(tmp_path),
            "--write-silence-wav",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    for channel in range(52):
        assert (tmp_path / f"channel-{channel:02d}.npy").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["probes"]) == 52
    assert (tmp_path / "silence.wav").is_file()
