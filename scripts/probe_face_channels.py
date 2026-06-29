from __future__ import annotations

import argparse
import json
import wave
from array import array
from pathlib import Path

import numpy as np


CHANNEL_COUNT = 52


def build_probe_frames(
    *,
    channel: int,
    value: float,
    frames: int,
    channel_count: int = CHANNEL_COUNT,
) -> np.ndarray:
    if channel < 0 or channel >= channel_count:
        raise ValueError(f"channel must be in [0, {channel_count - 1}]")
    if frames < 1:
        raise ValueError("frames must be >= 1")
    output = np.zeros((frames, channel_count), dtype=np.float32)
    output[:, channel] = np.float32(value)
    return output


def write_silence_wav(path: Path, *, seconds: float, sample_rate: int = 16000) -> None:
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = array("h", [0] * int(round(seconds * sample_rate)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def build_manifest(
    records: list[dict[str, object]],
    *,
    fps: int,
    frames: int,
    value: float,
) -> dict[str, object]:
    return {
        "format": "morpheus_52_raw",
        "channel_count": CHANNEL_COUNT,
        "fps": fps,
        "frames": frames,
        "value": value,
        "probes": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate morpheus_52_raw single-channel face probes.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--channel", type=int)
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--value", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--write-silence-wav", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")
    if args.fps < 1:
        raise SystemExit("--fps must be >= 1")

    if args.all:
        if args.output_dir is None:
            raise SystemExit("--all requires --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        for channel in range(CHANNEL_COUNT):
            path = args.output_dir / f"channel-{channel:02d}.npy"
            np.save(path, build_probe_frames(channel=channel, value=args.value, frames=args.frames))
            records.append({"channel": channel, "path": path.name})
        manifest = build_manifest(records, fps=args.fps, frames=args.frames, value=args.value)
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if args.write_silence_wav:
            write_silence_wav(args.output_dir / "silence.wav", seconds=args.frames / args.fps)
        print(args.output_dir)
        return

    if args.output is None:
        raise SystemExit("--channel requires --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, build_probe_frames(channel=args.channel, value=args.value, frames=args.frames))
    print(args.output)


if __name__ == "__main__":
    main()
