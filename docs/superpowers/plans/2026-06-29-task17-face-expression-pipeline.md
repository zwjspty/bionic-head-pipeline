# Task 17 Face Expression Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the probe, mapping, expression postprocess, and visual smoke workflow needed to understand and lightly control morpheus_52_raw / EmoTalk 52D face expressions.

**Architecture:** Implement Task 17 in independent slices: probe generation first, mapping config validation second, expression postprocess third, stream integration fourth, and visual smoke last. The probe/mapping slices are safe to merge before any emotion enhancement because unverified mappings keep expression logic disabled/no-op.

**Tech Stack:** Python 3.11 target, NumPy, stdlib `wave/json/argparse`, Pydantic v2 config patterns already in `src/bionic_head/config.py`, pytest, existing `scripts/render_emotalk_grey_head.py` for optional manual rendering.

## Global Constraints

- Do not train models.
- Do not implement Student FaceDriver.
- Do not change EmoTalk sidecar.
- Do not change ASR, TTS, or LLM providers.
- Do not change the WebSocket protocol.
- Do not integrate real UE5.
- Default tests must not require GPU, EmoTalk, Blender, microphone, speaker, or real providers.
- Channel semantics must come from probe videos and manual observation.
- Expression profiles must not hard-code guessed channel indices.

---

## File Structure

- Create `scripts/probe_face_channels.py`
  - Generates single-channel/all-channel `[N,52]` float32 probe `.npy` files.
  - Writes `manifest.json` and optional `silence.wav`.
- Create `tests/unit/test_probe_face_channels.py`
  - Covers single-channel, all-channel, manifest, silence WAV, invalid channel.
- Create `config/expression_channels.example.json`
  - Conservative unverified mapping template.
- Create `src/bionic_head/expression.py`
  - Loads and validates expression channel mapping.
  - Later owns `ExpressionPostProcessor`.
- Create `tests/unit/test_expression_channel_config.py`
  - Validates example JSON and index rules.
- Create `docs/operations/face-channel-mapping.md`
  - Documents probe generation, grey-head rendering, and observation table.
- Later modify:
  - `src/bionic_head/config.py`
  - `src/bionic_head/orchestrators/stream.py`
  - `tests/unit/test_config.py`
  - `tests/unit/test_stream_orchestrator.py`
- Later create:
  - `tests/unit/test_expression_postprocess.py`
  - `scripts/expression_smoke.py`
  - `tests/unit/test_expression_smoke.py`

---

### Task 1: Face Channel Probe Generator

**Files:**
- Create: `scripts/probe_face_channels.py`
- Create: `tests/unit/test_probe_face_channels.py`

**Interfaces:**
- Produces:
  - `build_probe_frames(*, channel: int, value: float, frames: int, channel_count: int = 52) -> np.ndarray`
  - `write_silence_wav(path: Path, *, seconds: float, sample_rate: int = 16000) -> None`
  - `build_manifest(records: list[dict[str, object]], *, fps: int, frames: int, value: float) -> dict[str, object]`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_probe_face_channels.py`:

```python
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
    with pytest.raises(ValueError, match="channel must be in \\[0, 51\\]"):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_probe_face_channels.py -q
```

Expected: FAIL because `scripts.probe_face_channels` does not exist.

- [ ] **Step 3: Implement probe generator**

Create `scripts/probe_face_channels.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_probe_face_channels.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_face_channels.py tests/unit/test_probe_face_channels.py
git commit -m "feat: add face channel probe generator"
```

---

### Task 2: Expression Channel Mapping Config

**Files:**
- Create: `config/expression_channels.example.json`
- Create: `src/bionic_head/expression.py`
- Create: `tests/unit/test_expression_channel_config.py`

**Interfaces:**
- Consumes: JSON mapping file.
- Produces:
  - `ExpressionChannelMapping`
  - `load_expression_channel_mapping(path: Path) -> ExpressionChannelMapping`

- [ ] **Step 1: Write failing mapping tests**

Create `tests/unit/test_expression_channel_config.py`:

```python
from pathlib import Path
import json

import pytest

from bionic_head.expression import load_expression_channel_mapping


def test_expression_channel_example_is_parseable_and_unverified() -> None:
    mapping = load_expression_channel_mapping(Path("config/expression_channels.example.json"))

    assert mapping.format == "morpheus_52_raw"
    assert mapping.channel_count == 52
    assert mapping.verified is False
    assert mapping.channels["jaw_open"] is None
    assert mapping.groups["mouth"] == []


def test_expression_channel_mapping_rejects_channel_index_outside_52(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "format": "morpheus_52_raw",
                "channel_count": 52,
                "verified": True,
                "channels": {"jaw_open": 52},
                "groups": {},
                "notes": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="channel index must be null or in \\[0, 51\\]"):
        load_expression_channel_mapping(path)


def test_expression_channel_mapping_rejects_group_index_outside_52(tmp_path: Path) -> None:
    path = tmp_path / "bad-group.json"
    path.write_text(
        json.dumps(
            {
                "format": "morpheus_52_raw",
                "channel_count": 52,
                "verified": True,
                "channels": {"jaw_open": 1},
                "groups": {"jaw": [1, 99]},
                "notes": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="group channel index must be in \\[0, 51\\]"):
        load_expression_channel_mapping(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_expression_channel_config.py -q
```

Expected: FAIL because `bionic_head.expression` and config file do not exist.

- [ ] **Step 3: Add conservative mapping config and loader**

Create `config/expression_channels.example.json` with the JSON shown in the design spec.

Create `src/bionic_head/expression.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ExpressionChannelMapping:
    format: str
    channel_count: int
    verified: bool
    channels: dict[str, int | None]
    groups: dict[str, list[int]]
    notes: dict[str, object]


def load_expression_channel_mapping(path: Path) -> ExpressionChannelMapping:
    body = json.loads(path.read_text(encoding="utf-8"))
    channel_count = int(body.get("channel_count", 52))
    channels = dict(body.get("channels") or {})
    groups = dict(body.get("groups") or {})

    for value in channels.values():
        if value is None:
            continue
        if not isinstance(value, int) or value < 0 or value >= channel_count:
            raise ValueError(f"channel index must be null or in [0, {channel_count - 1}]")

    normalized_groups: dict[str, list[int]] = {}
    for group_name, values in groups.items():
        if not isinstance(values, list):
            raise ValueError("group values must be lists")
        normalized: list[int] = []
        for value in values:
            if not isinstance(value, int) or value < 0 or value >= channel_count:
                raise ValueError(f"group channel index must be in [0, {channel_count - 1}]")
            normalized.append(value)
        normalized_groups[str(group_name)] = normalized

    return ExpressionChannelMapping(
        format=str(body.get("format", "")),
        channel_count=channel_count,
        verified=bool(body.get("verified", False)),
        channels={str(key): value for key, value in channels.items()},
        groups=normalized_groups,
        notes=dict(body.get("notes") or {}),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_expression_channel_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/expression_channels.example.json src/bionic_head/expression.py tests/unit/test_expression_channel_config.py
git commit -m "feat: add expression channel mapping config"
```

---

### Task 3: Face Channel Mapping Operations Doc

**Files:**
- Create: `docs/operations/face-channel-mapping.md`

**Interfaces:**
- Consumes: `scripts/probe_face_channels.py`
- Produces: documented command workflow and 52-row observation table.

- [ ] **Step 1: Create documentation**

Create `docs/operations/face-channel-mapping.md` with:

```markdown
# Face channel mapping workflow

This document records how to probe morpheus_52_raw / EmoTalk 52D channels.

## Generate one probe

```bash
.venv/bin/python scripts/probe_face_channels.py \
  --channel 12 \
  --value 1.0 \
  --frames 90 \
  --fps 30 \
  --output /tmp/channel-12.npy
```

## Generate all probes

```bash
.venv/bin/python scripts/probe_face_channels.py \
  --all \
  --value 1.0 \
  --frames 90 \
  --fps 30 \
  --output-dir /tmp/bionic-channel-probes \
  --write-silence-wav
```

## Render one grey-head video

```bash
.venv/bin/python scripts/render_emotalk_grey_head.py \
  --face-npy /tmp/bionic-channel-probes/channel-12.npy \
  --audio-wav /tmp/bionic-channel-probes/silence.wav \
  --output /tmp/bionic-channel-12.mp4 \
  --resolution 720
```

## Batch render

```bash
mkdir -p /tmp/bionic-channel-videos

for i in $(seq -w 0 51); do
  .venv/bin/python scripts/render_emotalk_grey_head.py \
    --face-npy /tmp/bionic-channel-probes/channel-${i}.npy \
    --audio-wav /tmp/bionic-channel-probes/silence.wav \
    --output /tmp/bionic-channel-videos/channel-${i}.mp4 \
    --resolution 720
done
```

## Mapping rule

Only channels observed with medium or high confidence may be copied into `config/expression_channels.example.json`.

Low confidence channels must remain `null` or be documented only in notes.

## Observation table

| Channel | Observed effect | Confidence | Notes |
|---:|---|---|---|
| 0 | pending | low | |
| 1 | pending | low | |
| 2 | pending | low | |
| 3 | pending | low | |
| 4 | pending | low | |
| 5 | pending | low | |
| 6 | pending | low | |
| 7 | pending | low | |
| 8 | pending | low | |
| 9 | pending | low | |
| 10 | pending | low | |
| 11 | pending | low | |
| 12 | pending | low | |
| 13 | pending | low | |
| 14 | pending | low | |
| 15 | pending | low | |
| 16 | pending | low | |
| 17 | pending | low | |
| 18 | pending | low | |
| 19 | pending | low | |
| 20 | pending | low | |
| 21 | pending | low | |
| 22 | pending | low | |
| 23 | pending | low | |
| 24 | pending | low | |
| 25 | pending | low | |
| 26 | pending | low | |
| 27 | pending | low | |
| 28 | pending | low | |
| 29 | pending | low | |
| 30 | pending | low | |
| 31 | pending | low | |
| 32 | pending | low | |
| 33 | pending | low | |
| 34 | pending | low | |
| 35 | pending | low | |
| 36 | pending | low | |
| 37 | pending | low | |
| 38 | pending | low | |
| 39 | pending | low | |
| 40 | pending | low | |
| 41 | pending | low | |
| 42 | pending | low | |
| 43 | pending | low | |
| 44 | pending | low | |
| 45 | pending | low | |
| 46 | pending | low | |
| 47 | pending | low | |
| 48 | pending | low | |
| 49 | pending | low | |
| 50 | pending | low | |
| 51 | pending | low | |
```

- [ ] **Step 2: Verify docs contain all 52 rows**

Run:

```bash
rg -n "^\\| [0-9]+ \\|" docs/operations/face-channel-mapping.md | wc -l
```

Expected output: `52`

- [ ] **Step 3: Commit**

```bash
git add docs/operations/face-channel-mapping.md
git commit -m "docs: document face channel mapping workflow"
```

---

## Later Task Outline

These tasks are intentionally not expanded into line-by-line implementation steps until Task 17A probe and mapping slices land:

```text
Task 4: ExpressionPostProcessor unit implementation
Task 5: Expression config in AppSettings
Task 6: Stream integration after EyeContinuity before UE5 formatter
Task 7: expression_smoke.py non-render and optional render workflow
Task 8: verified mapping update after manual grey-head observation
```

Before starting Task 4, write a follow-up plan section based on the actual verified mapping state.

## Final Verification for Task 17A Slice

- [ ] Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_probe_face_channels.py tests/unit/test_expression_channel_config.py tests/unit/test_render_emotalk_grey_head.py -q
PYTHONPATH=src .venv/bin/python -m pytest -q
```

- [ ] Confirm:

```text
probe generator tests pass
expression mapping config tests pass
full pytest passes
```

- [ ] Ask user to run manual grey-head probe rendering if Blender/EmoTalk are available.
