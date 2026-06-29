# Face channel mapping workflow

This document records how to probe `morpheus_52_raw` / EmoTalk 52D channels.

The rule is intentionally strict: only channels observed with medium or high confidence may be copied into `config/expression_channels.example.json`. Low confidence channels must remain `null` or be documented only in notes.

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
