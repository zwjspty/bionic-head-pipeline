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

## Rendered verification artifacts

The first partial mapping was verified from these local grey-head artifacts:

```text
/tmp/bionic-channel-probes/
/tmp/bionic-channel-videos/
/tmp/bionic-channel-pairs/contact-sheet.png
/tmp/bionic-channel-sheets/upper_00_23.png
/tmp/bionic-channel-sheets/mouth_24_51.png
/tmp/bionic-channel-eye-crops/eye-candidates.png
```

The generated videos are temporary artifacts and are not committed. Re-run the
commands above to reproduce them.

## Observation table

| Channel | Observed effect | Confidence | Notes |
|---:|---|---|---|
| 0 | brow down / inner brow crease on viewer-right side | medium | mapped as `brow_down_left`; left/right inferred from frontal render |
| 1 | brow down / forehead ridge on viewer-left side | medium | mapped as `brow_down_right`; left/right inferred from frontal render |
| 2 | brow/upper eyelid lift, broad upper-face change | low | keep out of expression profiles until rechecked |
| 3 | one-sided brow/eye opening on viewer-right side | low | eye/brow mixed; not mapped as blink |
| 4 | brow squeeze / eye squint | medium | regional brow group only |
| 5 | subtle nose/cheek/lower-face change | low | no semantic mapping |
| 6 | subtle nose/cheek asymmetry | low | no semantic mapping |
| 7 | subtle eye/cheek change | low | no semantic mapping |
| 8 | viewer-right eyelid squint/partial close | medium | regional eye/brow group only; not mapped as blink yet |
| 9 | subtle eye/cheek change | low | no semantic mapping |
| 10 | subtle brow/eye change | low | no semantic mapping |
| 11 | subtle brow/eye change | low | no semantic mapping |
| 12 | subtle cheek/eye change | low | no semantic mapping |
| 13 | subtle cheek/eye change | low | no semantic mapping |
| 14 | subtle viewer-left eyelid/eye shape change | medium | regional eye/brow group only; not mapped as blink yet |
| 15 | subtle eye/cheek change | low | no semantic mapping |
| 16 | subtle nose/cheek change | low | no semantic mapping |
| 17 | subtle eye/cheek change | low | no semantic mapping |
| 18 | subtle brow/eye change | low | no semantic mapping |
| 19 | subtle brow/eye change | low | no semantic mapping |
| 20 | viewer-right eyelid/eye opening shape change | medium | regional eye/brow group only |
| 21 | subtle eye/cheek change | low | no semantic mapping |
| 22 | lower cheek / mouth-area compression | low | no semantic mapping |
| 23 | mouth corner / lower cheek asymmetry | low | no semantic mapping |
| 24 | jaw open / mouth open | high | mapped as `jaw_open` |
| 25 | subtle mouth corner / cheek pull | low | no semantic mapping |
| 26 | lip press / mouth compression | medium | regional mouth group only |
| 27 | subtle mouth corner / lower cheek change | low | no semantic mapping |
| 28 | subtle mouth corner pull | low | no semantic mapping |
| 29 | subtle lower lip / mouth change | low | no semantic mapping |
| 30 | subtle mouth / lower-face change | low | no semantic mapping |
| 31 | rounded lip pucker, “oo” shape | high | mapped as `mouth_pucker` |
| 32 | one-sided mouth corner smile/pull on viewer-left side | medium | regional mouth group only |
| 33 | mouth twist / side pull | low | no semantic mapping |
| 34 | mouth protrude / side pull | low | no semantic mapping |
| 35 | mild mouth corner lift | low | no semantic mapping |
| 36 | lower lip / chin compression | low | no semantic mapping |
| 37 | narrow lip funnel / pucker | high | mapped as `mouth_funnel` |
| 38 | slight smile / mouth-corner lift | medium | regional mouth group only |
| 39 | lip press / mouth-corner downturn | medium | mapped as `mouth_press` |
| 40 | lip press / lower-lip raise | medium | regional mouth group only |
| 41 | subtle mouth/nasolabial change | low | no semantic mapping |
| 42 | subtle mouth-corner downturn | low | no semantic mapping |
| 43 | smile / mouth corner lift on viewer-right side | high | mapped as `mouth_smile_left`; left/right inferred from frontal render |
| 44 | smile / mouth corner lift on viewer-left side | high | mapped as `mouth_smile_right`; left/right inferred from frontal render |
| 45 | subtle mouth/lip downturn | low | no semantic mapping |
| 46 | subtle lower mouth change | low | no semantic mapping |
| 47 | frown / mouth corner down on viewer-right side | medium | mapped as `mouth_frown_left`; left/right inferred from frontal render |
| 48 | frown / mouth corner down on viewer-left side | medium | mapped as `mouth_frown_right`; left/right inferred from frontal render |
| 49 | subtle brow/mouth change | low | no semantic mapping |
| 50 | subtle nose/mouth asymmetry | low | no semantic mapping |
| 51 | subtle mouth corner / cheek change | low | no semantic mapping |
