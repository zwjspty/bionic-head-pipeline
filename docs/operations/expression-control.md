# Expression control workflow

Task 17 adds a lightweight expression postprocess after face stitching and eye
continuity, before UE5 formatting:

```text
Audio2Face
-> FaceSegmentStitcher
-> EyeContinuityProcessor
-> ExpressionPostProcessor
-> UE5 formatter
```

It does not change EmoTalk, the WebSocket protocol, ASR, TTS, or LLM behavior.

## Preconditions

`config/expression_channels.example.json` must contain only channels confirmed
from grey-head probe rendering. Do not add channels from memory or guessed
ARKit/MetaHuman order.

Current status:

```text
format: morpheus_52_raw
verification_status: partial
confirmed examples:
  jaw_open: 24
  mouth_smile_left/right: 43 / 44
  mouth_frown_left/right: 47 / 48
  brow_down_left/right: 0 / 1
unverified:
  blink and cheek channels
```

See [face-channel-mapping.md](face-channel-mapping.md) for the observation table.

## Enable expression postprocess

The example configs include an `expression` block. It is off by default:

```json
{
  "expression": {
    "enabled": false,
    "channel_mapping_path": "config/expression_channels.example.json",
    "max_delta": 0.3,
    "profiles": {
      "happy": {"mouth_smile_left": 0.15, "mouth_smile_right": 0.15},
      "friendly": {"mouth_smile_left": 0.08, "mouth_smile_right": 0.08},
      "surprised": {"jaw_open": 0.18},
      "sad": {"mouth_frown_left": 0.1, "mouth_frown_right": 0.1},
      "thinking": {"brow_down_left": 0.08, "brow_down_right": 0.08}
    }
  }
}
```

For local validation, copy a config and set:

```json
"expression": {
  "enabled": true
}
```

`max_delta` caps each per-channel expression delta. `intensity` from the LLM
scales the configured profile delta.

## Stream metrics

Each segment timing payload can include:

```text
expression_enabled
expression_applied
expression_emotion
expression_intensity
expression_profile_channel_count
expression_max_delta
expression_warning_count
```

If a profile references an unknown or unverified channel, the postprocessor skips
that entry and increments `expression_warning_count`.

## Visual smoke

Use the same base face frames and audio to render several expression variants:

```bash
.venv/bin/python scripts/expression_smoke.py \
  --base-face /tmp/bionic-channel-probes/neutral.npy \
  --audio /tmp/bionic-channel-probes/silence.wav \
  --output-dir /tmp/bionic-expression-smoke \
  --emotions neutral happy surprised sad thinking \
  --resolution 720
```

Expected outputs:

```text
/tmp/bionic-expression-smoke/neutral.mp4
/tmp/bionic-expression-smoke/happy.mp4
/tmp/bionic-expression-smoke/surprised.mp4
/tmp/bionic-expression-smoke/sad.mp4
/tmp/bionic-expression-smoke/thinking.mp4
/tmp/bionic-expression-smoke/report.json
```

For a no-Blender smoke that only writes `.npy` files and `report.json`:

```bash
.venv/bin/python scripts/expression_smoke.py \
  --base-face /tmp/bionic-channel-probes/neutral.npy \
  --audio /tmp/bionic-channel-probes/silence.wav \
  --output-dir /tmp/bionic-expression-smoke \
  --emotions neutral happy surprised sad thinking \
  --skip-render
```

## Safety rules

- Keep expression disabled by default in shared configs until a visual demo has
  been accepted.
- Do not map blink/cheek channels until the grey-head probe observation is clear.
- Do not increase profile deltas to hide bad mapping; fix the mapping first.
- If a visual smoke looks broken, set `expression.enabled=false` to roll back
  without touching the rest of the pipeline.
