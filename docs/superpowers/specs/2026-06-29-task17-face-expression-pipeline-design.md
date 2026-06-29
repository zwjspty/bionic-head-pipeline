# Task 17: Face Channel Semantics and Emotion Expression Pipeline Design

## Summary

Task 17 upgrades the current face pipeline from:

```text
EmoTalk generates [N,52] -> play exactly what the model produced
```

to:

```text
EmoTalk generates [N,52]
-> channels are probeable and documented
-> verified semantic mapping is configurable
-> emotion/intensity can apply safe, reversible expression postprocess
-> grey-head videos prove visible expression differences
```

The hard rule is: channel semantics must come from probe videos and manual observation. No expression profile may rely on guessed or hard-coded morpheus_52_raw indices.

## Scope

Task 17 is split into three implementation phases:

```text
Task 17A: 52-channel probe, grey-head mapping workflow, expression channel config
Task 17B: emotion/intensity -> ExpressionPostProcessor
Task 17C: expression visual smoke videos
```

This keeps the mapping work separate from expression control. Task 17A can be merged before any emotion modification exists.

## In Scope

- Generate single-channel `[N,52]` probe `.npy` files.
- Generate all 52 channel probes and `manifest.json`.
- Optionally generate a matching silence WAV.
- Document grey-head render workflow for channel observation.
- Create `config/expression_channels.example.json`.
- Validate mapping config:
  - channel indices are `null` or `0..51`;
  - groups only reference legal indices;
  - unverified mapping remains safe by default.
- Implement expression postprocess only after verified mapping exists.
- Apply expression postprocess after face stitching and eye continuity, before UE5 formatting.
- Record expression metrics in stream timing.
- Generate neutral/happy/surprised/sad/thinking grey-head smoke artifacts.

## Out of Scope

- No model training.
- No Student FaceDriver.
- No EmoTalk sidecar changes.
- No ASR, TTS, or LLM provider changes.
- No WebSocket protocol changes.
- No real UE5 integration.
- No AEC, WebRTC, browser UI, or real-time Blender playback.

## Existing Project Context

Relevant existing files:

```text
scripts/render_emotalk_grey_head.py
tests/unit/test_render_emotalk_grey_head.py
src/bionic_head/orchestrators/stream.py
src/bionic_head/face_stitcher.py
src/bionic_head/eye_continuity.py
src/bionic_head/config.py
tests/unit/test_config.py
```

`render_emotalk_grey_head.py` already validates `[N,52]` npy files and shells out to EmoTalk's Blender renderer. Task 17A should build probe files that this existing renderer can consume.

## Architecture

### Probe generation

New script:

```text
scripts/probe_face_channels.py
```

Responsibilities:

- create a `float32` numpy array shaped `[frames, 52]`;
- set exactly one selected channel to `value` for single-channel mode;
- generate all 52 probes in `--all` mode;
- write `manifest.json`;
- optionally write `silence.wav` at 16kHz mono PCM16;
- fail clearly for invalid channel, frame count, fps, or output arguments.

It should be pure Python + NumPy + stdlib. It must not require Blender, EmoTalk, GPU, or a live server.

### Mapping config

New config:

```text
config/expression_channels.example.json
```

Initial state must be conservative:

```json
{
  "format": "morpheus_52_raw",
  "channel_count": 52,
  "verified": false,
  "channels": {
    "jaw_open": null,
    "mouth_smile_left": null,
    "mouth_smile_right": null,
    "mouth_frown_left": null,
    "mouth_frown_right": null,
    "eye_blink_left": null,
    "eye_blink_right": null,
    "brow_up_left": null,
    "brow_up_right": null,
    "brow_down_left": null,
    "brow_down_right": null,
    "cheek_raise_left": null,
    "cheek_raise_right": null
  },
  "groups": {
    "mouth": [],
    "jaw": [],
    "eyes": [],
    "brows": [],
    "cheeks": []
  },
  "notes": {
    "source": "manual probe using scripts/probe_face_channels.py",
    "do_not_use_for_expression_until_verified": true
  }
}
```

`verified=false` means expression postprocess must no-op by default.

### Observation workflow

New documentation:

```text
docs/operations/face-channel-mapping.md
```

It should include:

- probe generation command;
- grey-head render command for one channel;
- batch render loop for all channels;
- observation table for channels 0..51;
- instruction to keep low-confidence channels out of the verified mapping;
- instruction that only medium/high confidence observations may enter expression profiles.

### Expression postprocess

New module:

```text
src/bionic_head/expression.py
```

Conceptual flow:

```text
Audio2Face sidecar
-> FaceSegmentStitcher
-> EyeContinuityProcessor
-> ExpressionPostProcessor
-> UE5 formatter
```

Rules:

- disabled expression is no-op;
- missing mapping is no-op;
- `verified=false` mapping is no-op;
- profile references to unknown/unmapped channels produce warnings/metrics, not crashes;
- intensity scales deltas;
- absolute per-channel delta is clamped by `max_delta`;
- output shape, frame count, fps, timing, and UE5 chunking stay unchanged.

Metrics:

```text
expression_enabled
expression_applied
expression_emotion
expression_intensity
expression_profile_channel_count
expression_max_delta
expression_warning_count
```

### Visual smoke

New script:

```text
scripts/expression_smoke.py
```

It loads a base face npy, applies expression profiles for requested emotions, writes emotion-specific face npy files, optionally renders grey-head videos, and writes `report.json` with metrics.

The render step may be optional because default tests cannot depend on Blender/EmoTalk.

## Testing Strategy

Default tests are hardware-free and provider-free.

Test coverage:

- probe single-channel shape and non-zero channel;
- probe all 52 outputs and manifest;
- silence WAV format;
- invalid channel rejection;
- expression channel config JSON validation;
- expression no-op when mapping missing or unverified;
- expression applies when verified mapping exists;
- intensity increases expression delta;
- stream keeps frame count/fps and includes expression metrics;
- smoke script argument parsing and non-render report generation.

Manual verification:

- render at least one channel probe video;
- batch-render all 52 channel probes when Blender/EmoTalk are available;
- manually fill observation table;
- only after manual observation, set verified mapping for medium/high confidence channels;
- generate neutral vs happy grey-head videos and confirm visible difference.

## Acceptance Criteria

Task 17 is complete when:

```text
1. full pytest passes.
2. scripts/probe_face_channels.py can generate single-channel and all-channel probes.
3. probe outputs are legal [N,52] float32 arrays.
4. manifest.json and optional silence.wav are generated.
5. docs/operations/face-channel-mapping.md contains the observation workflow and 52-row table.
6. config/expression_channels.example.json exists and validates.
7. verified mapping is based on manual observation, not guesses.
8. ExpressionPostProcessor is configurable, disabled by default when mapping is unverified, and reversible.
9. emotion/intensity changes face frames only through verified mapping.
10. stream timing includes expression metrics without changing protocol or frame counts.
11. expression smoke can generate emotion-specific artifacts.
12. grey-head video proves at least happy vs neutral has a visible difference.
```

## First Implementation Slice

Start with Task 17A-1:

```text
scripts/probe_face_channels.py
tests/unit/test_probe_face_channels.py
```

This slice is useful even before any channel semantics are known, and it has no dependency on Blender, EmoTalk, GPU, or real providers.
