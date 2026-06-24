# Task 10 Eye Continuity Design

## Goal

Task 10 adds session-level eye continuity for stream face output, after Task 9 face stitching and before UE5 formatting. It prevents eye/blink-related blendshape channels from behaving like isolated per-segment state while keeping the existing low-latency sidecar, WebSocket protocol, and TTS/ASR/LLM paths unchanged.

The first version is deliberately configuration-driven: it does not assume which indices inside `morpheus_52_raw` are eyes or blinks. If no eye channels are configured, the processor reports metrics and returns frames unchanged.

## Scope

Do:

- Add a pure `EyeContinuityProcessor` with session-aware state.
- Smooth only explicitly configured eye channel indices across same-session, same-turn, same-generation consecutive segments.
- Add an optional deterministic blink scheduler that only modifies explicitly configured blink channel indices.
- Preserve frame count, channel count, timestamps, UE5 chunking, and audio/face alignment.
- Reset smoothing state on `session_id`, `turn_id`, `generation_epoch`, or non-consecutive segment changes.
- Keep blink scheduler state at session level by default, so a human-like blink cadence can continue across turns.
- Record timing metrics for stream, client summaries, and benchmark reports.

Do not:

- Do not hard-code 52-dimensional channel semantics.
- Do not change EmoTalk sidecar, sidecar protocol, or provider behavior.
- Do not change TTS, ASR, LLM, audio overlap, GPU settings, or student FaceDriver.
- Do not add complex gaze or emotion-eye modeling.

## Recommended Approach

Add a pure module:

```text
src/bionic_head/eye_continuity.py
```

Processing order in stream:

```text
Audio2Face sidecar
-> FaceSegmentStitcher
-> EyeContinuityProcessor
-> UE5 formatter
-> server.ue5.frames
```

This order keeps Task 9’s broad blendshape smoothing intact, then applies eye-specific state only to configured channels. It also keeps Task 10 independent of model inference and binary sidecar work.

## Alternatives Considered

1. **Config-driven output processor** — recommended. Safe without a confirmed 52-channel mapping, easy to test, and preserves all existing timing contracts.
2. **Hard-code ARKit-style eye indices** — rejected for Task 10 because `morpheus_52_raw` is explicitly not yet confirmed as ARKit or MetaHuman order.
3. **Modify EmoTalk inputs or sidecar protocol** — rejected because Task 10 is about output continuity, not model inference or transport.

## Components

### `src/bionic_head/eye_continuity.py`

Pure module with no FastAPI, provider, sidecar, or Pydantic dependency.

Primary interface:

```python
@dataclass(frozen=True)
class EyeContinuityMetrics:
    enabled: bool
    applied: bool
    reset: bool
    smooth_channel_count: int
    blink_channel_count: int
    overlap_frames: int
    actual_overlap_frames: int
    boundary_delta_before: float | None
    boundary_delta_after: float | None
    blink_enabled: bool
    blink_applied_count: int
    blink_frame_count: int
    blink_reset_count: int
    global_frame_start: int
    global_frame_end: int


class EyeContinuityProcessor:
    def __init__(
        self,
        *,
        enabled: bool = True,
        eye_smooth_channel_indices: list[int] | None = None,
        blink_enabled: bool = False,
        blink_channel_indices: list[int] | None = None,
        overlap_frames: int = 6,
        blink_interval_min_sec: float = 2.5,
        blink_interval_max_sec: float = 6.0,
        blink_duration_frames: int = 5,
        blink_strength: float = 1.0,
        seed: int = 42,
        reset_blink_on_new_turn: bool = False,
    ) -> None: ...

    def process(
        self,
        frames: list[list[float]],
        *,
        session_id: str,
        turn_id: str,
        generation_epoch: int,
        segment_index: int,
        fps: int,
    ) -> tuple[list[list[float]], EyeContinuityMetrics]: ...

    def reset(self) -> None: ...
```

Rules:

- Disabled processor returns a deep copy unchanged and reports `enabled=False`.
- Empty `eye_smooth_channel_indices` means no smoothing.
- Empty `blink_channel_indices` means no blink modification even when `blink_enabled=True`.
- Smoothing applies only when the previous segment is same `session_id`, same `turn_id`, same `generation_epoch`, and `segment_index == previous + 1`.
- Smoothing only modifies configured `eye_smooth_channel_indices`.
- `actual_overlap = min(overlap_frames, len(previous_tail), len(frames))`.
- Boundary deltas are measured only over configured eye smooth channels:

```text
eye_boundary_delta_before = mean(abs(previous_eye_tail_last - current_eye_first))
eye_boundary_delta_after = mean(abs(previous_eye_tail_last - processed_eye_first))
```

Blink scheduler:

- Maintains `global_frame_index` per processor/session state.
- Uses a deterministic random interval sequence from `seed`.
- First blink is scheduled after an interval between `blink_interval_min_sec` and `blink_interval_max_sec`.
- Blink curve for 5 frames defaults to:

```text
0.0 -> 0.5 -> 1.0 -> 0.5 -> 0.0
```

- Blink writes `max(existing, blink_strength * curve)` into configured blink channels.
- Blink does not change frame count.
- Blink state resets when `session_id` changes.
- Blink state resets on turn change only when `reset_blink_on_new_turn=True`; default is false so blink cadence can behave like session/person state.

### `src/bionic_head/config.py`

Add:

```python
class EyeContinuitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    eye_smooth_channel_indices: list[int] = Field(default_factory=list)
    blink_enabled: bool = False
    blink_channel_indices: list[int] = Field(default_factory=list)
    overlap_frames: int = Field(default=6, ge=0)
    record_boundary_metrics: bool = True
    blink_interval_min_sec: float = Field(default=2.5, gt=0)
    blink_interval_max_sec: float = Field(default=6.0, gt=0)
    blink_duration_frames: int = Field(default=5, ge=1)
    blink_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int = 42
    reset_blink_on_new_turn: bool = False
```

Add `eye_continuity: EyeContinuitySettings` to `AppSettings`.

Configuration files include:

```json
"eye_continuity": {
  "enabled": true,
  "eye_smooth_channel_indices": [],
  "blink_enabled": false,
  "blink_channel_indices": [],
  "overlap_frames": 6,
  "record_boundary_metrics": true,
  "blink_interval_min_sec": 2.5,
  "blink_interval_max_sec": 6.0,
  "blink_duration_frames": 5,
  "blink_strength": 1.0,
  "seed": 42,
  "reset_blink_on_new_turn": false
}
```

Validation:

- Channel indices must be in `[0, 51]`.
- `blink_interval_max_sec >= blink_interval_min_sec`.

### `src/bionic_head/orchestrators/stream.py`

Create one `EyeContinuityProcessor` per `StreamOrchestrator.run(...)` call. Apply it after `FaceSegmentStitcher.stitch(...)` and before `server.face.frames` / UE5 formatting.

Add timing payload keys:

- `eye_continuity_enabled`
- `eye_continuity_applied`
- `eye_continuity_reset`
- `eye_smooth_channel_count`
- `eye_continuity_overlap_frames`
- `eye_continuity_actual_overlap_frames`
- `eye_boundary_delta_before`
- `eye_boundary_delta_after`
- `blink_enabled`
- `blink_applied_count`
- `blink_frame_count`
- `blink_reset_count`
- `eye_global_frame_start`
- `eye_global_frame_end`

Existing stale-drop checks remain authoritative. If a segment becomes stale before emit, processed eye frames are still discarded and not sent.

### `scripts/stream_client.py`

Existing timing merge already carries boolean and numeric keys into per-segment summaries. Tests should prove eye continuity timing survives into `summary["segments"]`.

### `scripts/benchmark.py`

Promote eye metrics into stream benchmark reports:

- First-segment configuration/state metrics:
  - `eye_continuity_enabled`
  - `eye_smooth_channel_count`
  - `eye_continuity_overlap_frames`
  - `eye_continuity_actual_overlap_frames`
  - `eye_boundary_delta_before`
  - `eye_boundary_delta_after`
- Count metrics across all segments:
  - `eye_continuity_applied_count`
  - `eye_continuity_reset_count`
  - `blink_applied_count`
  - `blink_frame_count`
  - `blink_reset_count`

## Error Handling

- Invalid frame shape or out-of-range configured channels raises `ValueError` in the pure processor.
- Stream integration does not hide Audio2Face, stitcher, or UE5 formatter errors.
- Disabled processor must not alter frames.
- Empty channel configs must be safe no-ops.

## Testing Strategy

Unit tests:

- No configured channels means no-op frames and `eye_continuity_applied=False`.
- Configured eye channels modify only those channels.
- Boundary delta after smoothing is less than or equal to before.
- Same session multi-segment processing maintains global frame index.
- Session change resets state.
- Turn/generation changes reset smoothing and do not blend old output into new segment.
- Blink scheduler emits deterministic blink curve when configured.
- Blink does not change frame count.
- Disabled processor returns unchanged frames.

Stream tests:

- Timing for `server.face.frames` and `server.ue5.frames` includes eye continuity metrics.
- With configured eye channels, second segment reports `eye_continuity_applied=True`.
- Existing stale/cancel tests continue to prove old turn frames do not leak.

Benchmark/client tests:

- Stream client preserves eye timing in segment summaries.
- Benchmark extracts and counts eye continuity metrics across segments.

Real smoke:

- Existing `BIONIC_HEAD_RUN_REAL_EMOTALK=1` stream smoke remains default skipped.
- Real smoke only asserts metrics exist and stale leak counts stay zero; it does not require blink channels to be configured.

## Acceptance

Minimum acceptance:

```text
pytest all passes
eye_boundary_delta_after <= eye_boundary_delta_before in deterministic unit/stream tests
old_turn_face_leak_count = 0 remains true in stale/drop coverage
stream benchmark reports eye continuity and blink metrics
default config does not alter frames without explicit channel indices
```

Expected interpretation:

```text
Task 10 provides the safe framework for session-level eye continuity.
Real visual changes only happen after the 52-channel eye/blink mapping is explicitly configured.
```
