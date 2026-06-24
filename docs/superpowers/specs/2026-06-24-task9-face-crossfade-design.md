# Task 9 Face Crossfade Design

## Goal

Task 9 makes stream face output visually continuous across adjacent TTS segments by applying a small turn-aware crossfade to blendshape frames after Audio2Face inference and before `server.face.frames` / `server.ue5.frames` emission.

This task proves “连续且不跳”: same-turn segment boundaries should have lower blendshape jump magnitude, while new turns, new sessions, interrupts, and generation changes must not inherit old face state.

## Scope

Do:

- Apply output blendshape-level crossfade only.
- Smooth same-session, same-turn, same-generation consecutive segments.
- Reset stitching state when `session_id`, `turn_id`, `generation_epoch`, or segment continuity changes.
- Preserve `frame_count`, frame timestamps, UE5 chunking, and audio/face duration alignment.
- Record boundary metrics:
  - `face_stitch_enabled`
  - `face_stitch_applied_count`
  - `face_stitch_reset_count`
  - `face_stitch_overlap_frames`
  - `face_stitch_actual_overlap_frames`
  - `face_boundary_delta_before`
  - `face_boundary_delta_after`

Do not:

- Do not change EmoTalk input audio.
- Do not change sidecar protocol.
- Do not change TTS, ASR, or LLM behavior.
- Do not add GPU logic.
- Do not add student FaceDriver.
- Do not implement audio overlap.
- Do not split mouth/expression/eye channel groups in this first version.

## Recommended Approach

Use output-head replacement:

```text
current_frames[:actual_overlap] = crossfade(previous_tail, current_head)
return current_frames
```

This keeps the segment length unchanged. It is safer than concatenating old tail and new head because it does not alter frame count, PTS alignment, UE5 chunk count, or segment send timing.

The first version uses one uniform `overlap_frames` value across all 52 channels. Mouth/expression/eye channel groups remain a later refinement after the Morpheus 52-channel semantic mapping is confirmed.

## Alternatives Considered

1. **Output-head replacement crossfade** — recommended. Low risk, preserves shape and timing, easy to test.
2. **Concatenate old tail + blended overlap + new remainder** — rejected for Task 9 because it changes per-segment frame count and can desync UE5 playback from TTS audio.
3. **Audio input overlap before EmoTalk** — rejected for Task 9 because it touches TTS/audio segmentation and sidecar worker assumptions. It belongs after output smoothing is measured.

## Components

### `src/bionic_head/face_stitcher.py`

New pure module with no FastAPI, sidecar, or provider dependency.

Primary interface:

```python
@dataclass(frozen=True)
class FaceStitchKey:
    session_id: str
    turn_id: str
    generation_epoch: int


@dataclass(frozen=True)
class FaceStitchMetrics:
    enabled: bool
    applied: bool
    reset: bool
    overlap_frames: int
    actual_overlap_frames: int
    boundary_delta_before: float | None = None
    boundary_delta_after: float | None = None


class FaceSegmentStitcher:
    def __init__(self, *, enabled: bool = True, overlap_frames: int = 8) -> None: ...

    def stitch(
        self,
        frames: list[list[float]],
        *,
        session_id: str,
        turn_id: str,
        generation_epoch: int,
        segment_index: int,
    ) -> tuple[list[list[float]], FaceStitchMetrics]: ...

    def reset(self) -> None: ...
```

Rules:

- If disabled, return a deep copy of frames and metrics with `applied=False`.
- If this is the first segment for a key, return frames unchanged and store its tail.
- If key changes, reset first, return frames unchanged, store new tail.
- If `segment_index` is not exactly previous segment index + 1, reset first, return frames unchanged, store new tail.
- If `actual_overlap = min(overlap_frames, len(previous_tail), len(frames))` is 0, return unchanged and store new tail.
- Otherwise blend the first `actual_overlap` frames of current segment using previous tail:

```text
alpha = (i + 1) / actual_overlap
stitched_frame = previous_tail_frame * (1 - alpha) + current_head_frame * alpha
```

This formula keeps the last blended frame equal to the original current frame when `i == actual_overlap - 1`, so the crossfade exits cleanly into the unmodified current segment.

Boundary metrics:

```text
boundary_delta_before = mean(abs(previous_tail[-1] - current_frames[0]))
boundary_delta_after = mean(abs(previous_tail[-1] - stitched_frames[0]))
```

Expected acceptance:

```text
boundary_delta_after < boundary_delta_before
```

for intentionally discontinuous adjacent segments.

### `src/bionic_head/config.py`

Add:

```python
class FaceStitchingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    overlap_frames: int = Field(default=8, ge=0)
    reset_on_new_turn: bool = True
    record_boundary_metrics: bool = True
```

Add `face_stitching: FaceStitchingSettings` to `AppSettings`.

Configuration files include:

```json
"face_stitching": {
  "enabled": true,
  "overlap_frames": 8,
  "reset_on_new_turn": true,
  "record_boundary_metrics": true
}
```

### `src/bionic_head/orchestrators/stream.py`

Create one `FaceSegmentStitcher` per `StreamOrchestrator.run(...)` call. Apply it inside `_process_face_segment` immediately after `audio2face.drive(...)` returns and before:

- `server.face.frames`
- UE5 formatting
- `server.ue5.frames`

Update segment timing payload with stitch metrics. Keep existing stale-drop checks and do not emit stale stitched frames.

### `scripts/stream_client.py`

Merge numeric stitch metrics from event `payload["timing"]` into per-segment summary. Existing Task 8 timing merge already handles numeric keys, so only tests may be needed unless metrics need top-level aliases.

### `scripts/benchmark.py`

Promote first segment stitch metrics into stream benchmark metrics:

- `face_stitch_overlap_frames`
- `face_stitch_actual_overlap_frames`
- `face_boundary_delta_before`
- `face_boundary_delta_after`
- `face_stitch_applied_count`
- `face_stitch_reset_count`

## Error Handling

- Invalid frame shape should raise `ValueError` in the stitcher unit layer; this is a programming/provider contract error.
- Stream integration should not hide Audio2Face provider errors.
- Stale/cancel behavior remains owned by existing `TurnHandle` checks. If a face task is cancelled before emit, Task 8 stale-drop metrics still apply.
- Disabled stitching must never alter frames or metrics except to report disabled state.

## Testing Strategy

Unit tests:

- First segment is unchanged and not crossfaded.
- Second same-turn consecutive segment is crossfaded.
- `boundary_delta_after < boundary_delta_before`.
- `turn_id` change resets.
- `generation_epoch` change resets.
- `session_id` change resets.
- Non-consecutive `segment_index` resets.
- Short frames and overlap larger than frame count do not crash.
- Disabled stitching returns unchanged frames.

Stream tests:

- `server.face.frames` and `server.ue5.frames` timing includes stitch metrics.
- `timeline.json["stream"]["segments"]` includes stitch metrics.
- Existing cancel/stale tests continue to show `old_turn_face_leak_count = 0`.

Benchmark/client tests:

- Stream client carries stitch timing into `summary["segments"]`.
- Benchmark extracts first segment stitch metrics.

Real smoke:

- Existing `BIONIC_HEAD_RUN_REAL_EMOTALK=1` stream smoke remains default skipped.
- Real smoke should only assert metrics exist and old-turn leak remains 0; it should not require a specific improvement threshold because real content may not produce a large boundary jump.

## Acceptance

Minimum acceptance:

```text
pytest all passes
boundary_delta_after < boundary_delta_before in deterministic unit test
old_turn_face_leak_count = 0 in stream cancel/stale tests
stream benchmark still reports Task 8 timing
```

Expected real interpretation:

```text
Face latency should stay roughly in the Task 8 range.
UE5 first frame after TTS should not regress materially from Task 8.
Same-turn segment boundaries should be smoother when adjacent segments differ.
```
