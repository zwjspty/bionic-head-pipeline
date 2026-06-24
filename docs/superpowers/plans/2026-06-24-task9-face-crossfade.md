# Task 9 Face Crossfade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add turn-aware blendshape output crossfade so adjacent stream face segments have lower boundary jumps without changing audio, sidecar protocol, frame count, or UE5 chunking.

**Architecture:** Add a pure `FaceSegmentStitcher` that owns only face-frame state and metrics. Configure it through `AppSettings.face_stitching`, instantiate one stitcher per stream turn, and apply it after Audio2Face returns but before face/UE5 events. Existing stream client and benchmark already merge numeric timing keys, so Task 9 extends those summaries with stitch metrics.

**Tech Stack:** Python 3.11, Pydantic v2, FastAPI/Uvicorn stream path, pytest, pure Python list math for 52-channel blendshape frames.

## Global Constraints

- Work on branch `task9-face-crossfade`.
- Do not change EmoTalk input audio.
- Do not change sidecar protocol.
- Do not change ASR, LLM, TTS, provider selection, or UE5 event type names.
- Do not add GPU, student FaceDriver, audio overlap, or channel-group-specific smoothing.
- Preserve `frame_count`, per-frame `time_seconds`, and UE5 chunking.
- Stitch only same `session_id`, same `turn_id`, same `generation_epoch`, and consecutive `segment_index`.
- Reset on `session_id`, `turn_id`, `generation_epoch`, or non-consecutive `segment_index`.
- Automated tests must not require real EmoTalk, Conda, GPU, Ollama, Piper, or network access.
- Existing Task 8 metrics must remain available.

---

## File Structure

```text
src/bionic_head/face_stitcher.py
  New pure module with FaceSegmentStitcher, FaceStitchKey, FaceStitchMetrics.

src/bionic_head/config.py
  Adds FaceStitchingSettings and AppSettings.face_stitching.

config/mock.json
config/real.example.json
config/emotalk.example.json
config/local.json
  Add face_stitching JSON block so strict Pydantic config remains explicit.

src/bionic_head/orchestrators/stream.py
  Creates per-run stitcher and applies it before server.face.frames / UE5 formatting.

scripts/benchmark.py
  Promotes first segment stitch metrics from stream_client summary into latency report.

tests/unit/test_face_stitcher.py
tests/unit/test_config.py
tests/unit/test_stream_orchestrator.py
tests/unit/test_benchmark.py
tests/integration/test_stream_emotalk_sidecar_real.py
  Regression and smoke coverage.
```

---

### Task 1: Pure FaceSegmentStitcher

**Files:**
- Create: `src/bionic_head/face_stitcher.py`
- Create: `tests/unit/test_face_stitcher.py`

**Interfaces:**
- Produces:
  - `FaceStitchKey(session_id: str, turn_id: str, generation_epoch: int)`
  - `FaceStitchMetrics(enabled: bool, applied: bool, reset: bool, overlap_frames: int, actual_overlap_frames: int, boundary_delta_before: float | None, boundary_delta_after: float | None)`
  - `FaceSegmentStitcher(enabled: bool = True, overlap_frames: int = 8)`
  - `FaceSegmentStitcher.stitch(frames: list[list[float]], *, session_id: str, turn_id: str, generation_epoch: int, segment_index: int) -> tuple[list[list[float]], FaceStitchMetrics]`
  - `FaceSegmentStitcher.reset() -> None`
- Consumes: no app config, no provider types, no FastAPI.

- [ ] **Step 1: Write failing tests for first segment, second segment, reset, short frames, and disabled mode**

Create `tests/unit/test_face_stitcher.py`:

```python
from __future__ import annotations

import pytest

from bionic_head.face_stitcher import FaceSegmentStitcher


def _frames(values: list[float]) -> list[list[float]]:
    return [[value] * 52 for value in values]


def test_first_segment_is_unchanged_and_not_applied() -> None:
    stitcher = FaceSegmentStitcher(enabled=True, overlap_frames=3)
    frames = _frames([0.1, 0.2, 0.3])

    stitched, metrics = stitcher.stitch(
        frames,
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
    )

    assert stitched == frames
    assert stitched is not frames
    assert metrics.enabled is True
    assert metrics.applied is False
    assert metrics.reset is True
    assert metrics.actual_overlap_frames == 0
    assert metrics.boundary_delta_before is None
    assert metrics.boundary_delta_after is None


def test_second_consecutive_segment_crossfades_head_and_reduces_boundary_delta() -> None:
    stitcher = FaceSegmentStitcher(enabled=True, overlap_frames=3)
    stitcher.stitch(_frames([0.0, 0.0, 0.0]), session_id="s1", turn_id="t1", generation_epoch=0, segment_index=1)

    stitched, metrics = stitcher.stitch(
        _frames([1.0, 1.0, 1.0, 1.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=2,
    )

    assert metrics.applied is True
    assert metrics.reset is False
    assert metrics.actual_overlap_frames == 3
    assert metrics.boundary_delta_before == pytest.approx(1.0)
    assert metrics.boundary_delta_after == pytest.approx(1.0 / 3.0)
    assert stitched[0] == pytest.approx([1.0 / 3.0] * 52)
    assert stitched[1] == pytest.approx([2.0 / 3.0] * 52)
    assert stitched[2] == pytest.approx([1.0] * 52)
    assert stitched[3] == pytest.approx([1.0] * 52)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"session_id": "s2", "turn_id": "t1", "generation_epoch": 0, "segment_index": 2},
        {"session_id": "s1", "turn_id": "t2", "generation_epoch": 0, "segment_index": 2},
        {"session_id": "s1", "turn_id": "t1", "generation_epoch": 1, "segment_index": 2},
        {"session_id": "s1", "turn_id": "t1", "generation_epoch": 0, "segment_index": 4},
    ],
)
def test_context_change_or_non_consecutive_segment_resets(kwargs: dict[str, object]) -> None:
    stitcher = FaceSegmentStitcher(enabled=True, overlap_frames=2)
    stitcher.stitch(_frames([0.0, 0.0]), session_id="s1", turn_id="t1", generation_epoch=0, segment_index=1)

    stitched, metrics = stitcher.stitch(_frames([1.0, 1.0]), **kwargs)

    assert stitched == _frames([1.0, 1.0])
    assert metrics.reset is True
    assert metrics.applied is False
    assert metrics.actual_overlap_frames == 0


def test_overlap_is_clamped_for_short_segments() -> None:
    stitcher = FaceSegmentStitcher(enabled=True, overlap_frames=8)
    stitcher.stitch(_frames([0.0]), session_id="s1", turn_id="t1", generation_epoch=0, segment_index=1)

    stitched, metrics = stitcher.stitch(
        _frames([1.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=2,
    )

    assert metrics.actual_overlap_frames == 1
    assert stitched == _frames([1.0])


def test_disabled_stitcher_returns_unchanged_frames_without_stateful_crossfade() -> None:
    stitcher = FaceSegmentStitcher(enabled=False, overlap_frames=8)
    first, first_metrics = stitcher.stitch(_frames([0.0]), session_id="s1", turn_id="t1", generation_epoch=0, segment_index=1)
    second, second_metrics = stitcher.stitch(_frames([1.0]), session_id="s1", turn_id="t1", generation_epoch=0, segment_index=2)

    assert first == _frames([0.0])
    assert second == _frames([1.0])
    assert first_metrics.enabled is False
    assert second_metrics.enabled is False
    assert first_metrics.applied is False
    assert second_metrics.applied is False
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_face_stitcher.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'bionic_head.face_stitcher'`.

- [ ] **Step 3: Implement minimal stitcher**

Create `src/bionic_head/face_stitcher.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import copy


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

    def to_timing_payload(self) -> dict[str, float | bool]:
        payload: dict[str, float | bool] = {
            "face_stitch_enabled": self.enabled,
            "face_stitch_applied": self.applied,
            "face_stitch_reset": self.reset,
            "face_stitch_overlap_frames": float(self.overlap_frames),
            "face_stitch_actual_overlap_frames": float(self.actual_overlap_frames),
        }
        if self.boundary_delta_before is not None:
            payload["face_boundary_delta_before"] = self.boundary_delta_before
        if self.boundary_delta_after is not None:
            payload["face_boundary_delta_after"] = self.boundary_delta_after
        return payload


class FaceSegmentStitcher:
    def __init__(self, *, enabled: bool = True, overlap_frames: int = 8) -> None:
        if overlap_frames < 0:
            raise ValueError("overlap_frames must be non-negative")
        self.enabled = enabled
        self.overlap_frames = overlap_frames
        self._previous_key: FaceStitchKey | None = None
        self._previous_segment_index: int | None = None
        self._previous_tail: list[list[float]] = []

    def reset(self) -> None:
        self._previous_key = None
        self._previous_segment_index = None
        self._previous_tail = []

    def stitch(
        self,
        frames: list[list[float]],
        *,
        session_id: str,
        turn_id: str,
        generation_epoch: int,
        segment_index: int,
    ) -> tuple[list[list[float]], FaceStitchMetrics]:
        copied = copy.deepcopy(frames)
        if not self.enabled:
            return copied, FaceStitchMetrics(
                enabled=False,
                applied=False,
                reset=False,
                overlap_frames=self.overlap_frames,
                actual_overlap_frames=0,
            )

        key = FaceStitchKey(session_id=session_id, turn_id=turn_id, generation_epoch=generation_epoch)
        reset = self._should_reset(key, segment_index)
        actual_overlap = 0 if reset else min(self.overlap_frames, len(self._previous_tail), len(copied))
        if actual_overlap <= 0:
            self._store_tail(key, segment_index, copied)
            return copied, FaceStitchMetrics(
                enabled=True,
                applied=False,
                reset=True if reset else False,
                overlap_frames=self.overlap_frames,
                actual_overlap_frames=0,
            )

        previous_overlap = self._previous_tail[-actual_overlap:]
        before = _mean_abs_delta(previous_overlap[-1], copied[0])
        for index in range(actual_overlap):
            alpha = (index + 1) / float(actual_overlap)
            copied[index] = [
                previous * (1.0 - alpha) + current * alpha
                for previous, current in zip(previous_overlap[index], copied[index])
            ]
        after = _mean_abs_delta(previous_overlap[-1], copied[0])
        self._store_tail(key, segment_index, copied)
        return copied, FaceStitchMetrics(
            enabled=True,
            applied=True,
            reset=False,
            overlap_frames=self.overlap_frames,
            actual_overlap_frames=actual_overlap,
            boundary_delta_before=before,
            boundary_delta_after=after,
        )

    def _should_reset(self, key: FaceStitchKey, segment_index: int) -> bool:
        return (
            self._previous_key != key
            or self._previous_segment_index is None
            or segment_index != self._previous_segment_index + 1
        )

    def _store_tail(self, key: FaceStitchKey, segment_index: int, frames: list[list[float]]) -> None:
        tail_length = min(self.overlap_frames, len(frames))
        self._previous_key = key
        self._previous_segment_index = segment_index
        self._previous_tail = copy.deepcopy(frames[-tail_length:]) if tail_length > 0 else []


def _mean_abs_delta(left: Iterable[float], right: Iterable[float]) -> float:
    deltas = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    return sum(deltas) / len(deltas) if deltas else 0.0
```

- [ ] **Step 4: Run stitcher tests to verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_face_stitcher.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/bionic_head/face_stitcher.py tests/unit/test_face_stitcher.py
git commit -m "feat: add face segment stitcher"
```

---

### Task 2: Face Stitching Config

**Files:**
- Modify: `src/bionic_head/config.py`
- Modify: `config/mock.json`
- Modify: `config/real.example.json`
- Modify: `config/emotalk.example.json`
- Modify: `config/local.json`
- Modify: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `FaceSegmentStitcher(enabled: bool, overlap_frames: int)`.
- Produces: `settings.face_stitching.enabled`, `settings.face_stitching.overlap_frames`, `settings.face_stitching.reset_on_new_turn`, `settings.face_stitching.record_boundary_metrics`.

- [ ] **Step 1: Write failing config tests**

Add to `tests/unit/test_config.py`:

```python
def test_default_face_stitching_settings(mock_settings) -> None:
    assert mock_settings.face_stitching.enabled is True
    assert mock_settings.face_stitching.overlap_frames == 8
    assert mock_settings.face_stitching.reset_on_new_turn is True
    assert mock_settings.face_stitching.record_boundary_metrics is True


def test_accepts_face_stitching_config(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        """
        {
          "face_stitching": {
            "enabled": false,
            "overlap_frames": 5,
            "reset_on_new_turn": true,
            "record_boundary_metrics": false
          }
        }
        """,
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.face_stitching.enabled is False
    assert settings.face_stitching.overlap_frames == 5
    assert settings.face_stitching.reset_on_new_turn is True
    assert settings.face_stitching.record_boundary_metrics is False
```

If `Path` or `load_settings` is already imported in `tests/unit/test_config.py`, reuse the existing import.

- [ ] **Step 2: Run config tests to verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_config.py -q
```

Expected: fail because `AppSettings` has no `face_stitching`.

- [ ] **Step 3: Implement settings model and config files**

In `src/bionic_head/config.py`, add:

```python
class FaceStitchingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    overlap_frames: int = Field(default=8, ge=0)
    reset_on_new_turn: bool = True
    record_boundary_metrics: bool = True
```

In `AppSettings`, add:

```python
face_stitching: FaceStitchingSettings = Field(default_factory=FaceStitchingSettings)
```

Add this top-level JSON block to `config/mock.json`, `config/real.example.json`, `config/emotalk.example.json`, and `config/local.json`:

```json
"face_stitching": {
  "enabled": true,
  "overlap_frames": 8,
  "reset_on_new_turn": true,
  "record_boundary_metrics": true
}
```

- [ ] **Step 4: Run config tests to verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_config.py -q
```

Expected: all config tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/bionic_head/config.py config/mock.json config/real.example.json config/emotalk.example.json config/local.json tests/unit/test_config.py
git commit -m "feat: configure face stitching"
```

---

### Task 3: Stream Integration and Metrics

**Files:**
- Modify: `src/bionic_head/orchestrators/stream.py`
- Modify: `tests/unit/test_stream_orchestrator.py`
- Modify: `tests/integration/test_stream_emotalk_sidecar_real.py`

**Interfaces:**
- Consumes: `FaceSegmentStitcher.stitch(...) -> tuple[list[list[float]], FaceStitchMetrics]`.
- Produces:
  - `server.face.frames.payload["timing"]` includes stitch metrics.
  - `server.ue5.frames.payload["timing"]` includes stitch metrics.
  - `timeline.json["stream"]["segments"][i]` includes stitch metrics.

- [ ] **Step 1: Write failing stream tests**

Add to `tests/unit/test_stream_orchestrator.py`:

```python
@pytest.mark.asyncio
async def test_stream_applies_face_stitching_to_second_segment_and_records_boundary_metrics(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.mock.reply = "第一句话很短。第二句话也短。"
    settings.stream.sentence_min_chars = 4
    settings.stream.sentence_max_chars = 8
    settings.face_stitching.enabled = True
    settings.face_stitching.overlap_frames = 2
    harness = stream_harness_factory(settings=settings, registry=build_registry(settings))

    await harness.run()

    timeline_path = (
        harness.store.runs
        / str(harness.turn.session_id)
        / str(harness.turn.turn_id)
        / "timeline.json"
    )
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    segments = timeline["stream"]["segments"]
    stitched_segments = [
        segment for segment in segments if segment.get("face_stitch_applied") is True
    ]

    assert stitched_segments
    assert stitched_segments[0]["face_stitch_actual_overlap_frames"] > 0
    assert stitched_segments[0]["face_boundary_delta_after"] <= stitched_segments[0]["face_boundary_delta_before"]
    assert timeline["stream"]["old_turn_face_leak_count"] == 0

    ue5_payloads = [
        envelope.payload
        for envelope in harness.json_envelopes
        if envelope.type.value == "server.ue5.frames"
    ]
    assert any(payload["timing"].get("face_stitch_applied") is True for payload in ue5_payloads)
```

Make sure `build_registry` is imported in the test file if not already available.

Update `tests/integration/test_stream_emotalk_sidecar_real.py` to assert:

```python
assert "face_stitch_enabled" in timing
assert "face_stitch_overlap_frames" in timing
```

- [ ] **Step 2: Run stream tests to verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_orchestrator.py::test_stream_applies_face_stitching_to_second_segment_and_records_boundary_metrics -q
```

Expected: fail because timing does not contain stitch metrics yet.

- [ ] **Step 3: Integrate stitcher in stream**

In `src/bionic_head/orchestrators/stream.py`:

1. Import:

```python
from bionic_head.face_stitcher import FaceSegmentStitcher, FaceStitchMetrics
```

2. Add fields to `_StreamSegmentTiming`:

```python
face_stitch_enabled: bool | None = None
face_stitch_applied: bool | None = None
face_stitch_reset: bool | None = None
face_stitch_overlap_frames: float | None = None
face_stitch_actual_overlap_frames: float | None = None
face_boundary_delta_before: float | None = None
face_boundary_delta_after: float | None = None
```

3. Add method:

```python
def apply_stitch_metrics(self, metrics: FaceStitchMetrics) -> None:
    payload = metrics.to_timing_payload()
    self.face_stitch_enabled = bool(payload["face_stitch_enabled"])
    self.face_stitch_applied = bool(payload["face_stitch_applied"])
    self.face_stitch_reset = bool(payload["face_stitch_reset"])
    self.face_stitch_overlap_frames = float(payload["face_stitch_overlap_frames"])
    self.face_stitch_actual_overlap_frames = float(payload["face_stitch_actual_overlap_frames"])
    self.face_boundary_delta_before = payload.get("face_boundary_delta_before")  # type: ignore[assignment]
    self.face_boundary_delta_after = payload.get("face_boundary_delta_after")  # type: ignore[assignment]
```

4. Update `timing_payload()` to include boolean and numeric stitch fields when not `None`.

5. Inside `run(...)`, instantiate:

```python
face_stitcher = FaceSegmentStitcher(
    enabled=self.settings.face_stitching.enabled,
    overlap_frames=self.settings.face_stitching.overlap_frames,
)
```

6. Pass `face_stitcher` into `_process_face_segment(...)`.

7. After `face = await self.registry.audio2face.drive(...)`, call:

```python
stitched_frames, stitch_metrics = face_stitcher.stitch(
    face.frames,
    session_id=str(turn.session_id),
    turn_id=str(turn.turn_id),
    generation_epoch=turn.generation_epoch,
    segment_index=chunk_index,
)
segment_timing.apply_stitch_metrics(stitch_metrics)
if stitched_frames != face.frames:
    face = face.model_copy(update={"frames": stitched_frames, "frame_count": len(stitched_frames)})
```

- [ ] **Step 4: Run stream tests to verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_orchestrator.py tests/integration/test_stream_emotalk_sidecar_real.py -q
```

Expected: stream unit tests pass; real smoke remains skipped by default.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add src/bionic_head/orchestrators/stream.py tests/unit/test_stream_orchestrator.py tests/integration/test_stream_emotalk_sidecar_real.py
git commit -m "feat: stitch stream face segments"
```

---

### Task 4: Benchmark Extraction and Final Verification

**Files:**
- Modify: `scripts/benchmark.py`
- Modify: `tests/unit/test_benchmark.py`
- Optionally modify: `tests/unit/test_stream_client.py`

**Interfaces:**
- Consumes: stream client segment summary containing stitch metrics.
- Produces benchmark metrics:
  - `face_stitch_overlap_frames`
  - `face_stitch_actual_overlap_frames`
  - `face_boundary_delta_before`
  - `face_boundary_delta_after`
  - `face_stitch_applied_count`
  - `face_stitch_reset_count`

- [ ] **Step 1: Write failing benchmark tests**

Add to `tests/unit/test_benchmark.py`:

```python
def test_stream_metrics_from_summary_extracts_face_stitch_metrics() -> None:
    metrics = stream_metrics_from_summary(
        {
            "segments": {
                "chunk-0001": {
                    "tts_audio_event_ms": 100.0,
                    "face_stitch_enabled": True,
                    "face_stitch_applied": True,
                    "face_stitch_reset": False,
                    "face_stitch_overlap_frames": 8.0,
                    "face_stitch_actual_overlap_frames": 5.0,
                    "face_boundary_delta_before": 0.4,
                    "face_boundary_delta_after": 0.1,
                }
            }
        },
        wall_ms=1000.0,
    )

    assert metrics["face_stitch_overlap_frames"] == 8.0
    assert metrics["face_stitch_actual_overlap_frames"] == 5.0
    assert metrics["face_boundary_delta_before"] == 0.4
    assert metrics["face_boundary_delta_after"] == 0.1
    assert metrics["face_stitch_applied_count"] == 1.0
    assert metrics["face_stitch_reset_count"] == 0.0
```

- [ ] **Step 2: Run benchmark test to verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_benchmark.py::test_stream_metrics_from_summary_extracts_face_stitch_metrics -q
```

Expected: fail with missing metric keys.

- [ ] **Step 3: Implement benchmark extraction**

In `scripts/benchmark.py`, inside `stream_metrics_from_summary(...)`, after first segment extraction, add:

```python
for key in (
    "face_stitch_overlap_frames",
    "face_stitch_actual_overlap_frames",
    "face_boundary_delta_before",
    "face_boundary_delta_after",
):
    value = _float_or_none(first_segment.get(key))
    if value is not None:
        metrics[key] = value

applied = first_segment.get("face_stitch_applied")
if isinstance(applied, bool):
    metrics["face_stitch_applied_count"] = 1.0 if applied else 0.0
reset = first_segment.get("face_stitch_reset")
if isinstance(reset, bool):
    metrics["face_stitch_reset_count"] = 1.0 if reset else 0.0
```

- [ ] **Step 4: Run focused tests to verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_benchmark.py tests/unit/test_stream_client.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run full verification**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: all ordinary tests pass; real provider smoke tests remain skipped unless env vars are set.

- [ ] **Step 6: Run one mock stream benchmark to confirm report fields**

Start server in one terminal:

```bash
PYTHONPATH=src BIONIC_CONFIG=config/mock.json .venv/bin/uvicorn bionic_head.api.app:create_app --factory --host 127.0.0.1 --port 8039
```

Run benchmark:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark.py \
  --mode stream \
  --ws-url ws://127.0.0.1:8039/pipeline/stream \
  --wav /tmp/bionic-task9-input.wav \
  --runs 1 \
  --output /tmp/bionic-task9-stream-report.json
```

Inspect report:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("/tmp/bionic-task9-stream-report.json").read_text(encoding="utf-8"))
keys = [
    "face_stitch_overlap_frames",
    "face_stitch_actual_overlap_frames",
    "face_boundary_delta_before",
    "face_boundary_delta_after",
    "face_stitch_applied_count",
    "face_stitch_reset_count",
]
print(json.dumps({key: report["metrics"].get(key) for key in keys}, ensure_ascii=False, indent=2))
PY
```

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add scripts/benchmark.py tests/unit/test_benchmark.py tests/unit/test_stream_client.py
git commit -m "feat: report face stitching benchmark metrics"
```

---

## Final Review Checklist

- [ ] `git status --short` only shows intended files before each commit.
- [ ] `FaceSegmentStitcher` has no provider, FastAPI, sidecar, or config dependency.
- [ ] First segment does not alter frames.
- [ ] Same-turn second segment can reduce deterministic boundary delta.
- [ ] `session_id`, `turn_id`, `generation_epoch`, and non-consecutive `segment_index` reset stitcher state.
- [ ] Frame count remains unchanged.
- [ ] Existing stale/cancel tests still report `old_turn_face_leak_count = 0`.
- [ ] Full pytest passes.
- [ ] No real provider tests run unless explicitly enabled.

## Execution Recommendation

Use Subagent-Driven execution if several agents are available:

1. Task 1 stitcher pure logic.
2. Task 2 config.
3. Task 3 stream integration.
4. Task 4 benchmark/report verification.

Inline execution is also safe because Task 9 is mostly contained and tests are fast.
