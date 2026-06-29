# Task 18 Client AV Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add client-side audio/face synchronization to local and interactive demo clients with `immediate_audio` and `wait_for_face` strategies plus measurable client-side AV offset metrics.

**Architecture:** Add focused `PlaybackClock` and `SegmentSyncCoordinator` modules under `src/bionic_head/client/`. Keep playback engines in the scripts for now, but route TTS and UE5 events through the coordinator before enqueueing. The backend protocol stays unchanged.

**Tech Stack:** Python 3.11-compatible code, asyncio clients, pytest, existing WebSocket protocol, no new required runtime dependencies.

## Global Constraints

- Do not change backend protocol, ASR, TTS, LLM, EmoTalk, UE5 adapter, or sidecar.
- Default tests must not require sound card, microphone, GPU, Ollama, Piper, or EmoTalk.
- Default playback strategy is `immediate_audio`.
- `wait_for_face` must have a timeout fallback; default `wait_for_face_timeout_ms = 800`.
- `playback.stop` must clear audio, face, and pending sync state.
- Old generation audio/face must not play.

---

### Task 1: PlaybackClock metrics

**Files:**
- Create: `src/bionic_head/client/playback_clock.py`
- Test: `tests/unit/test_playback_clock.py`

**Interfaces:**
- Produces: `PlaybackClock(clock: Callable[[], float])`
- Produces: `PlaybackClock.mark_tts_received(segment_id: str) -> float`
- Produces: `PlaybackClock.mark_ue5_received(segment_id: str) -> float`
- Produces: `PlaybackClock.mark_audio_play_start(segment_id: str) -> float`
- Produces: `PlaybackClock.mark_face_display(segment_id: str) -> float`
- Produces: `PlaybackClock.mark_playback_stop_received() -> float`
- Produces: `PlaybackClock.mark_audio_stopped() -> float`
- Produces: `PlaybackClock.mark_face_cleared() -> float`
- Produces: `PlaybackClock.metrics() -> dict[str, object]`
- Produces: `PlaybackClock.segment_metrics() -> dict[str, dict[str, object]]`

- [x] **Step 1: Write failing tests**

```python
def test_clock_records_audio_face_offset():
    values = iter([0.0, 0.6, 1.0, 1.2])
    clock = PlaybackClock(clock=lambda: next(values))
    clock.mark_tts_received("chunk-0001")
    clock.mark_audio_play_start("chunk-0001")
    clock.mark_ue5_received("chunk-0001")
    clock.mark_face_display("chunk-0001")
    metrics = clock.metrics()
    assert metrics["client_audio_play_start_ms"] == 600.0
    assert metrics["client_face_first_frame_displayed_ms"] == 1200.0
    assert metrics["client_audio_face_offset_ms"] == 600.0
    assert metrics["client_face_late_by_ms"] == 600.0
```

```python
def test_clock_records_wait_for_face_and_stop_offsets():
    values = iter([0.0, 0.4, 0.9, 0.91, 1.0, 1.05, 1.06])
    clock = PlaybackClock(clock=lambda: next(values))
    clock.mark_tts_received("chunk-0001")
    clock.mark_ue5_received("chunk-0001")
    clock.mark_audio_play_start("chunk-0001")
    clock.mark_face_display("chunk-0001")
    clock.mark_playback_stop_received()
    clock.mark_audio_stopped()
    clock.mark_face_cleared()
    metrics = clock.metrics()
    assert metrics["client_audio_wait_for_face_ms"] == 510.0
    assert metrics["client_audio_face_offset_ms"] == 10.0
    assert metrics["client_playback_stop_to_audio_stop_ms"] == 50.0
    assert metrics["client_playback_stop_to_face_clear_ms"] == 60.0
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_playback_clock.py -q
```

Expected: import failure or missing `PlaybackClock`.

- [x] **Step 3: Implement minimal PlaybackClock**

Create dataclasses for per-segment timestamps, compute first top-level metrics,
and preserve all per-segment metrics under `playback_segments`.

- [x] **Step 4: Run tests and verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_playback_clock.py -q
```

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add src/bionic_head/client/playback_clock.py tests/unit/test_playback_clock.py
git commit -m "feat: add client playback clock metrics"
```

---

### Task 2: SegmentSyncCoordinator

**Files:**
- Create: `src/bionic_head/client/segment_sync.py`
- Test: `tests/unit/test_segment_sync.py`

**Interfaces:**
- Consumes: `PlaybackClock`
- Produces: `PlaybackSyncStrategy = Literal["immediate_audio", "wait_for_face"]`
- Produces: `SegmentSyncCoordinator(strategy, clock, wait_for_face_timeout_ms=800)`
- Produces: `accept_tts(segment_id, chunk_id, wav_bytes, generation_epoch, received_ms=None) -> list[PlaybackAction]`
- Produces: `accept_face(segment_id, chunk_id, payload, generation_epoch, received_ms=None) -> list[PlaybackAction]`
- Produces: `flush_timeouts(now_ms: float | None = None) -> list[PlaybackAction]`
- Produces: `clear(reason: str) -> None`

- [x] **Step 1: Write failing tests**

Tests must cover:

- `immediate_audio` returns a `play_audio` action when TTS arrives.
- `wait_for_face` stores TTS until matching face arrives.
- `wait_for_face` handles face-before-TTS.
- `wait_for_face` timeout returns `play_audio` and records timeout.
- stale generation events are dropped.
- `clear()` removes pending segments.

- [x] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_segment_sync.py -q
```

- [x] **Step 3: Implement coordinator**

Use a small action dataclass:

```python
@dataclass(frozen=True)
class PlaybackAction:
    kind: Literal["play_audio", "display_face"]
    segment_id: str
    chunk_id: str
    generation_epoch: int | None
    wav_bytes: bytes | None = None
    face_payload: dict[str, object] | None = None
```

- [x] **Step 4: Run tests and verify GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_segment_sync.py -q
```

- [x] **Step 5: Commit**

```bash
git add src/bionic_head/client/segment_sync.py tests/unit/test_segment_sync.py
git commit -m "feat: add client segment sync coordinator"
```

---

### Task 3: Local demo client integration

**Files:**
- Modify: `scripts/local_demo_client.py`
- Test: `tests/unit/test_local_demo_client.py`

**Interfaces:**
- Consumes: `PlaybackClock`
- Consumes: `SegmentSyncCoordinator`
- Adds CLI: `--playback-sync {immediate_audio,wait_for_face}`
- Adds CLI: `--wait-for-face-timeout-ms 800`

- [x] **Step 1: Write failing tests**

Add tests for:

- parser default strategy is `immediate_audio`.
- local receiver immediate strategy enqueues audio on TTS binary.
- local receiver wait strategy does not enqueue audio until matching UE5 frames.
- summary includes `playback_sync_strategy`, `client_audio_wait_for_face_ms`, and `client_audio_face_offset_ms`.
- playback stop clears pending sync.

- [x] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

- [x] **Step 3: Wire coordinator into LocalDemoReceiver**

Replace direct `audio.enqueue_wav()` and `face.enqueue_frames()` calls with
coordinator actions. Preserve file writing for `tts/` and `ue5/`.

- [x] **Step 4: Run tests and verify GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

- [x] **Step 5: Commit**

```bash
git add scripts/local_demo_client.py tests/unit/test_local_demo_client.py
git commit -m "feat: sync local demo audio and face playback"
```

---

### Task 4: Interactive and scripted client integration

**Files:**
- Modify: `scripts/interactive_demo_client.py`
- Modify: `src/bionic_head/client/scripted.py`
- Test: `tests/unit/test_interactive_demo_client.py`
- Test: `tests/unit/test_scripted_interactive_client.py`

**Interfaces:**
- Adds CLI: `--playback-sync {immediate_audio,wait_for_face}`
- Adds CLI: `--wait-for-face-timeout-ms 800`
- Interaction report includes AV sync metrics.

- [x] **Step 1: Write failing tests**

Add tests for:

- parser accepts both strategies.
- `run_interactive_demo()` passes strategy into `LocalDemoReceiver`.
- scripted mode can run with `immediate_audio`.
- scripted mode can run with `wait_for_face`.
- interaction report includes `client_audio_face_offset_ms`.

- [x] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_interactive_demo_client.py tests/unit/test_scripted_interactive_client.py -q
```

- [x] **Step 3: Wire strategy through interactive/scripted paths**

Pass `playback_sync` and `wait_for_face_timeout_ms` from CLI to
`LocalDemoReceiver`.

- [x] **Step 4: Run tests and verify GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_interactive_demo_client.py tests/unit/test_scripted_interactive_client.py -q
```

- [x] **Step 5: Commit**

```bash
git add scripts/interactive_demo_client.py src/bionic_head/client/scripted.py tests/unit/test_interactive_demo_client.py tests/unit/test_scripted_interactive_client.py
git commit -m "feat: sync interactive demo audio and face playback"
```

---

### Task 5: Documentation and final verification

**Files:**
- Create: `docs/operations/client-audio-face-sync.md`

- [x] **Step 1: Document usage**

Include:

```bash
.venv/bin/python scripts/interactive_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --output-dir /tmp/bionic-task18-immediate \
  --mode scripted \
  --scripted-turns 1 \
  --mic-backend fake \
  --audio-backend null \
  --chunk-ms 40 \
  --no-play-audio \
  --playback-sync immediate_audio
```

```bash
.venv/bin/python scripts/interactive_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --output-dir /tmp/bionic-task18-wait-face \
  --mode scripted \
  --scripted-turns 1 \
  --mic-backend fake \
  --audio-backend null \
  --chunk-ms 40 \
  --no-play-audio \
  --playback-sync wait_for_face
```

- [x] **Step 2: Run full tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Result: `407 passed, 9 skipped, 1 warning`.

- [x] **Step 3: Commit**

```bash
git add docs/operations/client-audio-face-sync.md
git commit -m "docs: document client audio face sync"
```

---

## Completion checklist

- [x] Full pytest passes.
- [x] local demo supports both strategies.
- [x] interactive demo supports both strategies.
- [x] wait_for_face has timeout fallback.
- [x] summary/report includes `client_audio_face_offset_ms`.
- [x] playback.stop clears pending sync, audio, and face.
- [x] old generation audio/face do not play.
