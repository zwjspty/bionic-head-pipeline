# Task 13 Client Playback Interrupt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local demo client prove that playback-side interrupts stop local audio, clear local face buffers, drop stale generations, and record client-side stop latency metrics.

**Architecture:** Keep the WebSocket protocol unchanged. Anchor `--cancel-after-ms` to the first local audio playback start rather than to the end of upload. Extend `PlaybackMetrics`, `AudioPlaybackEngine`, and `LocalDemoReceiver` in `scripts/local_demo_client.py`, then document a real smoke command.

**Tech Stack:** Python 3.10+, asyncio, websockets, pytest, existing `scripts/local_demo_client.py` and `tests/unit/test_local_demo_client.py`.

## Global Constraints

- Do not add microphone input.
- Do not add acoustic echo cancellation.
- Do not add WebRTC.
- Do not add browser UI.
- Do not add real UE5 runtime integration.
- Do not add real-time Blender playback.
- Default tests must not require sound hardware, GPU, Ollama, Piper, EmoTalk, or a running FastAPI server.
- Preserve existing `summary.json` and `client_playback_metrics.json` fields for backward compatibility.
- Use TDD: write each failing test, verify it fails, implement the minimum code, verify it passes.

---

## File Structure

- Modify `scripts/local_demo_client.py`
  - `PlaybackMetrics`: add local interrupt timestamps and delta calculation.
  - `AudioPlaybackEngine`: add first-play callback.
  - `LocalDemoReceiver`: ensure stop/cancel updates new metrics and clears buffers.
  - `run_local_demo`: schedule `client.turn.cancel` from first local audio playback start.
- Modify `tests/unit/test_local_demo_client.py`
  - Add unit tests for metrics, playback-start anchored cancellation, stale drops, and summary contract.
- Modify `docs/operations/local-demo-client.md`
  - Add playback interrupt smoke command and expected summary fields.

---

### Task 13.1: Add interrupt metrics to `PlaybackMetrics`

**Files:**
- Modify: `scripts/local_demo_client.py`
- Modify: `tests/unit/test_local_demo_client.py`

**Interfaces:**
- Consumes: `PlaybackMetrics(clock: Callable[[], float])`
- Produces:
  - `PlaybackMetrics.mark_client_interrupt_sent() -> None`
  - `PlaybackMetrics.to_dict()["client_interrupt_sent_ms"]`
  - `PlaybackMetrics.to_dict()["server_playback_stop_received_ms"]`
  - `PlaybackMetrics.to_dict()["client_interrupt_to_playback_stop_ms"]`
  - `PlaybackMetrics.to_dict()["client_interrupt_to_audio_stop_ms"]`
  - `PlaybackMetrics.to_dict()["client_interrupt_to_face_clear_ms"]`

- [ ] **Step 1: Write the failing metrics test**

Append this test near the other `PlaybackMetrics` tests in `tests/unit/test_local_demo_client.py`:

```python
def test_playback_metrics_record_interrupt_deltas(fake_clock: FakeClock) -> None:
    metrics = PlaybackMetrics(clock=fake_clock)

    fake_clock.advance(0.100)
    metrics.mark_client_interrupt_sent()
    fake_clock.advance(0.025)
    metrics.mark_playback_stop_received()
    fake_clock.advance(0.010)
    metrics.mark_audio_stopped()
    fake_clock.advance(0.005)
    metrics.mark_face_buffer_cleared()

    payload = metrics.to_dict()
    assert payload["client_interrupt_sent_ms"] == 100.0
    assert payload["server_playback_stop_received_ms"] == 125.0
    assert payload["client_playback_stop_received_ms"] == 125.0
    assert payload["client_interrupt_to_playback_stop_ms"] == 25.0
    assert payload["client_interrupt_to_audio_stop_ms"] == 35.0
    assert payload["client_interrupt_to_face_clear_ms"] == 40.0
```

- [ ] **Step 2: Run the failing metrics test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py::test_playback_metrics_record_interrupt_deltas -q
```

Expected: FAIL because `mark_client_interrupt_sent` and the new metric keys do not exist.

- [ ] **Step 3: Implement the minimal metrics changes**

In `PlaybackMetrics.__init__`, add:

```python
self.client_interrupt_sent_ms: float | None = None
self.server_playback_stop_received_ms: float | None = None
self.client_interrupt_to_playback_stop_ms: float | None = None
self.client_interrupt_to_face_clear_ms: float | None = None
```

Add this method:

```python
def mark_client_interrupt_sent(self) -> None:
    if self.client_interrupt_sent_ms is None:
        self.client_interrupt_sent_ms = self._elapsed_ms()
        self._update_interrupt_to_playback_stop()
        self._update_interrupt_to_audio_stop()
        self._update_interrupt_to_face_clear()
```

Update `mark_playback_stop_received`:

```python
def mark_playback_stop_received(self) -> None:
    if self.client_playback_stop_received_ms is None:
        stopped_at = self._elapsed_ms()
        self.client_playback_stop_received_ms = stopped_at
        self.server_playback_stop_received_ms = stopped_at
        self._update_interrupt_to_playback_stop()
        self._update_interrupt_to_audio_stop()
```

Update `mark_audio_stopped`:

```python
def mark_audio_stopped(self) -> None:
    if self.client_audio_stopped_ms is None:
        self.client_audio_stopped_ms = self._elapsed_ms()
        self._update_interrupt_to_audio_stop()
```

Update `mark_face_buffer_cleared`:

```python
def mark_face_buffer_cleared(self) -> None:
    if self.client_face_buffer_cleared_ms is None:
        self.client_face_buffer_cleared_ms = self._elapsed_ms()
        self._update_interrupt_to_face_clear()
```

Add these helpers:

```python
def _update_interrupt_to_playback_stop(self) -> None:
    if (
        self.client_interrupt_to_playback_stop_ms is None
        and self.client_interrupt_sent_ms is not None
        and self.client_playback_stop_received_ms is not None
    ):
        self.client_interrupt_to_playback_stop_ms = round(
            self.client_playback_stop_received_ms - self.client_interrupt_sent_ms,
            3,
        )

def _update_interrupt_to_face_clear(self) -> None:
    if (
        self.client_interrupt_to_face_clear_ms is None
        and self.client_interrupt_sent_ms is not None
        and self.client_face_buffer_cleared_ms is not None
    ):
        self.client_interrupt_to_face_clear_ms = round(
            self.client_face_buffer_cleared_ms - self.client_interrupt_sent_ms,
            3,
        )
```

Update `_update_interrupt_to_audio_stop` to subtract `client_interrupt_sent_ms` when present:

```python
def _update_interrupt_to_audio_stop(self) -> None:
    if (
        self.client_interrupt_to_audio_stop_ms is None
        and self.client_interrupt_sent_ms is not None
        and self.client_audio_stopped_ms is not None
    ):
        self.client_interrupt_to_audio_stop_ms = round(
            self.client_audio_stopped_ms - self.client_interrupt_sent_ms,
            3,
        )
```

Add the new keys to `to_dict`.

- [ ] **Step 4: Verify the metrics test passes**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py::test_playback_metrics_record_interrupt_deltas -q
```

Expected: PASS.

- [ ] **Step 5: Run local demo client unit tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/local_demo_client.py tests/unit/test_local_demo_client.py
git commit -m "feat: record local interrupt playback metrics"
```

---

### Task 13.2: Anchor `--cancel-after-ms` to first local audio playback

**Files:**
- Modify: `scripts/local_demo_client.py`
- Modify: `tests/unit/test_local_demo_client.py`

**Interfaces:**
- Consumes:
  - `AudioPlaybackEngine(metrics, sink=MemoryAudioSink())`
  - `run_local_demo(url, wav_path, output_dir, chunk_ms, play_audio, cancel_after_ms)`
- Produces:
  - `AudioPlaybackEngine(metrics, sink=MemoryAudioSink(), on_first_play=callback)`
  - `run_local_demo` sends `client.turn.cancel` only after first local playback starts.

- [ ] **Step 1: Write the failing playback-start cancel test**

Add this async test near `test_run_local_demo_sends_turn_cancel_after_delay`:

```python
@pytest.mark.asyncio
async def test_run_local_demo_sends_cancel_after_first_audio_play(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    server_envelope,
) -> None:
    ready = server_envelope("server.session.ready", payload={})
    ready["turn_id"] = None
    ready["payload"]["turn_id"] = None
    wav_payload = b"RIFF----WAVE"
    tts = server_envelope(
        "server.tts.audio",
        payload={
            "chunk_id": "chunk-1",
            "segment_id": "segment-1",
            "format": "wav",
            "byte_length": len(wav_payload),
            "generation_epoch": 0,
        },
        generation_epoch=0,
    )
    stop = server_envelope("server.playback.stop", payload={}, generation_epoch=1)
    cancelled = server_envelope("server.turn.cancelled", payload={}, generation_epoch=1)
    websocket = FakeWebSocket([json.dumps(ready), json.dumps(tts), wav_payload, json.dumps(stop), json.dumps(cancelled)])

    monkeypatch.setattr(local_demo_client, "read_pcm16_from_wav", lambda _: b"\x01\x02" * 320)
    ids = iter([SESSION_ID, TURN_ID])
    monkeypatch.setattr(local_demo_client, "uuid4", lambda: next(ids))
    monkeypatch.setattr(local_demo_client, "pcm_chunks", lambda pcm, *, chunk_ms: [pcm])
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=lambda url: FakeConnect(websocket)))

    terminal = await run_local_demo(
        "ws://127.0.0.1:8005/pipeline/stream",
        tmp_path / "input.wav",
        tmp_path / "out",
        20,
        play_audio=False,
        cancel_after_ms=0,
    )

    assert terminal == "server.turn.cancelled"
    sent_events = [json.loads(message) for message in websocket.sent if isinstance(message, str)]
    assert [event["type"] for event in sent_events] == [
        "client.session.start",
        "client.audio.start",
        "client.audio.chunk",
        "client.audio.end",
        "client.turn.cancel",
    ]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["client_interrupt_sent_ms"] is not None
    assert summary["playback_stop_count"] == 1
```

- [ ] **Step 2: Run the failing playback-start cancel test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py::test_run_local_demo_sends_cancel_after_first_audio_play -q
```

Expected: FAIL because the current cancel scheduling is not tied to `AudioPlaybackEngine` first playback.

- [ ] **Step 3: Add an `on_first_play` callback to `AudioPlaybackEngine`**

Change the constructor:

```python
def __init__(
    self,
    metrics: PlaybackMetrics,
    sink: AudioSink | None = None,
    *,
    on_first_play: Callable[[], None] | None = None,
) -> None:
    self._metrics = metrics
    self._sink = sink or MemoryAudioSink()
    self._on_first_play = on_first_play
    self._first_play_callback_fired = False
    self._queued_chunks: dict[str, bytes] = {}
    self._pending_playback: deque[tuple[str, bytes, int | None]] = deque()
    self._draining = False
```

In `_drain_pending_playback`, after `mark_audio_play_started()` and before `self._sink.play(wav_bytes)`, add:

```python
if not self._first_play_callback_fired and self._on_first_play is not None:
    self._first_play_callback_fired = True
    self._on_first_play()
```

- [ ] **Step 4: Change `run_local_demo` cancel scheduling**

Remove the block that schedules cancellation immediately after sending `client.audio.end`. Replace it with a first-play callback:

```python
sequence_lock = asyncio.Lock()
cancel_task: asyncio.Task[None] | None = None
cancel_sent = False

async def send_cancel_after_playback_delay() -> None:
    nonlocal sequence, cancel_sent
    if cancel_after_ms is not None and cancel_after_ms > 0:
        await asyncio.sleep(cancel_after_ms / 1000.0)
    async with sequence_lock:
        if cancel_sent:
            return
        cancel_sent = True
        current_sequence = sequence
        sequence += 1
    metrics.mark_client_interrupt_sent()
    await websocket.send(
        json.dumps(
            client_event(
                "client.turn.cancel",
                session_id=session_id,
                turn_id=turn_id,
                sequence=current_sequence,
                payload={"reason": "client_playback_interrupt"},
            )
        )
    )

def schedule_cancel_on_first_play() -> None:
    nonlocal cancel_task
    if cancel_after_ms is None or cancel_task is not None:
        return
    cancel_task = asyncio.create_task(send_cancel_after_playback_delay())
```

Create `AudioPlaybackEngine` with:

```python
AudioPlaybackEngine(metrics, sink=audio_sink, on_first_play=schedule_cancel_on_first_play)
```

Keep the existing cleanup that cancels `cancel_task` if it is still pending at the end.

- [ ] **Step 5: Verify the playback-start cancel test passes**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py::test_run_local_demo_sends_cancel_after_first_audio_play -q
```

Expected: PASS.

- [ ] **Step 6: Verify the old immediate-cancel test remains meaningful**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py::test_run_local_demo_sends_turn_cancel_after_delay -q
```

Expected: update the old test if needed so it describes playback-start cancellation. It must pass and must not assert cancellation before any `server.tts.audio` binary has been played.

- [ ] **Step 7: Run local demo client unit tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/local_demo_client.py tests/unit/test_local_demo_client.py
git commit -m "feat: send demo cancel after playback starts"
```

---

### Task 13.3: Strengthen playback stop summary contract

**Files:**
- Modify: `tests/unit/test_local_demo_client.py`
- Modify: `scripts/local_demo_client.py`

**Interfaces:**
- Consumes: `LocalDemoReceiver.finish()`
- Produces: `summary.json` and `client_playback_metrics.json` containing all Task 13 metrics.

- [ ] **Step 1: Update the metrics contract test**

In `test_receiver_finish_writes_metrics_contract_files`, extend `expected_metric_keys`:

```python
expected_metric_keys = {
    "client_tts_received_ms",
    "client_audio_enqueued_count",
    "client_audio_play_start_ms",
    "client_audio_stopped_ms",
    "client_ue5_first_frame_received_ms",
    "client_face_buffered_chunk_count",
    "client_face_first_frame_displayed_ms",
    "client_audio_face_offset_ms",
    "client_interrupt_sent_ms",
    "server_playback_stop_received_ms",
    "client_playback_stop_received_ms",
    "client_interrupt_to_playback_stop_ms",
    "client_interrupt_to_audio_stop_ms",
    "client_interrupt_to_face_clear_ms",
    "client_face_buffer_cleared_ms",
    "client_stale_audio_drop_count",
    "client_stale_face_drop_count",
}
```

Before the existing `server.playback.stop`, add:

```python
fake_clock.advance(0.005)
metrics.mark_client_interrupt_sent()
```

Then assert:

```python
assert metrics_payload["client_interrupt_sent_ms"] == 15.0
assert metrics_payload["server_playback_stop_received_ms"] == 30.0
assert metrics_payload["client_interrupt_to_playback_stop_ms"] == 15.0
assert metrics_payload["client_interrupt_to_audio_stop_ms"] == 15.0
assert metrics_payload["client_interrupt_to_face_clear_ms"] == 15.0
```

- [ ] **Step 2: Run the contract test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py::test_receiver_finish_writes_metrics_contract_files -q
```

Expected: FAIL until Task 13.1 metrics are included everywhere.

- [ ] **Step 3: Implement any missing summary keys**

If Task 13.1 already added all keys to `PlaybackMetrics.to_dict`, no extra production code is needed. If the test exposes missing keys, add them to `to_dict`.

- [ ] **Step 4: Verify the contract test passes**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py::test_receiver_finish_writes_metrics_contract_files -q
```

Expected: PASS.

- [ ] **Step 5: Run local demo client unit tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/local_demo_client.py tests/unit/test_local_demo_client.py
git commit -m "test: assert local interrupt metrics contract"
```

---

### Task 13.4: Document playback interrupt smoke

**Files:**
- Modify: `docs/operations/local-demo-client.md`

**Interfaces:**
- Consumes: `scripts/local_demo_client.py --cancel-after-ms`
- Produces: documented smoke command and expected metrics.

- [ ] **Step 1: Add the docs section**

Append this section after the no-audio verification notes:

````markdown
## Playback interrupt smoke

Use this to validate playback-side cancel behavior without a microphone:

```bash
.venv/bin/python scripts/local_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-demo-input.wav \
  --output-dir /tmp/bionic-local-demo-cancel \
  --chunk-ms 40 \
  --no-play-audio \
  --cancel-after-ms 300
```

`--cancel-after-ms` starts counting when the first local TTS chunk enters playback. For `--cancel-after-ms 0`, the client sends `client.turn.cancel` immediately after first local playback starts.

Expected summary fields:

- `terminal_event` is usually `server.turn.cancelled` for a successful interrupt smoke.
- `client_interrupt_sent_ms` is not null.
- `server_playback_stop_received_ms` is not null if the server emitted `server.playback.stop`.
- `client_interrupt_to_audio_stop_ms` is non-negative when local audio was stopped after the interrupt.
- `client_interrupt_to_face_clear_ms` is non-negative when local face buffers were cleared after the interrupt.
- `client_stale_audio_drop_count` and `client_stale_face_drop_count` record stale old-generation drops.
````

- [ ] **Step 2: Verify docs mention the new fields**

Run:

```bash
rg -n "Playback interrupt smoke|client_interrupt_sent_ms|client_interrupt_to_audio_stop_ms|client_interrupt_to_face_clear_ms" docs/operations/local-demo-client.md
```

Expected: all terms are found.

- [ ] **Step 3: Commit**

```bash
git add docs/operations/local-demo-client.md
git commit -m "docs: document local playback interrupt smoke"
```

---

### Task 13.5: Final verification

**Files:**
- No new files.

**Interfaces:**
- Consumes: completed Task 13 implementation.
- Produces: verification evidence before merge or push.

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: PASS with integration skips only.

- [ ] **Step 3: Run a no-audio real smoke if a server is already running**

Run:

```bash
.venv/bin/python scripts/local_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-demo-input.wav \
  --output-dir /tmp/bionic-task13-cancel-smoke \
  --chunk-ms 40 \
  --no-play-audio \
  --cancel-after-ms 0
```

Expected: command exits successfully and writes `summary.json`. If no server is running on port 8005, skip this smoke and state that it was not run because the server was unavailable.

- [ ] **Step 4: Inspect git state**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: branch is `task13-client-playback-interrupt`; working tree is clean after commits.
