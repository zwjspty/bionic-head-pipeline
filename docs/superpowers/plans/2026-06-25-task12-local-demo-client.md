# Task 12 Local Demo Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local demo client that sends a WAV to `/pipeline/stream`, queues/saves/optionally plays TTS chunks, buffers UE5 frames, handles playback stop, and records client-side playback metrics.

**Architecture:** Reuse `scripts/stream_client.py` protocol helpers for session/audio events and add a focused `scripts/local_demo_client.py` with testable playback engines. Runtime audio playback is optional through `sounddevice`; tests use in-memory sinks and never require a sound card.

**Tech Stack:** Python 3.11, asyncio, websockets, wave, optional sounddevice, pytest.

## Global Constraints

- Work on branch `task12-local-demo-client`.
- Do not change server protocol.
- Do not require real ASR / LLM / TTS / EmoTalk for automated tests.
- Do not require sounddevice or a sound card for automated tests.
- Do not implement microphone capture, AEC, WebRTC, UE5, or real-time Blender playback in Task 12A.
- Preserve existing `scripts/stream_client.py` behavior unless a small helper extraction is needed.

---

## Task 1: Playback metric and buffer engines

**Files:**
- Create: `scripts/local_demo_client.py`
- Create: `tests/unit/test_local_demo_client.py`

**Interfaces:**
- Produces:
  - `PlaybackMetrics(clock: Callable[[], float])`
  - `AudioPlaybackEngine(metrics: PlaybackMetrics, sink: AudioSink | None = None)`
  - `FacePlaybackEngine(metrics: PlaybackMetrics)`
  - `AudioPlaybackEngine.enqueue_wav(chunk_id: str, wav_bytes: bytes, generation_epoch: int | None) -> None`
  - `AudioPlaybackEngine.stop() -> None`
  - `AudioPlaybackEngine.clear() -> None`
  - `FacePlaybackEngine.enqueue_frames(chunk_id: str, payload: dict[str, object], generation_epoch: int | None) -> None`
  - `FacePlaybackEngine.clear() -> None`
  - `PlaybackMetrics.to_dict() -> dict[str, object]`

- [ ] **Step 1: Write failing unit tests**

Add tests proving:

```python
def test_audio_engine_enqueues_wav_and_records_metrics(fake_clock):
    metrics = PlaybackMetrics(clock=fake_clock)
    sink = MemoryAudioSink()
    audio = AudioPlaybackEngine(metrics, sink=sink)

    audio.enqueue_wav("chunk-1", b"RIFF....WAVE", generation_epoch=0)

    assert audio.queued_count == 1
    assert sink.played_chunks == [b"RIFF....WAVE"]
    assert metrics.to_dict()["client_audio_enqueued_count"] == 1
    assert metrics.to_dict()["client_audio_play_start_ms"] == 0.0
```

```python
def test_stop_clears_audio_and_face_buffers(fake_clock):
    metrics = PlaybackMetrics(clock=fake_clock)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)

    audio.enqueue_wav("chunk-1", b"RIFF....WAVE", generation_epoch=0)
    face.enqueue_frames("ue5-1", {"frames": [{"frame_index": 0}]}, generation_epoch=0)
    fake_clock.advance(0.050)

    metrics.mark_playback_stop_received()
    audio.stop()
    audio.clear()
    face.clear()

    summary = metrics.to_dict()
    assert audio.queued_count == 0
    assert face.buffered_chunk_count == 0
    assert summary["client_audio_stopped_ms"] == 50.0
    assert summary["client_face_buffer_cleared_ms"] == 50.0
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

Expected: import or attribute failures because the new classes do not exist.

- [ ] **Step 3: Implement minimal engines**

In `scripts/local_demo_client.py`, implement:

```text
PlaybackMetrics
MemoryAudioSink
AudioPlaybackEngine
FacePlaybackEngine
```

`MemoryAudioSink` should expose `played_chunks: list[bytes]` and `stopped_count: int` for tests.

- [ ] **Step 4: Verify GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

Expected: tests pass.

---

## Task 2: Local demo receiver for TTS binary and UE5 frames

**Files:**
- Modify: `scripts/local_demo_client.py`
- Modify: `tests/unit/test_local_demo_client.py`

**Interfaces:**
- Produces:
  - `LocalDemoReceiver(output_dir: Path, audio: AudioPlaybackEngine, face: FacePlaybackEngine, ...)`
  - `LocalDemoReceiver.accept_json(envelope: dict[str, object]) -> None`
  - `LocalDemoReceiver.accept_binary(payload: bytes) -> None`
  - `LocalDemoReceiver.finish() -> None`

- [ ] **Step 1: Write failing receiver tests**

Add tests proving:

```python
def test_receiver_accepts_tts_metadata_then_binary(tmp_path, server_envelope):
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    receiver.accept_json(server_envelope(
        "server.tts.audio",
        payload={
            "chunk_id": "chunk-1",
            "segment_id": "segment-1",
            "format": "wav",
            "byte_length": 12,
            "generation_epoch": 0,
        },
        generation_epoch=0,
    ))
    receiver.accept_binary(b"RIFF....WAVE")

    assert (tmp_path / "tts" / "chunk-1.wav").read_bytes() == b"RIFF....WAVE"
    assert audio.queued_count == 1
```

```python
def test_receiver_playback_stop_clears_buffers(tmp_path, server_envelope):
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    audio.enqueue_wav("chunk-1", b"RIFF....WAVE", generation_epoch=0)
    face.enqueue_frames("ue5-1", {"frames": [{"frame_index": 0}]}, generation_epoch=0)
    receiver.accept_json(server_envelope("server.playback.stop", payload={}, generation_epoch=1))

    assert audio.queued_count == 0
    assert face.buffered_chunk_count == 0
    assert metrics.to_dict()["client_playback_stop_received_ms"] == 0.0
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

Expected: `LocalDemoReceiver` missing.

- [ ] **Step 3: Implement receiver**

Implement JSON envelope validation equivalent to `ClientReceiver`, pending TTS metadata tracking, binary byte-length validation, artifact saving, UE5 frame saving, stale-generation dropping, and terminal summary writing.

- [ ] **Step 4: Verify GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

Expected: tests pass.

---

## Task 3: WebSocket CLI client

**Files:**
- Modify: `scripts/local_demo_client.py`
- Modify: `tests/unit/test_local_demo_client.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces:
  - `async def run_local_demo(url: str, wav_path: Path, output_dir: Path, chunk_ms: int, play_audio: bool, cancel_after_ms: int | None = None) -> str`
  - CLI `python scripts/local_demo_client.py --url ... --wav ... --output-dir ...`
  - Optional dependency group `client-audio = ["sounddevice>=0.5,<1"]`

- [ ] **Step 1: Write failing CLI/unit tests**

Add tests proving:

```python
def test_build_parser_accepts_no_audio_mode():
    parser = build_parser()
    args = parser.parse_args([
        "--url", "ws://127.0.0.1:8005/pipeline/stream",
        "--wav", "/tmp/input.wav",
        "--output-dir", "/tmp/out",
        "--no-play-audio",
    ])
    assert args.play_audio is False
```

```python
def test_build_parser_accepts_cancel_after_ms():
    parser = build_parser()
    args = parser.parse_args([
        "--url", "ws://127.0.0.1:8005/pipeline/stream",
        "--wav", "/tmp/input.wav",
        "--output-dir", "/tmp/out",
        "--cancel-after-ms", "500",
    ])
    assert args.cancel_after_ms == 500
```

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

- [ ] **Step 3: Implement CLI and WebSocket loop**

Reuse these helpers from `scripts.stream_client`:

```python
from scripts.stream_client import client_event, pcm_chunks, read_pcm16_from_wav
```

The loop should:

```text
connect
send client.session.start
wait server.session.ready
send client.audio.start
send client.audio.chunk JSON + binary chunks
send client.audio.end
receive until terminal event
dispatch JSON/binary to LocalDemoReceiver
write summary files
```

If `cancel_after_ms` is set, schedule `client.turn.cancel` after that delay using the same session/turn and the next client sequence number.

- [ ] **Step 4: Add optional audio dependency**

In `pyproject.toml`, add:

```toml
client-audio = ["sounddevice>=0.5,<1"]
```

Do not add `sounddevice` to default dependencies.

- [ ] **Step 5: Verify GREEN**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_local_demo_client.py -q
```

---

## Task 4: Documentation and manual smoke commands

**Files:**
- Create: `docs/operations/local-demo-client.md`
- Modify: `README.md` if it exists and has a current-status section.

**Interfaces:**
- Produces documented commands for:
  - no-audio local demo
  - optional audio local demo
  - GPU EmoTalk server startup
  - generated metrics/artifact paths

- [ ] **Step 1: Write operations doc**

Document:

```bash
cd /home/user/code/端到端
source .venv/bin/activate

PYTHONPATH=src BIONIC_CONFIG=/tmp/bionic-local-emotalk-gpu.json \
  .venv/bin/uvicorn bionic_head.api.app:create_app \
  --factory --host 127.0.0.1 --port 8005
```

And:

```bash
PYTHONPATH=src .venv/bin/python scripts/local_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-demo-input.wav \
  --output-dir /tmp/bionic-local-demo \
  --chunk-ms 40 \
  --no-play-audio
```

For real audio:

```bash
.venv/bin/python -m pip install -e ".[client-audio]"
PYTHONPATH=src .venv/bin/python scripts/local_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-demo-input.wav \
  --output-dir /tmp/bionic-local-demo-audio \
  --chunk-ms 40 \
  --play-audio
```

- [ ] **Step 2: Run full tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Optional manual no-audio smoke**

With the server running, execute the no-audio command and confirm:

```text
terminal_event=server.pipeline.done
client_playback_metrics.json exists
tts/*.wav exists
ue5/*.json exists
```

- [ ] **Step 4: Commit**

```bash
git add scripts/local_demo_client.py tests/unit/test_local_demo_client.py pyproject.toml docs/operations/local-demo-client.md
git commit -m "feat: add local demo playback client"
```

---

## Self-Review

- Spec coverage: Tasks cover playback engines, receiver, CLI, optional audio backend, metrics, stale-drop, stop handling, tests, and docs.
- Placeholder scan: No task relies on unspecified future behavior.
- Type consistency: Public names are consistent across tasks.
- Out of scope: microphone, AEC, WebRTC, UE5, and real-time Blender are intentionally excluded.
