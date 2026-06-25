# Task 12 Local Demo Client Design

## Goal

Task 12 builds the first real local playback client for the bionic-head pipeline. The user should be able to send a local WAV file to `/pipeline/stream`, hear server TTS audio through a real playback path, receive face/UE5 frames into a playback buffer, and measure client-side playback/stop latency.

## Scope

Task 12A is intentionally smaller than a browser/WebRTC/UE5 client:

```text
Input: local mono PCM16 16 kHz WAV file
Transport: existing WebSocket `/pipeline/stream`
Audio output: WAV chunk queue with optional local playback
Face output: UE5 frame queue and timing metrics, no final 3D runtime yet
Interrupt: keyboard or scripted cancel sends `client.turn.cancel`
Metrics: real client receive/play/stop/buffer timings
```

The first implementation does not add microphone capture, AEC, WebRTC, true UE5, or real-time Blender playback.

## Current Code Fit

`scripts/stream_client.py` already validates the streaming protocol, sends WAV PCM chunks, saves `server.tts.audio` binaries, saves `server.ue5.frames`, tracks generation epochs, and records server event timing. Task 12 should reuse its protocol helpers and extend the client side with explicit playback engines.

The new client should not change server protocol behavior. It is a consumer of the existing stream events.

## Architecture

Add a local demo client with three small units:

```text
LocalDemoClient
  -> sends WAV as client.audio.* events
  -> receives JSON and binary frames
  -> dispatches TTS and UE5 payloads to playback engines

AudioPlaybackEngine
  -> stores WAV chunks
  -> optional sounddevice playback when requested
  -> stop() and clear() on server.playback.stop / server.turn.cancelled

FacePlaybackEngine
  -> stores UE5 frame chunks by segment/generation
  -> drops stale generations
  -> records first-frame and clear timings
```

Playback must be testable without a real sound card. The default automated-test path uses an in-memory playback sink; runtime playback can be enabled with an optional `sounddevice` backend.

## Event Handling

The client keeps the existing JSON-metadata-then-binary rule for TTS:

```text
server.tts.audio JSON
next binary message = WAV bytes for that chunk
```

For `server.playback.stop` and terminal cancellation events, the client immediately:

```text
audio.stop()
audio.clear()
face.clear()
record client_playback_stop_received_ms
record client_audio_stopped_ms
record client_face_buffer_cleared_ms
```

For stale generations, the client drops audio and face chunks whose `generation_epoch` is older than the latest observed generation and increments stale-drop counters.

## Metrics

Task 12 adds client-side metrics that complement existing server timings:

```text
client_tts_received_ms
client_audio_enqueued_count
client_audio_play_start_ms
client_audio_stopped_ms
client_ue5_first_frame_received_ms
client_face_buffered_chunk_count
client_face_first_frame_displayed_ms
client_audio_face_offset_ms
client_playback_stop_received_ms
client_interrupt_to_audio_stop_ms
client_face_buffer_cleared_ms
client_stale_audio_drop_count
client_stale_face_drop_count
```

When actual audio playback is disabled, the metrics still record enqueue/clear timing and set playback-specific fields to `null` rather than pretending audio reached a speaker.

## CLI

Add:

```bash
python scripts/local_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-demo-input.wav \
  --output-dir /tmp/bionic-local-demo \
  --chunk-ms 40 \
  --play-audio
```

Useful flags:

```text
--play-audio       Try sounddevice playback.
--no-play-audio    Save and queue audio only.
--cancel-after-ms  Send client.turn.cancel after this many ms.
--stop-on-terminal Exit after server.pipeline.done / error / turn.cancelled.
```

## Error Handling

The client fails fast on protocol corruption:

```text
wrong sequence
binary without pending TTS metadata
binary byte_length mismatch
unsupported TTS format
WAV input not mono PCM16 16 kHz
```

Optional audio playback errors should not hide protocol success. If `--play-audio` is set and `sounddevice` is missing or the audio device fails, the CLI exits with a clear message before starting the WebSocket session.

## Tests

Automated tests must not require a running server, a sound card, GPU, Piper, or EmoTalk. Unit tests cover:

```text
AudioPlaybackEngine enqueue/play/stop/clear
FacePlaybackEngine enqueue/drop stale/clear
Local demo receiver handles TTS metadata + binary
playback.stop clears both queues
stale generation audio/face is dropped
metrics summary is written
```

The real manual test runs against a local server started with the GPU EmoTalk config.

## Out of Scope

```text
No microphone capture.
No AEC.
No browser/WebRTC.
No UE5 engine integration.
No real-time Blender playback.
No ASR partial streaming.
No server protocol change.
No model/provider optimization.
```

## Acceptance

Task 12A is complete when:

```text
1. `scripts/local_demo_client.py` connects to `/pipeline/stream`.
2. It sends a local WAV and receives server TTS and UE5 events.
3. It saves received TTS/UE5 artifacts and writes `client_playback_metrics.json`.
4. In no-audio mode, tests pass without sounddevice.
5. In audio mode, a developer can hear Piper WAV chunks on a local machine with sounddevice installed.
6. `server.playback.stop` clears audio and face buffers.
7. Old generation audio/face chunks are not played or displayed.
8. Full pytest passes.
```
