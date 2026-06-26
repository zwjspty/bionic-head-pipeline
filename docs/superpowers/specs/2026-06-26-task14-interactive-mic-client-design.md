# Task 14: Interactive Mic Client Design

## Summary

Task 14 adds a minimal real interaction client for the local lab demo:

```text
keyboard command
-> microphone capture
-> /pipeline/stream WebSocket
-> existing TTS playback + UE5 frame buffering
-> keyboard interrupt
-> client.turn.cancel
```

This task intentionally does not add acoustic echo cancellation, WebRTC, browser UI, UE5 runtime integration, or real-time Blender rendering. It reuses the existing `bionic-head-stream-v1` WebSocket protocol and the Task 12/13 playback receiver logic.

## Scope

The first version is terminal-driven:

```text
Enter: start recording / stop recording
c: interrupt current playback
q: quit
```

The input model is toggle-to-talk instead of push-to-talk. Toggle-to-talk is chosen because reliable cross-platform "key down / key up" handling in a terminal is messy and not needed for the first real local demo. Pressing Enter once starts recording; pressing Enter again stops recording and sends `client.audio.end`.

## Components

### `scripts/interactive_demo_client.py`

New CLI script that owns the interactive session. It should:

- connect to `ws://.../pipeline/stream`;
- send `client.session.start`;
- wait for `server.session.ready`;
- use keyboard commands from stdin;
- support `--mic-backend sounddevice|fake`;
- support `--audio-backend sounddevice|null`;
- capture microphone chunks while recording;
- send `client.audio.start`, paired `client.audio.chunk` JSON and PCM binary frames, then `client.audio.end`;
- reuse `LocalDemoReceiver`, `AudioPlaybackEngine`, `FacePlaybackEngine`, `PlaybackMetrics`, and audio playback sinks from `scripts/local_demo_client.py`;
- send `client.turn.cancel` on `c`;
- stop local playback and face buffers when server stop/cancel events arrive;
- write `summary.json`, `client_playback_metrics.json`, `tts/*.wav`, and `ue5/*.json` through the reused receiver.

### `MicrophoneInput`

Define a small protocol-like interface:

```python
class MicrophoneInput(Protocol):
    async def start(self) -> None: ...
    async def read_chunk(self) -> bytes: ...
    async def stop(self) -> None: ...
    async def close(self) -> None: ...
```

Default tests use a fake microphone. The real backend uses `sounddevice.InputStream`, converts captured samples to PCM signed 16-bit little-endian, mono, 16000 Hz, and yields chunks sized by `--chunk-ms`.

The CLI exposes both:

```text
--mic-backend sounddevice
--mic-backend fake
```

The fake backend is for no-hardware protocol smoke runs. It emits legal PCM16LE chunks and must not depend on a real sound device.

### Keyboard command source

Define a testable command source:

```python
class CommandSource(Protocol):
    async def read_command(self) -> str: ...
```

The real source reads lines from stdin via `asyncio.to_thread(input, prompt)`. Tests use a scripted command source.

### Session controller

An `InteractiveDemoSession` coordinates:

```text
command reader
microphone input
websocket send/receive
LocalDemoReceiver playback handling
```

It keeps one recording task at a time. `Enter` toggles recording. `c` sends cancel once for the active turn/playback. `q` cancels any recording, closes playback, and exits.

## Metrics

Reuse existing Task 13 metrics where possible:

- `client_interrupt_sent_ms`
- `server_playback_stop_received_ms`
- `client_interrupt_to_audio_stop_ms`
- `client_interrupt_to_face_clear_ms`
- stale audio/face drop counts

Add minimal mic metrics to summary:

- `client_mic_recording_started_count`
- `client_mic_recording_stopped_count`
- `client_mic_chunks_sent`
- `client_mic_bytes_sent`
- `client_manual_cancel_count`

## Error Handling

- Missing `websockets` fails with a friendly install message.
- Missing `sounddevice` fails only when the real microphone backend is requested.
- Invalid commands print help and keep the session alive.
- `q` while recording stops the microphone and sends `client.audio.end` if a turn has started.
- Ctrl-C should attempt to stop microphone, stop local playback, finish summary, and exit.
- If the server closes or emits terminal error/cancel/done, the client should finish the current turn summary without crashing.
- Real speaker playback should be documented with a headphone warning because Task 14 does not include acoustic echo cancellation.

## Testing Strategy

Default pytest must not require a microphone, sound device, GPU, Ollama, Piper, EmoTalk, or a running FastAPI server.

Tests use:

- fake WebSocket;
- fake microphone chunks;
- fake command source;
- memory audio sink;
- fake clock where metrics need exact values.

Real hardware use is documented as an operation command, not part of default tests.

## Acceptance Criteria

Task 14 is complete when:

1. full pytest passes;
2. `scripts/interactive_demo_client.py --help` runs by path;
3. fake-mic tests prove `Enter -> audio.start/chunk/end`;
4. fake-mic tests prove `c -> client.turn.cancel`;
5. playback stop from server clears local audio and face buffers through existing receiver behavior;
6. docs explain how to run the real interactive client;
7. no AEC, WebRTC, browser UI, real UE5 runtime, or real-time Blender scope is added.
