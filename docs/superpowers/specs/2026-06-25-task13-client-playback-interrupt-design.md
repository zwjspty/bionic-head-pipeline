# Task 13: Client Playback Interrupt Design

## Summary

Task 13 validates the local demo client's playback-side interrupt behavior. It does not add microphone input, acoustic echo cancellation, WebRTC, browser UI, UE5 runtime integration, or real-time Blender playback. It focuses on the client path that already exists after Task 12:

```text
server.tts.audio + WAV binary
-> AudioPlaybackEngine

server.ue5.frames
-> FacePlaybackEngine

server.playback.stop / server.turn.cancelled
-> stop audio, clear face, drop stale generation data, record metrics
```

The user-facing goal is simple: when a turn is cancelled during playback, the local demo client must stop sounding and stop displaying old face frames quickly and measurably.

## Current State

`scripts/local_demo_client.py` already supports:

- WebSocket stream connection;
- `server.tts.audio` JSON plus following WAV binary;
- `server.ue5.frames` buffering;
- `server.playback.stop` and `server.turn.cancelled` clearing pending playback;
- stale generation drop accounting;
- `--cancel-after-ms`.

The gap is that `--cancel-after-ms` currently schedules from after the client finishes sending input audio, not from local playback start. This can cancel before playback begins and does not measure the real playback-side interrupt experience. The metrics also do not yet record the moment when the client sent its interrupt request, nor the elapsed time from that local interrupt request to playback stop, audio stop, and face buffer clear.

## Design Decision

Use playback-start anchored cancellation:

```text
first local audio play starts
-> wait --cancel-after-ms
-> send client.turn.cancel
-> mark client_interrupt_sent_ms
-> receive server.playback.stop or server.turn.cancelled
-> stop audio sink
-> clear audio queue
-> clear face queue
-> record stop deltas
```

This choice makes the smoke test represent the real concern: "what happens while the client is already playing an answer?" It avoids microphone, AEC, and server-side changes.

## Components

### `PlaybackMetrics`

Add interrupt-related metrics:

- `client_interrupt_sent_ms`
- `server_playback_stop_received_ms`
- `client_interrupt_to_playback_stop_ms`
- `client_interrupt_to_audio_stop_ms`
- `client_interrupt_to_face_clear_ms`

Keep existing names such as `client_playback_stop_received_ms` for backward compatibility. `server_playback_stop_received_ms` should mirror the same timestamp because the observed stop originates from a server event.

### `AudioPlaybackEngine`

Add an optional callback invoked exactly once when the first WAV actually enters playback:

```python
on_first_play: Callable[[], None] | None
```

The callback schedules cancellation in `run_local_demo`. It is called after `client_audio_play_start_ms` is recorded and before the sink play call returns. Tests use `MemoryAudioSink`; real audio remains optional through `SoundDeviceAudioSink`.

### `FacePlaybackEngine`

No new protocol behavior is needed. It already buffers frames and clears them. Task 13 only strengthens metrics so `client_interrupt_to_face_clear_ms` is recorded when a stop clears the face buffer.

### `LocalDemoReceiver`

`server.playback.stop` and `server.turn.cancelled` remain authoritative stop signals. Handling them must:

```text
mark server stop received
clear pending TTS metadata
stop audio
clear audio queue
clear face queue
clear UE5 sequence tracking
record summary counters
```

Stale lower-generation TTS and UE5 frames received after a newer stop must not be saved, played, or displayed.

### `run_local_demo`

`--cancel-after-ms` changes from "delay after input send" to "delay after first local audio playback start". For `--cancel-after-ms 0`, cancellation is sent immediately after first local audio playback starts.

The client sends one `client.turn.cancel` event per run. The outgoing event uses the next available client sequence number and records:

```text
client_interrupt_sent_ms
```

## Metrics Contract

`summary.json` and `client_playback_metrics.json` must include:

```text
client_interrupt_sent_ms
server_playback_stop_received_ms
client_playback_stop_received_ms
client_audio_stopped_ms
client_face_buffer_cleared_ms
client_interrupt_to_playback_stop_ms
client_interrupt_to_audio_stop_ms
client_interrupt_to_face_clear_ms
client_stale_audio_drop_count
client_stale_face_drop_count
```

The most important acceptance values are:

```text
client_interrupt_to_audio_stop_ms >= 0
client_interrupt_to_face_clear_ms >= 0
client_stale_audio_drop_count remains accurate
client_stale_face_drop_count remains accurate
```

## Error Handling

- Missing `websockets` still fails with the existing friendly install message.
- Missing `sounddevice` still affects only `--play-audio`.
- If `--cancel-after-ms` is provided but no audio ever starts, no cancel is sent and the run terminates based on server events.
- If `server.playback.stop` arrives without a prior local interrupt, stop metrics still record server stop and audio/face clearing. Interrupt delta metrics remain `null`.
- If `server.turn.cancelled` arrives without a preceding `server.playback.stop`, it is treated as a stop signal for local playback.

## Testing Strategy

Default tests remain hermetic:

- no microphone;
- no sound card;
- no GPU;
- no Ollama;
- no Piper;
- no EmoTalk;
- no running FastAPI server.

Unit tests use fake WebSocket responses, `MemoryAudioSink`, and fake clocks. A real smoke command is documented but not part of default pytest.

## Acceptance Criteria

Task 13 is complete when:

1. full pytest passes;
2. `local_demo_client.py --cancel-after-ms N` sends cancel only after first local audio playback starts;
3. `server.playback.stop` stops audio and clears face buffering;
4. stale old-generation TTS binaries are consumed but not saved or played;
5. stale old-generation UE5 frames are not displayed;
6. `summary.json` and `client_playback_metrics.json` contain interrupt delta metrics;
7. local demo docs include a playback interrupt smoke command;
8. no microphone, AEC, WebRTC, browser UI, real UE5 runtime, or real-time Blender scope is added.
