# Task 15: Scripted Interactive Demo Smoke Design

## Summary

Task 15 upgrades the Task 14 terminal microphone client from a manually driven demo into a repeatable local smoke test:

```text
fake microphone
-> /pipeline/stream WebSocket
-> local playback receiver
-> automatic cancel while playback is active
-> playback.stop verification
-> second fake turn
-> interaction_report.json
```

The scripted mode is intentionally hardware-free by default. It does not require a real microphone, speaker, GPU, Ollama, Piper, EmoTalk, or a running server in ordinary unit tests. A real smoke run still targets a running `/pipeline/stream` server.

## Scope

### In scope

- Add `--mode interactive|scripted` to `scripts/interactive_demo_client.py`.
- Keep `interactive` as the default mode.
- Use `--mic-backend fake` and `--audio-backend null` for scripted smoke.
- Automatically run at least two turns.
- Cancel the first turn after local playback starts.
- Continue with a second turn after `server.turn.cancelled`.
- Reuse `LocalDemoReceiver`, `AudioPlaybackEngine`, `FacePlaybackEngine`, `PlaybackMetrics`, playback stop handling, and stale-drop logic.
- Write `interaction_report.json` next to `summary.json`.

### Out of scope

- Real microphone automated testing.
- Acoustic echo cancellation.
- WebRTC.
- Browser UI.
- Real UE5 runtime.
- ASR partial / true duplex ASR.
- Real-time Blender rendering.

## CLI

Scripted smoke command:

```bash
.venv/bin/python scripts/interactive_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --output-dir /tmp/bionic-task15-scripted \
  --mode scripted \
  --scripted-turns 2 \
  --scripted-cancel-after-ms 300 \
  --mic-backend fake \
  --audio-backend null \
  --chunk-ms 40 \
  --no-play-audio
```

The command should run without keyboard input. It sends fake PCM16LE audio chunks, waits for server events, cancels the first playback, then sends a second fake turn.

## Scripted flow

The first version uses deterministic command generation rather than a separate process:

```text
1. client.session.start
2. turn 1: client.audio.start -> chunk/binary -> client.audio.end
3. wait until first local TTS playback starts
4. wait scripted_cancel_after_ms
5. client.turn.cancel
6. wait for server.playback.stop / server.turn.cancelled
7. turn 2: client.audio.start -> chunk/binary -> client.audio.end
8. wait for server.pipeline.done
9. write summary.json and interaction_report.json
```

If a server implementation emits `server.playback.stop` and `server.turn.cancelled` quickly, the client must still verify that audio stop and face clear paths ran through the existing receiver logic.

## Report

`interaction_report.json` should contain a compact acceptance-oriented summary:

```json
{
  "success": true,
  "mode": "scripted",
  "turn_count": 2,
  "completed_turn_count": 1,
  "cancelled_turn_count": 1,
  "playback_stop_count": 1,
  "old_generation_audio_play_count": 0,
  "old_generation_face_display_count": 0,
  "client_stale_audio_drop_count": 0,
  "client_stale_face_drop_count": 0,
  "client_interrupt_sent_ms": 1234.5,
  "server_playback_stop_received_ms": 1240.2,
  "client_audio_stopped_ms": 1240.8,
  "client_face_buffer_cleared_ms": 1240.9,
  "client_interrupt_to_playback_stop_ms": 5.7,
  "client_interrupt_to_audio_stop_ms": 6.3,
  "client_interrupt_to_face_clear_ms": 6.4
}
```

For unit tests, exact timing values can be fake-clock driven. For real smoke runs, non-null values and zero stale old-generation playback counts are more important than exact milliseconds.

## Success criteria

Task 15 is complete when:

1. full pytest passes;
2. default tests require no real microphone, sound device, GPU, Ollama, Piper, EmoTalk, or running FastAPI server;
3. `interactive_demo_client.py` supports `--mode scripted`;
4. scripted mode can drive at least two fake microphone turns;
5. first scripted turn can be automatically cancelled after playback starts;
6. `server.playback.stop` triggers local audio stop and face buffer clear through existing receiver behavior;
7. second scripted turn can continue after the cancellation;
8. `interaction_report.json` exists and has `success=true`;
9. old-generation audio/face playback counts are zero.
