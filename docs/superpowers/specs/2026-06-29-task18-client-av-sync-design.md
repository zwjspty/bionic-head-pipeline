# Task 18 Client AV Sync Design

## Goal

Task 18 adds client-side audio/face synchronization to the local demo clients.
The backend already emits `server.tts.audio` and `server.ue5.frames` with a
shared `segment_id`; this task makes the client use that relationship to choose
when audio and face playback begin, and to record measurable audio/face offset.

## Non-goals

- No backend protocol changes.
- No UE5 integration.
- No WebRTC or AEC.
- No ASR, TTS, LLM, EmoTalk, sidecar, or provider changes.
- No dependency on a real sound card, microphone, GPU, Ollama, Piper, or EmoTalk
  for default tests.

## Playback strategies

### `immediate_audio`

When a TTS binary arrives, the client enqueues audio immediately. UE5 frames for
the same segment may arrive later and then start face playback. This preserves
the current low first-audio latency.

Metrics:

- `client_audio_wait_for_face_ms = 0`
- `client_face_late_by_ms = face_first_frame_displayed_ms - audio_play_start_ms`
- `client_audio_face_offset_ms = face_first_frame_displayed_ms - audio_play_start_ms`

### `wait_for_face`

When TTS arrives, the client stores the audio segment. Playback starts only when
the same `segment_id` has at least one UE5 frame chunk. If face frames do not
arrive within `wait_for_face_timeout_ms`, the client falls back to audio playback
instead of waiting forever.

Metrics:

- `client_audio_wait_for_face_ms = audio_play_start_ms - tts_received_ms`
- `client_audio_face_offset_ms` should be near zero for successful synchronized
  starts.
- `client_audio_wait_for_face_timeout = true` when timeout fallback was needed.

Default strategy is `immediate_audio` to avoid surprising users with extra
latency.

## Components

### `PlaybackClock`

Lives in `src/bionic_head/client/playback_clock.py`. It wraps a monotonic clock
and records client-side timestamps:

- TTS received
- UE5 first frame received
- audio play start
- face first frame displayed
- playback.stop received
- audio stopped
- face buffer cleared

It computes:

- `client_audio_face_offset_ms`
- `client_audio_wait_for_face_ms`
- `client_face_late_by_ms`
- `client_playback_stop_to_audio_stop_ms`
- `client_playback_stop_to_face_clear_ms`

### `SegmentSyncCoordinator`

Lives in `src/bionic_head/client/segment_sync.py`. It owns per-segment sync
state keyed by `segment_id` and `generation_epoch`.

Responsibilities:

- Track TTS bytes and UE5 frame payloads per segment.
- Implement `immediate_audio` and `wait_for_face`.
- Drop stale generation segments.
- Clear all pending state on playback stop/cancel.
- Emit ready actions for audio and face playback without playing them directly.

The existing `AudioPlaybackEngine` and `FacePlaybackEngine` remain responsible
for actual playback/queue behavior. The coordinator only decides when each piece
is allowed to enter those engines.

## Client integration

`scripts/local_demo_client.py` and `scripts/interactive_demo_client.py` both get:

```bash
--playback-sync immediate_audio
--playback-sync wait_for_face
--wait-for-face-timeout-ms 800
```

`local_demo_client.py` uses the same receiver path as before, but TTS binaries
and UE5 frames go through `SegmentSyncCoordinator` before being enqueued.

`interactive_demo_client.py` passes the strategy into `LocalDemoReceiver` for
interactive and scripted modes. Scripted smoke must be able to run both
strategies with fake mic and null audio.

## Summary fields

Top-level summary and `client_playback_metrics.json` include:

- `playback_sync_strategy`
- `client_tts_received_ms`
- `client_ue5_first_frame_received_ms`
- `client_audio_play_start_ms`
- `client_face_first_frame_displayed_ms`
- `client_audio_face_offset_ms`
- `client_audio_wait_for_face_ms`
- `client_face_late_by_ms`
- `client_audio_wait_for_face_timeout`
- `client_playback_stop_to_audio_stop_ms`
- `client_playback_stop_to_face_clear_ms`

Per-segment summary is stored under `playback_segments`.

## Error handling

- Unknown strategy is rejected by argument parsing / coordinator validation.
- Stale generation audio and face are dropped before playback.
- `playback.stop`, `server.turn.cancelled`, and local cancel clear pending sync
  state, audio queue, and face queue.
- `wait_for_face` timeout fallback prevents client silence if face frames are
  lost or delayed indefinitely.

## Acceptance

- `local_demo_client.py` supports both strategies.
- `interactive_demo_client.py` supports both strategies.
- `immediate_audio` starts audio when TTS binary arrives.
- `wait_for_face` starts audio only after same-segment UE5 frames arrive, unless
  timeout fallback fires.
- Summary contains audio/face offset metrics.
- `playback.stop` clears pending state in both strategies.
- Old generation audio/face do not play.
- Full pytest passes.
