# Client audio/face synchronization

Task 18 adds client-side synchronization for received TTS WAV chunks and UE5 face frame chunks.

The backend protocol is unchanged. The client aligns local playback by `segment_id`.

## Strategies

### immediate_audio

Default behavior:

```text
server.tts.audio arrives
-> play audio immediately
-> display matching UE5 frames when they arrive
```

Use this when you prefer the fastest first audible response.

### wait_for_face

Synchronized behavior:

```text
server.tts.audio arrives
-> cache audio
-> wait for same-segment server.ue5.frames
-> start audio and face playback together
```

Use this when mouth/face timing matters more than the absolute first-audio latency.

If face frames do not arrive within `--wait-for-face-timeout-ms`, the client releases audio as a fallback.

## Local WAV demo

Fast-first-audio mode:

```bash
.venv/bin/python scripts/local_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-demo-input.wav \
  --output-dir /tmp/bionic-av-sync-immediate \
  --chunk-ms 40 \
  --no-play-audio \
  --playback-sync immediate_audio
```

Wait-for-face mode:

```bash
.venv/bin/python scripts/local_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-demo-input.wav \
  --output-dir /tmp/bionic-av-sync-wait \
  --chunk-ms 40 \
  --no-play-audio \
  --playback-sync wait_for_face \
  --wait-for-face-timeout-ms 800
```

## Interactive/scripted demo

Scripted smoke for `immediate_audio`:

```bash
.venv/bin/python scripts/interactive_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --output-dir /tmp/bionic-scripted-immediate \
  --mode scripted \
  --scripted-turns 2 \
  --scripted-cancel-after-ms 300 \
  --mic-backend fake \
  --audio-backend null \
  --chunk-ms 40 \
  --no-play-audio \
  --playback-sync immediate_audio
```

Scripted smoke for `wait_for_face`:

```bash
.venv/bin/python scripts/interactive_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --output-dir /tmp/bionic-scripted-wait-for-face \
  --mode scripted \
  --scripted-turns 2 \
  --scripted-cancel-after-ms 300 \
  --mic-backend fake \
  --audio-backend null \
  --chunk-ms 40 \
  --no-play-audio \
  --playback-sync wait_for_face \
  --wait-for-face-timeout-ms 800
```

The same `--playback-sync` options work in real microphone mode.

## Metrics

Check `summary.json`, `client_playback_metrics.json`, or `interaction_report.json`.

Important fields:

```text
playback_sync_strategy
client_audio_play_start_ms
client_face_first_frame_displayed_ms
client_audio_face_offset_ms
client_audio_wait_for_face_ms
client_face_late_by_ms
client_audio_wait_for_face_timeout
client_playback_stop_to_audio_stop_ms
client_playback_stop_to_face_clear_ms
playback_segments
```

Interpretation:

```text
client_audio_face_offset_ms > 0:
  face displayed after audio start

client_audio_face_offset_ms ≈ 0:
  audio and face started together

client_audio_wait_for_face_ms:
  how much first audio was intentionally delayed by wait_for_face

client_audio_wait_for_face_timeout = true:
  face did not arrive before the timeout, so audio was released anyway
```

Task 18 does not add UE5 runtime integration, WebRTC, acoustic echo cancellation, or backend protocol changes.
