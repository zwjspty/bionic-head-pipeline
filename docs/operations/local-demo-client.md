# Local demo client operations

Use this flow to run a local playback demo against a running `pipeline/stream` server.

Start server (GPU EmoTalk example path in `BIONIC_CONFIG` as in Task 12.4 brief):

```bash
cd /home/user/code/端到端
source .venv/bin/activate

PYTHONPATH=src BIONIC_CONFIG=/tmp/bionic-local-emotalk-gpu.json \
  .venv/bin/uvicorn bionic_head.api.app:create_app \
  --factory --host 127.0.0.1 --port 8005
```

Run no-audio local demo (recommended for headless smoke, CI-like, or other automation environments):

```bash
.venv/bin/python scripts/local_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-demo-input.wav \
  --output-dir /tmp/bionic-local-demo \
  --chunk-ms 40 \
  --no-play-audio
```

Run with optional local playback (fresh install needs both `client` and `client-audio` extras):

```bash
.venv/bin/python -m pip install -e ".[client,client-audio]"
.venv/bin/python scripts/local_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-demo-input.wav \
  --output-dir /tmp/bionic-local-demo-audio \
  --chunk-ms 40 \
  --play-audio
```

Generated artifacts:

- `summary.json` (embeds playback counters/timestamps from `PlaybackMetrics`)
- `client_playback_metrics.json` (standalone copy of the client playback metrics)
- `tts/{chunk_id}.wav`
- `ue5/{chunk_id}.json`

For no-audio verification, expect:

- CLI prints `{"terminal_event":"server.pipeline.done", ...}`
- `summary.json` exists
- `client_playback_metrics.json` exists
- `tts/*.wav` exists
- `ue5/*.json` exists
- `terminal_event` is `server.pipeline.done`

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

Note: `--play-audio` remains the default, but `--no-play-audio` is the preferred flag for headless smoke runs. If playback is enabled, `sounddevice` is optional at runtime and used only when `--play-audio` is set.
