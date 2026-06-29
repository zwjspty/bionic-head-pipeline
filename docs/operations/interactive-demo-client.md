# Interactive demo client operations

This page covers the Task 14/15 interactive client:

```text
scripts/interactive_demo_client.py
```

It has two modes:

- `interactive`: human terminal control with microphone input;
- `scripted`: repeatable fake-mic smoke for local acceptance.

For Task 18 client-side audio/face synchronization options, see
`docs/operations/client-audio-face-sync.md`.

## Start a stream server first

Example:

```bash
cd /home/user/code/端到端
source .venv/bin/activate

PYTHONPATH=src BIONIC_CONFIG=config/mock.json \
  .venv/bin/uvicorn bionic_head.api.app:create_app \
  --factory --host 127.0.0.1 --port 8005
```

Use your real provider config instead of `config/mock.json` when doing real local acceptance.

## Scripted fake-mic smoke

This is the preferred Task 15 automated acceptance command. It does not require a real microphone or speaker:

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
  --no-play-audio \
  --playback-sync immediate_audio
```

Expected behavior:

```text
turn 1 fake recording
-> server TTS / UE5 frames
-> local null playback starts
-> client sends client.turn.cancel
-> server.playback.stop clears local audio and face buffers
-> server.turn.cancelled
-> turn 2 fake recording
-> server.pipeline.done
```

Expected artifacts:

- `summary.json`
- `client_playback_metrics.json`
- `interaction_report.json`
- `tts/*.wav`
- `ue5/*.json`

`interaction_report.json` should contain:

```text
success: true
mode: scripted
turn_count: 2
completed_turn_count: 1
cancelled_turn_count: 1
playback_stop_count: >= 1
old_generation_audio_play_count: 0
old_generation_face_display_count: 0
```

To validate synchronized playback, rerun the same smoke with:

```bash
--playback-sync wait_for_face --wait-for-face-timeout-ms 800
```

Then check `client_audio_face_offset_ms`, `client_audio_wait_for_face_ms`, and `client_audio_wait_for_face_timeout` in `summary.json` or `interaction_report.json`.

With Task 16 enabled, the same command also exercises the normal multi-turn WebSocket session path while keeping the first-turn cancel behavior. The client-side report does not inspect server-internal LLM prompts; session history is verified on the server side through stream tests and `timeline.json` history metrics.

For a completed turn, the server-side run timeline contains:

```text
stream.history_enabled: true
stream.history_turn_count_before
stream.history_char_count_before
stream.history_turn_count_after
stream.history_char_count_after
```

The expected Task 16 behavior is:

```text
same WebSocket session
-> successful turn appends user + assistant
-> later turn LLM receives previous user/assistant messages
-> cancelled / stale / error turns do not append assistant replies
```

## Task 16.5 history smoke

The history smoke verifies that two turns in the same WebSocket session can share short-term conversation history.

Real provider acceptance:

```bash
PYTHONPATH=src .venv/bin/python scripts/history_smoke.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --output-dir /tmp/bionic-history-smoke-real \
  --mode real \
  --turn1-wav /path/to/wo-jiao-xiaozhang.wav \
  --turn2-wav /path/to/wo-jiao-shenme.wav \
  --expect 小张
```

Expected report:

```text
history_smoke_report.json
success: true
turn 2 history_turn_count_before > 0
turn 2 llm_reply contains 小张
```

If the real smoke fails, inspect `asr_text` first. A wrong ASR transcript means the audio sample or ASR provider failed before history can be judged.

## Real interactive microphone mode

Install optional audio dependencies:

```bash
.venv/bin/python -m pip install -e ".[client,client-audio]"
```

Run:

```bash
.venv/bin/python scripts/interactive_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --output-dir /tmp/bionic-interactive-demo \
  --mode interactive \
  --mic-backend sounddevice \
  --audio-backend sounddevice \
  --chunk-ms 40 \
  --play-audio
```

Controls:

- `Enter`: start recording if idle, stop recording if already recording.
- `c`: cancel current playback by sending `client.turn.cancel`.
- `q`: quit the client.

Wear headphones for real microphone testing. This client does not include acoustic echo cancellation, so speaker output can be captured by the microphone and sent back into the pipeline.

## Headless interactive protocol smoke

If you want to exercise the interactive client without real microphone or speaker hardware:

```bash
.venv/bin/python scripts/interactive_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --output-dir /tmp/bionic-interactive-fake \
  --mode interactive \
  --mic-backend fake \
  --audio-backend null \
  --chunk-ms 40 \
  --no-play-audio
```

In this mode, pressing `Enter` starts/stops fake PCM16LE input.
