# Demo acceptance package

Task 19 provides a repeatable local demo acceptance command.

## Start a server

Mock provider example:

```bash
cd /home/user/code/端到端
source .venv/bin/activate

PYTHONPATH=src BIONIC_CONFIG=config/mock.json \
  .venv/bin/uvicorn bionic_head.api.app:create_app \
  --factory --host 127.0.0.1 --port 8005
```

## Fake acceptance

```bash
.venv/bin/python scripts/run_demo_acceptance.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --http-base-url http://127.0.0.1:8005 \
  --output-dir /tmp/bionic-demo-acceptance \
  --mode fake \
  --audio-backend null \
  --playback-sync immediate_audio wait_for_face
```

Expected output:

```text
demo_acceptance_report.json
success: true
checks.scripted_interactive_smoke.success: true
checks.history_smoke.success: true
checks.playback_interrupt_smoke.success: true
checks.av_sync_immediate_audio.success: true
checks.av_sync_wait_for_face.success: true
```

## Real acceptance

```bash
.venv/bin/python scripts/run_demo_acceptance.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --http-base-url http://127.0.0.1:8005 \
  --output-dir /tmp/bionic-demo-acceptance-real \
  --mode real \
  --history-turn1-wav /path/to/wo-jiao-xiaozhang.wav \
  --history-turn2-wav /path/to/wo-jiao-shenme.wav \
  --expect 小张
```

## Report and artifacts

The runner writes:

- `demo_acceptance_report.json`
- `generated-input.wav`
- `scripted_interactive_smoke/`
- `history_smoke/`
- `playback_interrupt_smoke/`
- `av_sync_immediate_audio/`
- `av_sync_wait_for_face/`
- `artifacts/`

Inspect `failure_reasons` first when a run fails.

Common failure mapping:

- `server:server_health_unreachable` or `server:server_diagnostics_unreachable`: the API server is not healthy or `/diagnostics` is unreachable; verify the local server process, base URL, and those HTTP endpoints first.
- `history_smoke:*_exception` or other smoke-check exceptions: inspect the check's `failure_message` plus the corresponding `summary.json` / report artifact to find the thrown error quickly.
- `playback_interrupt_smoke:playback_interrupt_failed`: the cancel path did not produce `server.turn.cancelled` or playback-stop evidence; inspect `playback_interrupt_smoke/summary.json` and `client_playback_metrics.json`.
- `av_sync_*:av_sync_wait_for_face_timeout` or `av_sync_*:av_sync_offset_missing`: the wait-for-face path timed out or A/V sync metrics were not emitted; inspect `av_sync_*/interaction_report.json` and `summary.json`.

## Non-goals

This acceptance package does not start the server, install providers, connect UE5, add AEC, or run Blender rendering by default.
