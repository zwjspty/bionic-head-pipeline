# Task 19 Demo Acceptance Package Design

## Summary

Task 19 turns the current local interactive digital-human MVP into a repeatable acceptance package. It does not add a new model, protocol, UE5 integration, AEC, or pipeline behavior. It creates a small orchestration layer that runs existing smoke clients, checks the server, collects artifacts, and writes a single `demo_acceptance_report.json`.

The goal is to make the current demo state easy to prove:

- scripted interactive smoke still works;
- playback interrupt still works;
- session history smoke still works;
- both Task 18 playback sync strategies work;
- key artifacts are gathered in one output directory;
- failures are reported with clear failure codes and paths.

## Current reusable pieces

Existing scripts already cover most execution primitives:

- `scripts/interactive_demo_client.py`
  - scripted fake-mic smoke;
  - first-turn cancel / second-turn completion;
  - `interaction_report.json`;
  - `summary.json`;
  - Task 18 `--playback-sync immediate_audio|wait_for_face`.
- `scripts/history_smoke.py`
  - two-turn history smoke;
  - `mode=mock` without real WAV files;
  - `mode=real` with two explicit WAV files and expected reply text;
  - `events.jsonl`, `summary.json`, `history_smoke_report.json`.
- `scripts/local_demo_client.py`
  - WAV-based stream client;
  - playback-side cancel with `--cancel-after-ms`;
  - Task 18 playback sync metrics.
- API endpoints:
  - `GET /health`;
  - `GET /diagnostics`;
  - `GET /pipeline/latest`;
  - `GET /ue5/latest`.
- Optional grey-head renderer:
  - `scripts/render_emotalk_grey_head.py`.

Task 19 should reuse these pieces rather than duplicating their protocol loops.

## Proposed approaches

### Approach A: CLI wrapper only

`run_demo_acceptance.py` shells out to existing scripts and aggregates their JSON files.

Pros:

- quickest to implement;
- close to how a human runs the demo.

Cons:

- harder to unit test without subprocess monkeypatching;
- harder to share report-building logic.

### Approach B: small library plus thin scripts

Create a reusable acceptance report module under `src/bionic_head/client/demo_acceptance.py` and a focused artifact collector module under `src/bionic_head/client/demo_artifacts.py`. The CLI scripts call these modules.

Pros:

- clean unit tests for report building and artifact collection;
- scripts stay thin;
- easier to add UE5/AEC checks later.

Cons:

- one extra module.

### Approach C: integrate into existing benchmark script

Extend `scripts/benchmark.py` to run demo acceptance.

Pros:

- one fewer top-level script.

Cons:

- mixes latency benchmarking with product acceptance;
- acceptance has different pass/fail semantics and artifacts.

## Decision

Use Approach B.

The acceptance package gets focused modules and two scripts:

- `src/bionic_head/client/demo_acceptance.py`
- `src/bionic_head/client/demo_artifacts.py`
- `scripts/run_demo_acceptance.py`
- `scripts/collect_demo_artifacts.py`

This keeps Task 19 focused and testable while preserving the existing smoke scripts as the source of truth for WebSocket behavior.

## Architecture

### `DemoAcceptanceRunner`

Responsibilities:

1. Check server readiness via HTTP:
   - `/health`
   - `/diagnostics`
2. Run checks into separate subdirectories:
   - `scripted_interactive_smoke/`
   - `history_smoke/`
   - `playback_interrupt_smoke/`
   - `av_sync_immediate_audio/`
   - `av_sync_wait_for_face/`
3. Read each check's report files.
4. Build a single `demo_acceptance_report.json`.
5. Return non-zero exit status if the final report fails.

The runner can call existing Python functions directly where possible:

- `scripts.interactive_demo_client.run_scripted_demo`
- `scripts.history_smoke.run_history_smoke`
- `scripts.local_demo_client.run_local_demo`

Direct function calls keep tests fast and avoid fragile subprocess parsing.

### `DemoArtifactCollector`

Implemented in `src/bionic_head/client/demo_artifacts.py`.

Responsibilities:

1. Copy known output files into a stable `artifacts/` tree.
2. Resolve optional latest files from HTTP:
   - `/pipeline/latest`
   - `/ue5/latest`
3. Optionally copy local `data/latest/latest_pipeline.json` and `data/latest/latest_ue5_blendshape.json` if present.
4. Optionally render a grey-head video when given:
   - face `.npy` path;
   - audio `.wav` path;
   - renderer dependencies.

The first implementation should make grey-head rendering opt-in.

### `DemoAcceptanceReport`

The report is a plain JSON object:

```json
{
  "success": true,
  "generated_at": "2026-06-29T00:00:00Z",
  "mode": "mock",
  "server": {
    "health_ok": true,
    "diagnostics_ok": true
  },
  "checks": {
    "scripted_interactive_smoke": {
      "success": true,
      "failure_reasons": [],
      "artifacts": {}
    }
  },
  "artifacts": {},
  "failure_reasons": []
}
```

Every check has:

- `success: bool`
- `failure_reasons: list[str]`
- `artifacts: dict[str, str]`

Failure reason examples:

- `server_health_unreachable`
- `server_diagnostics_failed`
- `scripted_interactive_failed`
- `history_smoke_failed`
- `playback_interrupt_terminal_unexpected`
- `av_sync_wait_for_face_timeout`
- `artifact_missing`

## Modes

### `fake`

Default mode. It must not require:

- real microphone;
- real speaker;
- GPU;
- Ollama;
- Piper;
- EmoTalk;
- real WAV files.

It assumes the server is already running with a configuration that can respond to the stream protocol, typically `config/mock.json`. Internally, the history smoke script still uses its existing `mode="mock"` name; the acceptance runner maps external `--mode fake` to history-smoke `mock`.

### `real`

Real provider acceptance. It may require:

- real provider server configuration;
- two history WAV files;
- expected text such as `小张`.

In real mode, missing `--history-turn1-wav` or `--history-turn2-wav` is a CLI error.

## Command shape

Mock mode:

```bash
.venv/bin/python scripts/run_demo_acceptance.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --http-base-url http://127.0.0.1:8005 \
  --output-dir /tmp/bionic-demo-acceptance \
  --mode fake \
  --audio-backend null \
  --playback-sync immediate_audio wait_for_face
```

Real mode:

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

## Checks

### Server checks

`GET /health` passes when the response is JSON and contains `status: ok`.

`GET /diagnostics` passes when the response is JSON. The first version should not require every provider to be `ok`, because mock/real provider environments differ. It should include the diagnostics payload and mark `diagnostics_ok=true` when the endpoint responds with HTTP 200.

### Scripted interactive smoke

Run:

```text
mode=scripted
scripted_turns=2
scripted_cancel_after_ms=300
mic_backend=fake
audio_backend=null
playback_sync=immediate_audio
```

Pass when `interaction_report.json` has:

- `success=true`
- `turn_count=2`
- `completed_turn_count=1`
- `cancelled_turn_count=1`
- `old_generation_audio_play_count=0`
- `old_generation_face_display_count=0`

### History smoke

Fake mode uses `scripts.history_smoke.run_history_smoke(..., mode="mock", turn1_wav=None, turn2_wav=None)`.

Real mode uses explicit `--history-turn1-wav`, `--history-turn2-wav`, and `--expect`.

Pass when `history_smoke_report.json` has `success=true`.

### Playback interrupt smoke

Generate a short local demo WAV under the output directory and run `scripts.local_demo_client.run_local_demo` with:

```text
cancel_after_ms=300
play_audio=false
playback_sync=immediate_audio
```

Pass when:

- terminal event is `server.turn.cancelled`, or
- summary contains `playback_stop_count >= 1` and `client_interrupt_sent_ms != null`.

The check should report a warning if the server finishes too quickly for cancel, but it should fail only when neither cancellation nor playback stop evidence exists.

### AV sync checks

Run scripted mode once per requested strategy:

- `immediate_audio`
- `wait_for_face`

Pass when:

- `interaction_report.json.success=true`;
- `playback_sync_strategy` equals the requested strategy;
- `client_audio_face_offset_ms` is present.

For `wait_for_face`, additionally pass when:

- `client_audio_wait_for_face_ms` is present;
- `client_audio_wait_for_face_timeout` is not `true` unless the caller explicitly allows timeout.

## Artifact layout

```text
output-dir/
  demo_acceptance_report.json
  generated-input.wav
  scripted_interactive_smoke/
    summary.json
    client_playback_metrics.json
    interaction_report.json
    tts/
    ue5/
  history_smoke/
    events.jsonl
    summary.json
    history_smoke_report.json
    tts/
    ue5/
  playback_interrupt_smoke/
    summary.json
    client_playback_metrics.json
    tts/
    ue5/
  av_sync_immediate_audio/
    summary.json
    interaction_report.json
  av_sync_wait_for_face/
    summary.json
    interaction_report.json
  artifacts/
    latest_pipeline.json
    latest_ue5_blendshape.json
    grey_head.mp4
```

Missing optional artifacts should be recorded as warnings, not hard failures, unless the caller asked for them.

## Error handling

Each check should catch exceptions and convert them to:

```json
{
  "success": false,
  "failure_reasons": ["history_smoke_exception"],
  "error_message": "...",
  "artifacts": {}
}
```

The top-level report succeeds only when:

- health check succeeds;
- diagnostics endpoint responds;
- all required checks succeed.

## Testing strategy

Default tests must not require real services, devices, GPU, Ollama, Piper, or EmoTalk.

Unit tests should cover:

- report aggregation;
- failed check propagation;
- artifact collection path mapping;
- CLI argument validation;
- fake-mode acceptance runner with monkeypatched smoke functions and fake HTTP responses;
- real-mode argument validation.

Integration with an actual running server remains a documented command, not a default test.

## Non-goals

Task 19 does not:

- change `/pipeline/stream`;
- change `/pipeline/audio`;
- change WebSocket events;
- connect real UE5;
- add AEC;
- add partial ASR;
- add new model inference;
- require Blender rendering by default.
