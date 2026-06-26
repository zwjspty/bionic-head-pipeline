# Task 15 Scripted Interactive Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use TDD. Write each behavior test first, verify it fails for the expected reason, then implement the smallest code to pass.

**Goal:** Add a scripted mode to `interactive_demo_client.py` so one command can automatically run a fake-mic interaction smoke: first turn plays then cancels, playback stop clears local buffers, second turn continues, and `interaction_report.json` records acceptance metrics.

## Constraints

- Do not add real microphone automated tests.
- Do not add AEC.
- Do not add WebRTC.
- Do not add browser UI.
- Do not add true UE5 runtime integration.
- Do not add ASR partial.
- Do not add real-time Blender.
- Default tests must not require microphone, speaker, GPU, Ollama, Piper, EmoTalk, or a running server.

## Files

- Create: `src/bionic_head/client/scripted.py`
- Create: `tests/unit/test_scripted_interactive_client.py`
- Modify: `scripts/interactive_demo_client.py`
- Create: `docs/operations/interactive-demo-client.md`

## Task 15.1: Scripted command plan

- [ ] Write failing tests for a deterministic scripted command plan that produces:

```text
start_recording
stop_recording
wait_for_playback
cancel
wait_for_terminal_cancel
start_recording
stop_recording
wait_for_done
quit
```

- [ ] Implement `ScriptedAction`, `ScriptedController`, or equivalent in `src/bionic_head/client/scripted.py`.
- [ ] Verify focused tests pass.

## Task 15.2: Scripted fake mic turns

- [ ] Write failing tests that run `interactive_demo_client` in scripted mode with fake websocket/fake mic and verify:

```text
client.session.start
turn 1 client.audio.start/chunk/binary/end
client.turn.cancel
turn 2 client.audio.start/chunk/binary/end
```

- [ ] Implement `--mode scripted`, `--scripted-turns`, and `--scripted-cancel-after-ms`.
- [ ] Reuse `FakeMicBackend`.
- [ ] Verify focused tests pass.

## Task 15.3: Scripted cancel after playback

- [ ] Write failing tests where the fake server sends `server.tts.audio` + binary before `server.playback.stop`.
- [ ] Verify cancel is sent only after local audio playback begins.
- [ ] Verify `server.playback.stop` calls audio stop and face clear through `LocalDemoReceiver`.
- [ ] Verify stale audio/face drop counts remain zero unless the fake server intentionally sends stale events.

## Task 15.4: Interaction report

- [ ] Write failing tests for `interaction_report.json`.
- [ ] Include:

```text
success
mode
turn_count
completed_turn_count
cancelled_turn_count
playback_stop_count
old_generation_audio_play_count
old_generation_face_display_count
client_stale_audio_drop_count
client_stale_face_drop_count
client_interrupt_sent_ms
server_playback_stop_received_ms
client_audio_stopped_ms
client_face_buffer_cleared_ms
client_interrupt_to_playback_stop_ms
client_interrupt_to_audio_stop_ms
client_interrupt_to_face_clear_ms
```

- [ ] Verify report is written for scripted mode.

## Task 15.5: Docs and final verification

- [ ] Create `docs/operations/interactive-demo-client.md`.
- [ ] Document:
  - real interactive microphone mode;
  - scripted fake/null smoke mode;
  - no AEC / wear headphones warning;
  - expected `interaction_report.json`.
- [ ] Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_scripted_interactive_client.py tests/unit/test_interactive_demo_client.py -q
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python scripts/interactive_demo_client.py --help
```

- [ ] Commit in logical chunks.
