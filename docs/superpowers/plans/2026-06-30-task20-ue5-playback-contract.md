# UE5 Playback Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define and validate the UE5 playback contract for `server.ue5.frames`, `generation_epoch` stale-drop, and `server.playback.stop`.

**Architecture:** Add documentation for playback semantics, a small pure-Python validator/state model, JSON fixtures, and thin CLI tools. The contract layers on top of the existing `bionic-head-ue5-v1` Morpheus 52-channel format and does not change the stream wire protocol.

**Tech Stack:** Python 3.10+/3.11-compatible stdlib, pytest, existing stream/UE5 JSON format, Markdown protocol docs.

## Global Constraints

- Do not implement real UE5 Blueprints, Live Link, MetaHuman, WebRTC, AEC, or provider/model changes.
- Do not change ASR/TTS/LLM/EmoTalk behavior.
- Do not rename the existing wire field `start_frame_index`; Task 20 standardizes on this existing field.
- Default audio ownership is `external_audio_clock`; `ue5_audio_owner` is reserved only.
- Fixtures and scripts must run without GPU, Ollama, Piper, EmoTalk, UE5, microphone, or speaker.

---

## Task 1: Design and protocol documentation

**Files:**

- Create: `docs/superpowers/specs/2026-06-30-task20-ue5-playback-contract-design.md`
- Create: `docs/superpowers/plans/2026-06-30-task20-ue5-playback-contract.md`
- Create: `docs/protocols/bionic-head-ue5-playback-v1.md`
- Modify: `docs/protocols/bionic-head-ue5-v1.md`

**Interfaces:**

- Consumes existing `server.ue5.frames` payload shape from `src/bionic_head/core/ue5.py`.
- Produces stable documentation for later validator and UE5 receiver work.

- [ ] Write the design and plan documents.
- [ ] Write `docs/protocols/bionic-head-ue5-playback-v1.md` with required fields, stale drop, playback stop, ordering, audio ownership, reconnect, timeouts, and metrics.
- [ ] Update `docs/protocols/bionic-head-ue5-v1.md` to link to the playback contract.
- [ ] Run `rg -n "start_frame_index|generation_epoch|external_audio_clock|server.playback.stop" docs/protocols docs/superpowers/specs/2026-06-30-task20-ue5-playback-contract-design.md`.
- [ ] Commit with `docs: define ue5 playback contract`.

## Task 2: Contract model and validator

**Files:**

- Create: `src/bionic_head/ue5_playback_contract.py`
- Create: `tests/unit/test_ue5_playback_contract.py`

**Interfaces:**

- Produces:
  - `UE5PlaybackContractError(ValueError)`
  - `validate_ue5_frame_chunk(payload: Mapping[str, object]) -> dict[str, object]`
  - `validate_playback_stop(payload: Mapping[str, object]) -> dict[str, object]`

- [ ] Write failing tests for valid payload, invalid `format`, invalid `channel_count`, invalid `fps`, invalid `generation_epoch`, invalid `start_frame_index`, invalid `pts_start_ms`, wrong `frame_count`, and wrong frame weight length.
- [ ] Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_ue5_playback_contract.py -q` and verify RED.
- [ ] Implement minimal validation code in `src/bionic_head/ue5_playback_contract.py`.
- [ ] Run the same pytest command and verify GREEN.
- [ ] Commit with `feat: add ue5 playback contract validator`.

## Task 3: Receiver-state replay model

**Files:**

- Modify: `src/bionic_head/ue5_playback_contract.py`
- Modify: `tests/unit/test_ue5_playback_contract.py`

**Interfaces:**

- Produces:
  - `UE5PlaybackReceiverState`
  - `UE5PlaybackAction`
  - `replay_ue5_events(events: Iterable[Mapping[str, object]]) -> dict[str, object]`

- [ ] Write failing tests that prove stale frames are dropped after a newer epoch, `server.playback.stop` clears buffers and advances epoch, duplicate chunks are not counted twice, and valid ordered chunks increase buffered frame count.
- [ ] Run focused pytest and verify RED.
- [ ] Implement the state model.
- [ ] Run focused pytest and verify GREEN.
- [ ] Commit with `feat: add ue5 playback replay state`.

## Task 4: Fixtures and CLI tools

**Files:**

- Create: `tests/fixtures/ue5_playback/valid_segment.json`
- Create: `tests/fixtures/ue5_playback/stale_generation.json`
- Create: `tests/fixtures/ue5_playback/playback_stop.json`
- Create: `tests/fixtures/ue5_playback/invalid_channel_count.json`
- Create: `tests/fixtures/ue5_playback/invalid_frame_length.json`
- Create: `scripts/validate_ue5_playback_contract.py`
- Create: `scripts/replay_ue5_frames.py`
- Modify: `tests/unit/test_ue5_playback_contract.py`

**Interfaces:**

- Consumes validator and replay functions from `bionic_head.ue5_playback_contract`.
- Produces CLI entry points with `--help`.

- [ ] Write failing tests that validate fixtures and run both CLI `--help` commands by path.
- [ ] Run focused pytest and verify RED.
- [ ] Add fixtures and scripts.
- [ ] Run focused pytest and verify GREEN.
- [ ] Commit with `feat: add ue5 playback contract fixtures`.

## Task 5: Operations documentation and final verification

**Files:**

- Create: `docs/operations/ue5-playback-contract.md`
- Modify: `tests/unit/test_ue5_playback_contract.py` only if a help/doc smoke assertion is needed.

**Interfaces:**

- Consumes contract docs and CLI tools.
- Produces receiver-facing runbook for Task 21.

- [ ] Document what UE5 should subscribe to, how to buffer frames, how to play by `pts_start_ms`/`fps`, how to handle `server.playback.stop`, how to stale-drop old generations, how audio ownership works, and how to validate fixtures.
- [ ] Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_ue5_playback_contract.py -q`.
- [ ] Run `PYTHONPATH=src .venv/bin/python -m pytest -q`.
- [ ] Commit with `docs: document ue5 playback receiver behavior`.

## Completion checklist

- [ ] Full pytest passes.
- [ ] UE5 playback contract docs exist.
- [ ] Required `server.ue5.frames` fields are documented.
- [ ] `generation_epoch` stale-drop is documented and tested.
- [ ] `server.playback.stop` buffer clear is documented and tested.
- [ ] Default `external_audio_clock` ownership is documented.
- [ ] Validator accepts valid payloads and rejects invalid payloads.
- [ ] Fixtures cover valid, stale, stop, invalid channel count, and invalid frame length.
- [ ] Replay/validate CLI help works.
- [ ] No real UE5 integration, provider changes, or model changes were introduced.
