# Task 6 RMS Barge-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement dependency-free RMS VAD barge-in for `/pipeline/stream`.

**Architecture:** `StreamConnection` owns a pending interrupt candidate while the old turn remains active. Candidate PCM is buffered until RMS speech duration crosses the configured threshold, then Task 5 cancellation/epoch logic activates the new turn.

**Tech Stack:** Python 3.11-compatible code, FastAPI WebSocket handling, Pydantic v2 config, pytest.

## Global Constraints

- Do not add Silero or other model dependencies.
- Do not change the `bionic-head-stream-v1` protocol name.
- Keep mock provider tests deterministic.
- Preserve existing `client.turn.cancel` behavior.
- Use TDD for behavior changes.

---

### Task 6.1: VAD configuration

**Files:**
- Modify: `src/bionic_head/config.py`
- Modify: `config/mock.json`
- Modify: `config/real.example.json`
- Modify: `config/emotalk.example.json`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `AppSettings.vad`
- Produces: `VadSettings(engine="rms", interrupt_min_speech_ms=80, interrupt_rms_threshold=0.02)`

- [ ] **Step 1: Write failing config tests**

Assert all example configs load with `settings.vad.engine == "rms"`.

- [ ] **Step 2: Run failing tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_config.py -q`

- [ ] **Step 3: Implement settings and config files**

Add the Pydantic settings and JSON config sections.

- [ ] **Step 4: Verify and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -m 'not integration' -q`

Commit: `feat: add rms vad settings`

### Task 6.2: Interrupt candidate state machine

**Files:**
- Modify: `src/bionic_head/protocol/connection.py`
- Test: `tests/unit/test_stream_connection.py`

**Interfaces:**
- Produces internal dataclass: `PendingInterrupt(turn_id, pcm_buffer, speech_ms, turn_started_monotonic, last_non_silent_monotonic)`
- Consumes: `settings.vad.interrupt_min_speech_ms`
- Consumes: `settings.vad.interrupt_rms_threshold`

- [ ] **Step 1: Write failing WebSocket tests**

Add tests that new `client.audio.start` alone does not emit `server.playback.stop`, high-RMS chunks trigger it after threshold, and low-RMS chunks do not.

- [ ] **Step 2: Run failing tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_connection.py -q`

- [ ] **Step 3: Implement interrupt candidate flow**

Change `_start_turn`, `_validate_client_event`, `_handle_audio_chunk_metadata`, and `_handle_binary` so candidate audio is buffered until accepted.

- [ ] **Step 4: Verify and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -m 'not integration' -q`

Commit: `feat: add rms barge in interrupt`

## Self-review

- The plan deliberately avoids Silero and AEC.
- The plan reuses Task 5 epoch and playback stop semantics.
- The plan includes deterministic config and WebSocket behavior tests.
