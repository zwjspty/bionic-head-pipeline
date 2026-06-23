# Task 5 Interrupt Epoch and Stale-Drop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add minimal interruption-safe WebSocket semantics: `generation_epoch`, `server.playback.stop`, and stale output suppression.

**Architecture:** `StreamConnection` owns the session epoch and creates `TurnHandle` instances with the current epoch. `EventFactory` stamps every server event with that epoch. `TurnHandle` checks an injected epoch getter before emitting or committing.

**Tech Stack:** Python 3.11-compatible code, FastAPI WebSocket stack, Pydantic v2, pytest, asyncio.

## Global Constraints

- Keep the existing `bionic-head-stream-v1` protocol name.
- Do not implement real VAD in Task 5.
- Do not break `/pipeline/stream` happy path.
- Preserve existing mock and real provider behavior.
- Use TDD: write failing tests before implementation.

---

### Task 5.1: Server event epoch and playback stop

**Files:**
- Modify: `src/bionic_head/protocol/events.py`
- Modify: `src/bionic_head/core/state.py`
- Modify: `src/bionic_head/protocol/connection.py`
- Test: `tests/unit/test_state.py`
- Test: `tests/unit/test_stream_connection.py`

**Interfaces:**
- Produces: `EventFactory(session_id: UUID, generation_epoch_getter: Callable[[], int] | None = None)`
- Produces: `TurnHandle(..., generation_epoch: int, generation_epoch_getter: Callable[[], int] | None = None)`
- Produces event: `EventType.SERVER_PLAYBACK_STOP = "server.playback.stop"`

- [ ] **Step 1: Write failing tests**

Add tests that assert server envelopes include `generation_epoch`, epoch changes suppress old turn emits, and cancel emits `server.playback.stop` before `server.turn.cancelled`.

- [ ] **Step 2: Run tests to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_state.py tests/unit/test_stream_connection.py -q`

- [ ] **Step 3: Implement event/epoch plumbing**

Add the enum, optional epoch getter, payload stamping, and `TurnHandle.is_current()` epoch comparison.

- [ ] **Step 4: Implement playback stop emission**

In `_cancel_current_turn(emit=True)`, increment the epoch and send `server.playback.stop` before `server.turn.cancelled`.

- [ ] **Step 5: Verify and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -m 'not integration' -q`

Commit: `feat: add interrupt epoch playback stop`

### Task 5.2: Client stale-buffer clearing and metrics

**Files:**
- Modify: `scripts/stream_client.py`
- Modify: `scripts/benchmark.py`
- Test: `tests/unit/test_stream_client.py`
- Test: `tests/unit/test_benchmark.py`

**Interfaces:**
- Consumes: `server.playback.stop`
- Produces summary fields: `playback_stop_count`, `latest_generation_epoch`

- [ ] **Step 1: Write failing tests**

Add tests that `ClientReceiver` clears pending playback on `server.playback.stop`, tracks the latest epoch, and benchmark extracts `interrupt_to_playback_stop_ms` when present.

- [ ] **Step 2: Run tests to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_client.py tests/unit/test_benchmark.py -q`

- [ ] **Step 3: Implement client handling**

Clear pending TTS, segment, UE5 buffers; count playback stops; track the largest `generation_epoch` seen.

- [ ] **Step 4: Implement benchmark extraction**

Map `event_first_ms["server.playback.stop"]` to `interrupt_to_playback_stop_ms`.

- [ ] **Step 5: Verify and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -m 'not integration' -q`

Commit: `feat: track playback stop in stream client`

## Self-review

- The plan does not implement real VAD; that remains Task 6.
- The plan keeps existing protocol name and adds fields/events compatibly.
- Each task has a focused test cycle and commit.
