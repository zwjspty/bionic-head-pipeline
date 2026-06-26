# Task 16 Session History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use TDD. Write each behavior test first, verify it fails for the expected reason, then implement the smallest code to pass.

**Goal:** Add session-level conversation history to `/pipeline/stream` so later turns in the same WebSocket session can use earlier successful user/assistant turns.

**Architecture:** Add an in-memory `ConversationHistoryStore` owned by `AppContainer`. `StreamOrchestrator` reads a session history snapshot before calling LLM, passes it to `chat_stream`, and appends user/assistant only after successful completion.

**Tech Stack:** Python 3.10, dataclasses, Pydantic config, pytest, FastAPI WebSocket test client.

## Global Constraints

- Do not add long-term memory.
- Do not add database persistence.
- Do not add RAG or vector retrieval.
- Do not change ASR / TTS / EmoTalk / UE5 providers.
- Do not change AEC / WebRTC.
- Default tests must not require microphone, speaker, GPU, Ollama, Piper, EmoTalk, or a running server.

---

## File Structure

- Create `src/bionic_head/core/history.py`
  - `ConversationTurn`
  - `ConversationHistoryMetrics`
  - `ConversationHistoryStore`
- Modify `src/bionic_head/config.py`
  - `HistorySettings`
  - `AppSettings.history`
- Modify `src/bionic_head/api/dependencies.py`
  - `AppContainer.history`
  - pass history into `StreamOrchestrator`
- Modify `src/bionic_head/orchestrators/stream.py`
  - read history before LLM
  - pass history into `chat_stream`
  - append successful turns
  - write timeline metrics
- Add `tests/unit/test_history.py`
- Modify `tests/unit/test_stream_orchestrator.py`
- Modify `tests/unit/test_stream_connection.py` or add focused WebSocket history test if needed.
- Update docs/status or operations docs after implementation.

---

## Task 16.1: ConversationHistoryStore

**Files:**
- Create: `src/bionic_head/core/history.py`
- Create: `tests/unit/test_history.py`

**Interfaces:**
- Produces:
  - `ConversationTurn(role: Literal["user", "assistant"], content: str)`
  - `ConversationHistoryMetrics(turn_count: int, char_count: int)`
  - `ConversationHistoryStore(enabled: bool, max_turn_pairs: int, max_chars: int)`
  - `get(session_id: UUID) -> list[dict[str, str]]`
  - `append_pair(session_id: UUID, *, user: str, assistant: str) -> None`
  - `metrics(session_id: UUID) -> ConversationHistoryMetrics`

- [ ] Write failing tests for append/get.
- [ ] Write failing tests for `max_turn_pairs`.
- [ ] Write failing tests for `max_chars`.
- [ ] Write failing tests for disabled no-op behavior.
- [ ] Implement store.
- [ ] Run `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_history.py -q`.
- [ ] Commit `feat: add conversation history store`.

---

## Task 16.2: Config and container ownership

**Files:**
- Modify: `src/bionic_head/config.py`
- Modify: `src/bionic_head/api/dependencies.py`
- Modify: `tests/unit/test_config.py`
- Add/modify: `tests/unit/test_app_container.py`

**Interfaces:**
- Consumes `ConversationHistoryStore`.
- Produces `AppContainer.history`.

- [ ] Write failing config test for defaults:
  - enabled true
  - max_turn_pairs 6
  - max_chars 3000
- [ ] Write failing container test that `AppContainer.create()` owns a history store.
- [ ] Implement `HistorySettings`.
- [ ] Instantiate `ConversationHistoryStore` in `AppContainer.create()`.
- [ ] Pass history into `StreamOrchestrator`.
- [ ] Run focused config/container tests.
- [ ] Commit `feat: configure session history store`.

---

## Task 16.3: Pass history into stream LLM and commit successful turns

**Files:**
- Modify: `src/bionic_head/orchestrators/stream.py`
- Modify: `tests/unit/test_stream_orchestrator.py`

**Interfaces:**
- Consumes `ConversationHistoryStore`.
- `StreamOrchestrator(..., history: ConversationHistoryStore | None = None)`.

- [ ] Write failing test: history store preloaded with first turn, stream LLM receives that history.
- [ ] Write failing test: successful turn appends `asr.text + llm.reply`.
- [ ] Implement read-before-LLM and append-after-success.
- [ ] Ensure offline orchestrator remains unchanged.
- [ ] Run focused stream orchestrator tests.
- [ ] Commit `feat: pass session history into stream llm`.

---

## Task 16.4: Do not commit cancel/error turns

**Files:**
- Modify: `src/bionic_head/orchestrators/stream.py`
- Modify: `tests/unit/test_stream_orchestrator.py`

- [ ] Write failing test: cancelled turn does not append history.
- [ ] Write failing test: provider error turn does not append history.
- [ ] Implement by only appending on success path after `_ensure_complete()` and before/near `server.pipeline.done`.
- [ ] Run focused stream orchestrator tests.
- [ ] Commit `feat: commit only successful stream history`.

---

## Task 16.5: History timeline metrics

**Files:**
- Modify: `src/bionic_head/orchestrators/stream.py`
- Modify: `tests/unit/test_stream_orchestrator.py`

- [ ] Write failing test that `timeline["stream"]` includes:
  - `history_enabled`
  - `history_turn_count_before`
  - `history_char_count_before`
  - `history_turn_count_after`
  - `history_char_count_after`
- [ ] Implement metrics snapshot.
- [ ] Run focused tests.
- [ ] Commit `feat: record stream history metrics`.

---

## Task 16.6: Scripted smoke regression and docs

**Files:**
- Modify: `tests/unit/test_scripted_interactive_client.py`
- Modify: `docs/status/2026-06-24-current-state.md`
- Optionally modify: `docs/operations/interactive-demo-client.md`

- [ ] Add/verify scripted smoke unit coverage still passes with history enabled.
- [ ] Document Task16 behavior and limitations.
- [ ] Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_history.py tests/unit/test_stream_orchestrator.py tests/unit/test_scripted_interactive_client.py -q
PYTHONPATH=src .venv/bin/python -m pytest -q
```

- [ ] Commit `docs: document session history behavior`.
