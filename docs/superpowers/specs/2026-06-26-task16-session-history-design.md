# Task 16: Session-Level Conversation History Design

## Summary

Task 16 adds short-term conversation memory for `/pipeline/stream`.

The goal is:

```text
turn 1 user: 我叫小张。
turn 1 assistant: 你好小张。
turn 2 user: 我叫什么？
turn 2 LLM call receives turn 1 user/assistant history.
turn 2 assistant can answer: 你叫小张。
```

This is session-local memory only. It is not long-term memory, RAG, a user profile, a vector store, or a database-backed feature.

## Scope

### In scope

- Add `ConversationHistoryStore`.
- Store history by WebSocket `session_id`.
- Pass current session history into stream LLM calls.
- Commit `user + assistant` only after a stream turn succeeds.
- Do not commit cancelled, stale, or error turns.
- Enforce `max_turn_pairs` and `max_chars`.
- Record history metrics in stream timeline.
- Keep existing cancel / stale-drop / scripted smoke behavior intact.

### Out of scope

- `/pipeline/audio` cross-request history.
- Long-term memory.
- Database persistence.
- User profiles.
- RAG or vector retrieval.
- Memory extraction / summarization.
- ASR, TTS, EmoTalk, UE5, AEC, WebRTC changes.

## Architecture

### Store

Add:

```text
src/bionic_head/core/history.py
```

Core types:

```python
@dataclass(frozen=True)
class ConversationTurn:
    role: Literal["user", "assistant"]
    content: str

class ConversationHistoryStore:
    def get(self, session_id: UUID) -> list[dict[str, str]]: ...
    def append_pair(self, session_id: UUID, *, user: str, assistant: str) -> None: ...
    def metrics(self, session_id: UUID) -> ConversationHistoryMetrics: ...
```

The store is in-memory and owned by `AppContainer`. It is intentionally process-local.

### Config

Add:

```json
"history": {
  "enabled": true,
  "max_turn_pairs": 6,
  "max_chars": 3000
}
```

If disabled, stream passes an empty history and does not append.

### Stream integration

`AppContainer` owns a single `ConversationHistoryStore` and passes it into `StreamOrchestrator`.

In `StreamOrchestrator.run()`:

```text
after ASR final:
  history_snapshot = history_store.get(turn.session_id)

LLM:
  registry.llm.chat_stream(asr.text, history_snapshot, context)

after successful pipeline.done path:
  history_store.append_pair(session_id, user=asr.text, assistant=llm.reply)
```

Cancelled, stale, and error paths never append.

## Timeline metrics

Add these to `timeline["stream"]`:

```text
history_enabled
history_turn_count_before
history_char_count_before
history_turn_count_after
history_char_count_after
```

The metrics should be present even when history is disabled; disabled mode reports zero counts.

## Testing

Required tests:

- store append/get;
- max turn pair trimming;
- max char trimming;
- disabled config returns empty/no-op;
- stream second turn LLM receives first turn history;
- cancelled turn does not append;
- provider error turn does not append;
- existing scripted smoke still passes.

Default tests must not require a real microphone, speaker, GPU, Ollama, Piper, EmoTalk, or running FastAPI server.

## Acceptance criteria

Task 16 is complete when:

1. full pytest passes;
2. same WebSocket session second turn LLM receives first turn history;
3. only successful stream turns commit user/assistant pairs;
4. cancelled/error/stale turns do not commit assistant replies;
5. history is trimmed by `max_turn_pairs` and `max_chars`;
6. timeline exposes history metrics;
7. scripted interactive smoke still passes.
