# Task 16.5: Multi-Turn History Smoke Design

## Summary

Task 16 added session-level conversation history to `/pipeline/stream`. Task 16.5 verifies that behavior through a repeatable smoke path:

```text
turn 1: 我叫小张。
turn 2: 我叫什么？
```

The second turn must run in the same WebSocket session and prove that history from the first successful turn is available before the second LLM call.

The recommended scope is dual-mode:

```text
default automation: mock provider, deterministic and hardware-free
manual acceptance: real provider, real ASR/Ollama/TTS/EmoTalk when configured
```

## Goals

- Exercise two successful turns in the same `/pipeline/stream` WebSocket session.
- Produce a compact `history_smoke_report.json`.
- In mock automation, prove that second-turn history metrics are non-empty and that the second reply contains the expected name.
- In real acceptance, support user-provided WAV files for:
  - `turn1`: “我叫小张。”
  - `turn2`: “我叫什么？”
- Keep the test independent from microphone, speaker, GPU, Ollama, Piper, and EmoTalk unless explicitly running real mode.

## Non-Goals

- No long-term memory.
- No database, RAG, vector store, or user profile.
- No `/pipeline/audio` cross-request history.
- No ASR, TTS, EmoTalk, UE5, or protocol behavior changes.
- No browser UI, WebRTC, AEC, or real UE5 integration.

## Approach

Add a small smoke runner:

```text
scripts/history_smoke.py
```

It connects to a running WebSocket server, sends two turns through one session, waits for each turn to finish, and writes:

```text
history_smoke_report.json
summary.json
events.jsonl
tts/*.wav
ue5/*.json
```

The runner should reuse existing client protocol helpers where practical, but it should stay focused on history acceptance instead of becoming another full interactive client.

## CLI

Recommended command for deterministic mock acceptance:

```bash
PYTHONPATH=src .venv/bin/python scripts/history_smoke.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --output-dir data/demo/history-smoke-mock \
  --mode mock \
  --expect 小张
```

Recommended command for real acceptance:

```bash
PYTHONPATH=src .venv/bin/python scripts/history_smoke.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --output-dir data/demo/history-smoke-real \
  --mode real \
  --turn1-wav /path/to/wo-jiao-xiaozhang.wav \
  --turn2-wav /path/to/wo-jiao-shenme.wav \
  --expect 小张
```

## Mock Mode Behavior

Mock mode should avoid real ASR dependency while still exercising real stream session history. The server can run with a dedicated mock config where:

```text
turn1 ASR text = 我叫小张。
turn2 ASR text = 我叫什么？
```

Because the current mock ASR config has one fixed `asr_text`, Task 16.5 should not mutate core ASR behavior just to support smoke testing. The least invasive design is:

```text
1. use a tiny test-only/mock-history LLM behavior in the smoke server config or test harness; or
2. add a focused unit/integration test around StreamOrchestrator for exact two-turn history semantics; and
3. let scripts/history_smoke.py support real WAVs for end-user acceptance.
```

For implementation, prefer a small reusable `HistorySmokeResult` parser and a unit test with fake WebSocket/server events. If a live mock server flow needs per-turn ASR text later, that should be a separate mock-provider enhancement, not hidden inside Task 16.5.

## Real Mode Behavior

Real mode assumes the server is already running with a real or hybrid config. The smoke runner sends provided WAV files:

```text
client.session.start
turn 1 client.audio.start/chunk/end
wait server.pipeline.done
turn 2 client.audio.start/chunk/end
wait server.pipeline.done
```

Acceptance checks:

```text
turn1 terminal_event == server.pipeline.done
turn2 terminal_event == server.pipeline.done
turn2 reply contains expected text, e.g. 小张
turn2 history_turn_count_before > 0
turn2 history_enabled == true
```

If ASR misrecognizes the real WAV, the report should show actual ASR text and fail clearly rather than pretending history failed.

## Report Schema

`history_smoke_report.json` should contain at least:

```json
{
  "success": true,
  "mode": "real",
  "session_id": "00000000-0000-0000-0000-000000000001",
  "turns": [
    {
      "turn_index": 1,
      "turn_id": "00000000-0000-0000-0000-000000000011",
      "terminal_event": "server.pipeline.done",
      "asr_text": "我叫小张。",
      "llm_reply": "你好小张。",
      "history_turn_count_before": 0,
      "history_turn_count_after": 1
    },
    {
      "turn_index": 2,
      "turn_id": "00000000-0000-0000-0000-000000000012",
      "terminal_event": "server.pipeline.done",
      "asr_text": "我叫什么？",
      "llm_reply": "你叫小张。",
      "history_turn_count_before": 1,
      "history_turn_count_after": 2,
      "reply_contains_expected": true
    }
  ],
  "expected_text": "小张"
}
```

## Testing

Default tests must not require a live server or real providers.

Test layers:

1. Unit test report builder:
   - success when turn 2 has history before count > 0 and reply contains expected text;
   - failure when reply misses expected text;
   - failure when turn 2 history is empty;
   - clear reason when a turn terminal event is not `server.pipeline.done`.
2. Unit test script CLI argument parsing.
3. Optional/manual smoke command against a running server.
4. Full pytest must remain hardware-free by default.

## Error Handling

The report should include failure reasons such as:

```text
turn1_not_done
turn2_not_done
turn2_history_empty
expected_text_missing
websocket_error
protocol_error
missing_binary_after_tts_event
```

The script should exit non-zero when `success` is false.

## Acceptance Criteria

- `scripts/history_smoke.py --help` runs.
- Default unit tests do not require microphone, speaker, GPU, Ollama, Piper, EmoTalk, or a live server.
- The script can drive two turns in one WebSocket session.
- The report records ASR text, LLM reply, terminal event, and history metrics for each turn.
- Real mode can verify “我叫小张 / 我叫什么？” when provided suitable WAV files and a real server.
- Existing Task 12–16 tests keep passing.
