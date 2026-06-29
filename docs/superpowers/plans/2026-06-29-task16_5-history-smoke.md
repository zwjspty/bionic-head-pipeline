# Task 16.5 History Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable two-turn `/pipeline/stream` smoke runner that verifies session-level history through a `history_smoke_report.json` artifact.

**Architecture:** Add a focused `bionic_head.client.history_smoke` module for report modeling/evaluation and a thin `scripts/history_smoke.py` CLI for WebSocket driving. Default tests use fake events and fake websocket objects; real provider acceptance is a manual command against a running server with user-provided WAV files.

**Tech Stack:** Python 3.11 target, FastAPI/WebSocket protocol already implemented by the server, `websockets` client runtime, pytest, existing `scripts.stream_client` helpers.

## Global Constraints

- Do not change ASR, TTS, EmoTalk, UE5, or stream server protocol behavior.
- Do not add long-term memory, database, RAG, vector store, or user profile.
- Default tests must not require microphone, speaker, GPU, Ollama, Piper, EmoTalk, or a live server.
- Manual real smoke supports provided mono PCM16 16k WAV files.
- Output report file name is exactly `history_smoke_report.json`.
- Existing Task 12–16 tests must keep passing.

---

## File Structure

- Create `src/bionic_head/client/history_smoke.py`
  - Owns `HistorySmokeTurn`, `HistorySmokeReport`, report evaluation, and JSON writing.
  - Does not import `websockets`; stays pure and unit-testable.
- Create `scripts/history_smoke.py`
  - Owns CLI parsing and live WebSocket two-turn execution.
  - Reuses `scripts.stream_client.client_event`, `read_pcm16_from_wav`, and `pcm_chunks`.
- Create `tests/unit/test_history_smoke.py`
  - Tests report success/failure logic and fake two-turn runner behavior without live server.
- Modify `docs/operations/interactive-demo-client.md`
  - Adds Task 16.5 real smoke command.

---

### Task 1: Add History Smoke Report Model

**Files:**
- Create: `src/bionic_head/client/history_smoke.py`
- Create: `tests/unit/test_history_smoke.py`

**Interfaces:**
- Produces:
  - `HistorySmokeTurn`
  - `HistorySmokeReport`
  - `build_history_smoke_report(*, mode: str, session_id: str, expected_text: str, turns: list[HistorySmokeTurn]) -> HistorySmokeReport`
  - `write_history_smoke_report(path: Path, report: HistorySmokeReport) -> None`

- [ ] **Step 1: Write failing report tests**

Add this to `tests/unit/test_history_smoke.py`:

```python
from pathlib import Path
import json

from bionic_head.client.history_smoke import (
    HistorySmokeTurn,
    build_history_smoke_report,
    write_history_smoke_report,
)


def test_history_smoke_report_succeeds_when_second_turn_uses_history_and_reply_contains_expected() -> None:
    report = build_history_smoke_report(
        mode="mock",
        session_id="session-1",
        expected_text="小张",
        turns=[
            HistorySmokeTurn(
                turn_index=1,
                turn_id="turn-1",
                terminal_event="server.pipeline.done",
                asr_text="我叫小张。",
                llm_reply="你好小张。",
                history_enabled=True,
                history_turn_count_before=0,
                history_turn_count_after=1,
            ),
            HistorySmokeTurn(
                turn_index=2,
                turn_id="turn-2",
                terminal_event="server.pipeline.done",
                asr_text="我叫什么？",
                llm_reply="你叫小张。",
                history_enabled=True,
                history_turn_count_before=1,
                history_turn_count_after=2,
            ),
        ],
    )

    assert report.success is True
    assert report.failure_reasons == []
    assert report.turns[1].reply_contains_expected is True


def test_history_smoke_report_fails_when_second_turn_history_is_empty() -> None:
    report = build_history_smoke_report(
        mode="mock",
        session_id="session-1",
        expected_text="小张",
        turns=[
            HistorySmokeTurn(
                turn_index=1,
                turn_id="turn-1",
                terminal_event="server.pipeline.done",
                asr_text="我叫小张。",
                llm_reply="你好小张。",
                history_enabled=True,
                history_turn_count_before=0,
                history_turn_count_after=1,
            ),
            HistorySmokeTurn(
                turn_index=2,
                turn_id="turn-2",
                terminal_event="server.pipeline.done",
                asr_text="我叫什么？",
                llm_reply="你叫小张。",
                history_enabled=True,
                history_turn_count_before=0,
                history_turn_count_after=1,
            ),
        ],
    )

    assert report.success is False
    assert "turn2_history_empty" in report.failure_reasons


def test_history_smoke_report_fails_when_expected_text_is_missing() -> None:
    report = build_history_smoke_report(
        mode="real",
        session_id="session-1",
        expected_text="小张",
        turns=[
            HistorySmokeTurn(
                turn_index=1,
                turn_id="turn-1",
                terminal_event="server.pipeline.done",
                asr_text="我叫小张。",
                llm_reply="你好。",
                history_enabled=True,
                history_turn_count_before=0,
                history_turn_count_after=1,
            ),
            HistorySmokeTurn(
                turn_index=2,
                turn_id="turn-2",
                terminal_event="server.pipeline.done",
                asr_text="我叫什么？",
                llm_reply="我不知道。",
                history_enabled=True,
                history_turn_count_before=1,
                history_turn_count_after=2,
            ),
        ],
    )

    assert report.success is False
    assert "expected_text_missing" in report.failure_reasons
    assert report.turns[1].reply_contains_expected is False


def test_write_history_smoke_report_writes_json(tmp_path: Path) -> None:
    report = build_history_smoke_report(
        mode="mock",
        session_id="session-1",
        expected_text="小张",
        turns=[
            HistorySmokeTurn(
                turn_index=1,
                turn_id="turn-1",
                terminal_event="server.pipeline.done",
                asr_text="我叫小张。",
                llm_reply="你好小张。",
                history_enabled=True,
                history_turn_count_before=0,
                history_turn_count_after=1,
            ),
            HistorySmokeTurn(
                turn_index=2,
                turn_id="turn-2",
                terminal_event="server.pipeline.done",
                asr_text="我叫什么？",
                llm_reply="你叫小张。",
                history_enabled=True,
                history_turn_count_before=1,
                history_turn_count_after=2,
            ),
        ],
    )

    output = tmp_path / "history_smoke_report.json"
    write_history_smoke_report(output, report)

    body = json.loads(output.read_text(encoding="utf-8"))
    assert body["success"] is True
    assert body["turns"][1]["reply_contains_expected"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_history_smoke.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'bionic_head.client.history_smoke'`.

- [ ] **Step 3: Implement report model**

Create `src/bionic_head/client/history_smoke.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class HistorySmokeTurn:
    turn_index: int
    turn_id: str
    terminal_event: str | None
    asr_text: str | None
    llm_reply: str | None
    history_enabled: bool | None
    history_turn_count_before: int | None
    history_turn_count_after: int | None
    reply_contains_expected: bool = False


@dataclass
class HistorySmokeReport:
    success: bool
    mode: str
    session_id: str
    expected_text: str
    turns: list[HistorySmokeTurn]
    failure_reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "mode": self.mode,
            "session_id": self.session_id,
            "expected_text": self.expected_text,
            "turns": [asdict(turn) for turn in self.turns],
            "failure_reasons": self.failure_reasons,
        }


def build_history_smoke_report(
    *,
    mode: str,
    session_id: str,
    expected_text: str,
    turns: list[HistorySmokeTurn],
) -> HistorySmokeReport:
    failure_reasons: list[str] = []
    if len(turns) != 2:
        failure_reasons.append("expected_two_turns")

    for turn in turns:
        if turn.terminal_event != "server.pipeline.done":
            failure_reasons.append(f"turn{turn.turn_index}_not_done")

    if len(turns) >= 2:
        second = turns[1]
        second.reply_contains_expected = bool(
            expected_text and second.llm_reply and expected_text in second.llm_reply
        )
        if second.history_enabled is not True:
            failure_reasons.append("history_disabled")
        if not second.history_turn_count_before or second.history_turn_count_before < 1:
            failure_reasons.append("turn2_history_empty")
        if not second.reply_contains_expected:
            failure_reasons.append("expected_text_missing")

    return HistorySmokeReport(
        success=not failure_reasons,
        mode=mode,
        session_id=session_id,
        expected_text=expected_text,
        turns=turns,
        failure_reasons=failure_reasons,
    )


def write_history_smoke_report(path: Path, report: HistorySmokeReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_history_smoke.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bionic_head/client/history_smoke.py tests/unit/test_history_smoke.py
git commit -m "feat: add history smoke report builder"
```

---

### Task 2: Add History Smoke CLI Skeleton

**Files:**
- Modify: `tests/unit/test_history_smoke.py`
- Create: `scripts/history_smoke.py`

**Interfaces:**
- Consumes: `build_history_smoke_report`, `write_history_smoke_report`
- Produces:
  - `build_parser() -> argparse.ArgumentParser`
  - `main() -> None`

- [ ] **Step 1: Write failing CLI parser tests**

Append to `tests/unit/test_history_smoke.py`:

```python
from pathlib import Path
import subprocess
import sys

import scripts.history_smoke as history_smoke_script


def test_history_smoke_parser_accepts_real_mode_args() -> None:
    parser = history_smoke_script.build_parser()

    args = parser.parse_args(
        [
            "--url",
            "ws://127.0.0.1:8005/pipeline/stream",
            "--output-dir",
            "/tmp/history-smoke",
            "--mode",
            "real",
            "--turn1-wav",
            "/tmp/turn1.wav",
            "--turn2-wav",
            "/tmp/turn2.wav",
            "--expect",
            "小张",
        ]
    )

    assert args.mode == "real"
    assert args.turn1_wav == Path("/tmp/turn1.wav")
    assert args.turn2_wav == Path("/tmp/turn2.wav")
    assert args.expect == "小张"


def test_history_smoke_help_runs_by_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/history_smoke.py", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--mode" in result.stdout
    assert "--expect" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_history_smoke.py::test_history_smoke_parser_accepts_real_mode_args tests/unit/test_history_smoke.py::test_history_smoke_help_runs_by_path -q
```

Expected: FAIL because `scripts.history_smoke` does not exist.

- [ ] **Step 3: Create CLI skeleton**

Create `scripts/history_smoke.py`:

```python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bionic_head.client.history_smoke import write_history_smoke_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a two-turn stream history smoke.")
    parser.add_argument("--url", required=True, help="WebSocket URL, e.g. ws://127.0.0.1:8005/pipeline/stream")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["mock", "real"], default="real")
    parser.add_argument("--turn1-wav", type=Path)
    parser.add_argument("--turn2-wav", type=Path)
    parser.add_argument("--expect", default="小张")
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    return parser


async def run_history_smoke(
    *,
    url: str,
    output_dir: Path,
    mode: str,
    turn1_wav: Path | None,
    turn2_wav: Path | None,
    expected_text: str,
    chunk_ms: int,
    timeout_sec: float,
):
    raise NotImplementedError("run_history_smoke is implemented in Task 3")


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "real" and (args.turn1_wav is None or args.turn2_wav is None):
        raise SystemExit("real mode requires --turn1-wav and --turn2-wav")

    report = asyncio.run(
        run_history_smoke(
            url=args.url,
            output_dir=args.output_dir,
            mode=args.mode,
            turn1_wav=args.turn1_wav,
            turn2_wav=args.turn2_wav,
            expected_text=args.expect,
            chunk_ms=args.chunk_ms,
            timeout_sec=args.timeout_sec,
        )
    )
    write_history_smoke_report(args.output_dir / "history_smoke_report.json", report)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if not report.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_history_smoke.py::test_history_smoke_parser_accepts_real_mode_args tests/unit/test_history_smoke.py::test_history_smoke_help_runs_by_path -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/history_smoke.py tests/unit/test_history_smoke.py
git commit -m "feat: add history smoke cli skeleton"
```

---

### Task 3: Implement Fake-WebSocket Two-Turn Runner

**Files:**
- Modify: `scripts/history_smoke.py`
- Modify: `tests/unit/test_history_smoke.py`

**Interfaces:**
- Consumes:
  - `scripts.stream_client.client_event`
  - `scripts.stream_client.pcm_chunks`
  - `scripts.stream_client.read_pcm16_from_wav`
  - `HistorySmokeTurn`
  - `build_history_smoke_report`
- Produces:
  - `run_history_smoke(*, url: str, output_dir: Path, mode: str, turn1_wav: Path | None, turn2_wav: Path | None, expected_text: str, chunk_ms: int, timeout_sec: float) -> HistorySmokeReport`

- [ ] **Step 1: Write failing fake websocket runner test**

Append to `tests/unit/test_history_smoke.py`:

```python
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4
import wave
from array import array

import pytest


SESSION_ID = UUID("00000000-0000-0000-0000-000000000041")
TURN_1_ID = UUID("00000000-0000-0000-0000-000000000042")
TURN_2_ID = UUID("00000000-0000-0000-0000-000000000043")


class HistoryFakeConnect:
    def __init__(self, websocket) -> None:
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class HistoryFakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str | bytes] = []
        self._sequence = 1
        self._stage = "ready"

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        sent_types = [event["type"] for event in self.sent_json()]
        if self._stage == "ready":
            self._stage = "turn1_asr"
            return history_server_event("server.session.ready", sequence=self._next_sequence(), turn_id=None)
        if self._stage == "turn1_asr" and sent_types.count("client.audio.end") >= 1:
            self._stage = "turn1_llm"
            return history_server_event(
                "server.asr.final",
                sequence=self._next_sequence(),
                turn_id=TURN_1_ID,
                payload={"text": "我叫小张。"},
            )
        if self._stage == "turn1_llm":
            self._stage = "turn1_done"
            return history_server_event(
                "server.llm.token",
                sequence=self._next_sequence(),
                turn_id=TURN_1_ID,
                payload={"text": "你好小张。"},
            )
        if self._stage == "turn1_done":
            self._stage = "turn2_asr"
            return history_server_event(
                "server.pipeline.done",
                sequence=self._next_sequence(),
                turn_id=TURN_1_ID,
                payload={
                    "history_enabled": True,
                    "history_turn_count_before": 0,
                    "history_turn_count_after": 1,
                },
            )
        if self._stage == "turn2_asr" and sent_types.count("client.audio.end") >= 2:
            self._stage = "turn2_llm"
            return history_server_event(
                "server.asr.final",
                sequence=self._next_sequence(),
                turn_id=TURN_2_ID,
                payload={"text": "我叫什么？"},
            )
        if self._stage == "turn2_llm":
            self._stage = "turn2_done"
            return history_server_event(
                "server.llm.token",
                sequence=self._next_sequence(),
                turn_id=TURN_2_ID,
                payload={"text": "你叫小张。"},
            )
        if self._stage == "turn2_done":
            self._stage = "exhausted"
            return history_server_event(
                "server.pipeline.done",
                sequence=self._next_sequence(),
                turn_id=TURN_2_ID,
                payload={
                    "history_enabled": True,
                    "history_turn_count_before": 1,
                    "history_turn_count_after": 2,
                },
            )
        raise AssertionError(f"unexpected fake websocket stage: {self._stage}")

    def sent_json(self) -> list[dict[str, object]]:
        return [json.loads(message) for message in self.sent if isinstance(message, str)]

    def _next_sequence(self) -> int:
        current = self._sequence
        self._sequence += 1
        return current


def history_server_event(
    event_type: str,
    *,
    sequence: int,
    turn_id: UUID | None,
    payload: dict[str, object] | None = None,
) -> str:
    return json.dumps(
        {
            "protocol": "bionic-head-stream-v1",
            "type": event_type,
            "event_id": str(uuid4()),
            "session_id": str(SESSION_ID),
            "turn_id": str(turn_id) if turn_id is not None else None,
            "sequence": sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload or {},
        }
    )


def write_test_wav(path: Path) -> None:
    samples = array("h", [2000, -2000] * 800)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())


@pytest.mark.asyncio
async def test_history_smoke_runner_drives_two_turns_and_writes_success_report(monkeypatch, tmp_path) -> None:
    websocket = HistoryFakeWebSocket()
    ids = iter([SESSION_ID, TURN_1_ID, TURN_2_ID])
    turn1 = tmp_path / "turn1.wav"
    turn2 = tmp_path / "turn2.wav"
    write_test_wav(turn1)
    write_test_wav(turn2)

    monkeypatch.setattr(history_smoke_script, "uuid4", lambda: next(ids))
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        SimpleNamespace(connect=lambda url: HistoryFakeConnect(websocket)),
    )

    report = await history_smoke_script.run_history_smoke(
        url="ws://127.0.0.1:8005/pipeline/stream",
        output_dir=tmp_path,
        mode="real",
        turn1_wav=turn1,
        turn2_wav=turn2,
        expected_text="小张",
        chunk_ms=40,
        timeout_sec=5.0,
    )

    sent_types = [event["type"] for event in websocket.sent_json()]
    assert sent_types[0] == "client.session.start"
    assert sent_types.count("client.audio.start") == 2
    assert sent_types.count("client.audio.end") == 2
    assert report.success is True
    assert report.turns[0].asr_text == "我叫小张。"
    assert report.turns[1].asr_text == "我叫什么？"
    assert report.turns[1].history_turn_count_before == 1
    assert report.turns[1].reply_contains_expected is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_history_smoke.py::test_history_smoke_runner_drives_two_turns_and_writes_success_report -q
```

Expected: FAIL with `NotImplementedError: run_history_smoke is implemented in Task 3`.

- [ ] **Step 3: Implement runner**

Modify `scripts/history_smoke.py`:

```python
from uuid import UUID, uuid4

from scripts.stream_client import client_event, pcm_chunks, read_pcm16_from_wav
from bionic_head.client.history_smoke import (
    HistorySmokeReport,
    HistorySmokeTurn,
    build_history_smoke_report,
    write_history_smoke_report,
)


async def run_history_smoke(
    *,
    url: str,
    output_dir: Path,
    mode: str,
    turn1_wav: Path | None,
    turn2_wav: Path | None,
    expected_text: str,
    chunk_ms: int,
    timeout_sec: float,
) -> HistorySmokeReport:
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit("websockets is required; install the client extra") from exc

    if turn1_wav is None or turn2_wav is None:
        raise SystemExit("--turn1-wav and --turn2-wav are required for live history smoke")

    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid4()
    turn_ids = [uuid4(), uuid4()]
    sequence = 1
    turns: list[HistorySmokeTurn] = []

    async def send_json(websocket, event_type: str, turn_id: UUID | None, payload: dict[str, object]) -> None:
        nonlocal sequence
        current_sequence = sequence
        sequence += 1
        await websocket.send(
            json.dumps(
                client_event(
                    event_type,
                    session_id=session_id,
                    turn_id=turn_id,
                    sequence=current_sequence,
                    payload=payload,
                )
            )
        )

    async def send_wav_turn(websocket, turn_id: UUID, wav_path: Path) -> None:
        await send_json(
            websocket,
            "client.audio.start",
            turn_id,
            {"sample_rate": 16000, "channels": 1, "sample_width_bytes": 2},
        )
        for chunk in pcm_chunks(read_pcm16_from_wav(wav_path), chunk_ms=chunk_ms):
            await send_json(
                websocket,
                "client.audio.chunk",
                turn_id,
                {"byte_length": len(chunk), "duration_ms": int(len(chunk) / 2 / 16000 * 1000)},
            )
            await websocket.send(chunk)
        await send_json(websocket, "client.audio.end", turn_id, {"reason": "history_smoke"})

    async def wait_for_turn_done(websocket, *, turn_index: int, turn_id: UUID) -> HistorySmokeTurn:
        asr_text: str | None = None
        reply_parts: list[str] = []
        history_enabled: bool | None = None
        history_before: int | None = None
        history_after: int | None = None
        terminal_event: str | None = None

        while terminal_event is None:
            message = await asyncio.wait_for(websocket.recv(), timeout=timeout_sec)
            if isinstance(message, bytes):
                continue
            envelope = json.loads(message)
            event_type = str(envelope.get("type"))
            payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
            if str(envelope.get("turn_id")) != str(turn_id):
                continue
            if event_type == "server.asr.final":
                text = payload.get("text")
                asr_text = str(text) if text is not None else None
            elif event_type == "server.llm.token":
                text = payload.get("text")
                if text is not None:
                    reply_parts.append(str(text))
            elif event_type in {"server.pipeline.done", "server.pipeline.error", "server.turn.cancelled"}:
                terminal_event = event_type
                history_enabled = _bool_or_none(payload.get("history_enabled"))
                history_before = _int_or_none(payload.get("history_turn_count_before"))
                history_after = _int_or_none(payload.get("history_turn_count_after"))

        return HistorySmokeTurn(
            turn_index=turn_index,
            turn_id=str(turn_id),
            terminal_event=terminal_event,
            asr_text=asr_text,
            llm_reply="".join(reply_parts) or None,
            history_enabled=history_enabled,
            history_turn_count_before=history_before,
            history_turn_count_after=history_after,
        )

    async with websockets.connect(url) as websocket:
        await send_json(websocket, "client.session.start", None, {"client_name": "history_smoke"})
        first = await asyncio.wait_for(websocket.recv(), timeout=timeout_sec)
        if isinstance(first, bytes):
            raise RuntimeError("expected server.session.ready JSON")

        await send_wav_turn(websocket, turn_ids[0], turn1_wav)
        turns.append(await wait_for_turn_done(websocket, turn_index=1, turn_id=turn_ids[0]))
        await send_wav_turn(websocket, turn_ids[1], turn2_wav)
        turns.append(await wait_for_turn_done(websocket, turn_index=2, turn_id=turn_ids[1]))

    report = build_history_smoke_report(
        mode=mode,
        session_id=str(session_id),
        expected_text=expected_text,
        turns=turns,
    )
    write_history_smoke_report(output_dir / "history_smoke_report.json", report)
    return report


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_history_smoke.py::test_history_smoke_runner_drives_two_turns_and_writes_success_report -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/history_smoke.py tests/unit/test_history_smoke.py
git commit -m "feat: add two-turn history smoke runner"
```

---

### Task 4: Expose Server History Metrics to Terminal Events

**Files:**
- Modify: `src/bionic_head/orchestrators/stream.py`
- Modify: `tests/unit/test_stream_orchestrator.py`

**Interfaces:**
- Consumes: existing `_StreamTiming.snapshot()`
- Produces: `server.pipeline.done` payload includes:
  - `history_enabled`
  - `history_turn_count_before`
  - `history_char_count_before`
  - `history_turn_count_after`
  - `history_char_count_after`

- [ ] **Step 1: Write failing terminal payload test**

Add this to `tests/unit/test_stream_orchestrator.py` near the history tests:

```python
@pytest.mark.asyncio
async def test_stream_pipeline_done_payload_includes_history_metrics(
    stream_harness_factory,
) -> None:
    history = ConversationHistoryStore(max_turn_pairs=6, max_chars=3000)
    harness = stream_harness_factory(history=history)
    history.append_pair(
        harness.turn.session_id,
        user="我叫小张。",
        assistant="你好小张。",
    )

    await harness.run()

    done = next(
        envelope
        for envelope in harness.json_envelopes
        if envelope.type.value == "server.pipeline.done"
    )
    assert done.payload["history_enabled"] is True
    assert done.payload["history_turn_count_before"] == 1
    assert done.payload["history_turn_count_after"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_orchestrator.py::test_stream_pipeline_done_payload_includes_history_metrics -q
```

Expected: FAIL with `KeyError: 'history_enabled'`.

- [ ] **Step 3: Include history metrics in done payload**

Modify the success terminal emit in `src/bionic_head/orchestrators/stream.py`:

```python
            if await turn.emit_terminal_once(EventType.SERVER_PIPELINE_DONE):
                await emit_json(
                    factory.server(
                        EventType.SERVER_PIPELINE_DONE,
                        turn.turn_id,
                        stream_timing.history_payload(),
                    )
                )
```

Add this method to `_StreamTiming`:

```python
    def history_payload(self) -> dict[str, object]:
        return {
            "history_enabled": self.history_enabled,
            "history_turn_count_before": self.history_turn_count_before,
            "history_char_count_before": self.history_char_count_before,
            "history_turn_count_after": self.history_turn_count_after,
            "history_char_count_after": self.history_char_count_after,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_orchestrator.py::test_stream_pipeline_done_payload_includes_history_metrics tests/unit/test_stream_orchestrator.py::test_stream_records_history_metrics_in_timeline -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bionic_head/orchestrators/stream.py tests/unit/test_stream_orchestrator.py
git commit -m "feat: include history metrics in stream done payload"
```

---

### Task 5: Document History Smoke Commands and Verify

**Files:**
- Modify: `docs/operations/interactive-demo-client.md`
- Modify: `tests/unit/test_history_smoke.py`

**Interfaces:**
- Consumes: `scripts/history_smoke.py --help`
- Produces: documented mock/real commands.

- [ ] **Step 1: Add documentation**

Append this section to `docs/operations/interactive-demo-client.md` after the Task 16 history note:

```markdown
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
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_history_smoke.py tests/unit/test_stream_orchestrator.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: PASS with skipped real-provider tests unchanged.

- [ ] **Step 4: Commit**

```bash
git add docs/operations/interactive-demo-client.md tests/unit/test_history_smoke.py
git commit -m "docs: document history smoke"
```

---

## Final Verification

- [ ] Run:

```bash
git status --short --branch
PYTHONPATH=src .venv/bin/python -m pytest -q
```

- [ ] Confirm:

```text
working tree clean
full pytest passes
scripts/history_smoke.py --help exits 0
```

- [ ] Then ask whether to merge this branch back to `main`.
