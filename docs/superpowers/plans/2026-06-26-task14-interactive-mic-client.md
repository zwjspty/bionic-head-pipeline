# Task 14 Interactive Mic Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal terminal interactive client that records microphone audio, sends it to `/pipeline/stream`, plays received TTS, buffers UE5 frames, and supports manual keyboard interrupt.

**Architecture:** Add `scripts/interactive_demo_client.py` with testable abstractions for microphone input and keyboard commands. Reuse `scripts/local_demo_client.py` for playback engines, metrics, receiver, protocol helpers, and sound playback. Keep real microphone support optional through `sounddevice`; tests use fake microphone and fake WebSocket objects.

**Tech Stack:** Python 3.10+, asyncio, websockets, optional sounddevice, pytest, existing `bionic-head-stream-v1` protocol.

## Global Constraints

- Do not add acoustic echo cancellation.
- Do not add WebRTC.
- Do not add browser UI.
- Do not add UE5 runtime integration.
- Do not add real-time Blender rendering.
- Default tests must not require microphone hardware, sound hardware, GPU, Ollama, Piper, EmoTalk, or a running FastAPI server.
- Preserve Task 12/13 `LocalDemoReceiver`, playback stop, and stale-drop behavior.
- Use TDD for implementation changes.

---

## File Structure

- Create `scripts/interactive_demo_client.py`
  - CLI parser.
  - `MicrophoneInput` and `CommandSource` protocols.
  - `SoundDeviceMicrophoneInput` real backend.
  - `ScriptedMicrophoneInput` only if useful for CLI-free tests.
  - `InteractiveDemoSession`.
- Create `tests/unit/test_interactive_demo_client.py`
  - Hermetic unit tests with fake websocket, fake mic, fake commands.
- Modify `docs/operations/local-demo-client.md`
  - Add real interactive mic client commands.
- Support CLI backend selectors:
  - `--mic-backend sounddevice|fake`
  - `--audio-backend sounddevice|null`
- Optionally modify `pyproject.toml`
  - Only if a new optional extra is needed. Prefer reusing `client-audio`.

---

## Task 14.1: CLI and microphone backend primitives

**Files:**
- Create: `scripts/interactive_demo_client.py`
- Create: `tests/unit/test_interactive_demo_client.py`

**Interfaces:**
- Produces:
  - `build_parser() -> argparse.ArgumentParser`
  - `chunk_samples_for_ms(sample_rate: int, chunk_ms: int) -> int`
  - `SoundDeviceMicrophoneInput(sample_rate: int, chunk_ms: int, channels: int = 1)`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_interactive_demo_client.py`:

```python
import subprocess
import sys
from pathlib import Path

from scripts.interactive_demo_client import build_parser, chunk_samples_for_ms


def test_build_parser_accepts_required_interactive_args() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--url",
            "ws://127.0.0.1:8005/pipeline/stream",
            "--output-dir",
            "/tmp/interactive",
            "--no-play-audio",
        ]
    )

    assert args.url == "ws://127.0.0.1:8005/pipeline/stream"
    assert args.output_dir == Path("/tmp/interactive")
    assert args.play_audio is False


def test_chunk_samples_for_ms_uses_16k_pcm_window() -> None:
    assert chunk_samples_for_ms(16000, 40) == 640


def test_interactive_demo_client_help_runs_when_executed_by_path_with_src_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/interactive_demo_client.py", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        env={"PYTHONPATH": "src"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--url" in result.stdout
    assert "--output-dir" in result.stdout
```

- [ ] **Step 2: Run red tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_interactive_demo_client.py -q
```

Expected: FAIL because `scripts.interactive_demo_client` does not exist.

- [ ] **Step 3: Implement minimal CLI primitives**

Create `scripts/interactive_demo_client.py` with:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def chunk_samples_for_ms(sample_rate: int, chunk_ms: int) -> int:
    return int(sample_rate * chunk_ms / 1000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the interactive microphone demo client.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--play-audio", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    build_parser().parse_args()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify tests pass**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_interactive_demo_client.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/interactive_demo_client.py tests/unit/test_interactive_demo_client.py
git commit -m "feat: add interactive demo client skeleton"
```

---

## Task 14.2: Fake microphone and interactive recording sends audio

**Files:**
- Modify: `scripts/interactive_demo_client.py`
- Modify: `tests/unit/test_interactive_demo_client.py`

**Interfaces:**
- Produces:
  - `InteractiveDemoSession`
  - `run_interactive_demo(...) -> str`

- [ ] **Step 1: Write failing interaction test**

Append a fake websocket, fake command source, and fake microphone test:

```python
import json
from types import SimpleNamespace
from uuid import UUID

import pytest

import scripts.interactive_demo_client as interactive


SESSION_ID = UUID("00000000-0000-0000-0000-000000000021")
TURN_ID = UUID("00000000-0000-0000-0000-000000000022")


class FakeWebSocket:
    def __init__(self, responses: list[str | bytes]) -> None:
        self.responses = list(responses)
        self.sent: list[str | bytes] = []

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        if not self.responses:
            raise AssertionError("fake websocket exhausted")
        return self.responses.pop(0)


class FakeConnect:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> FakeWebSocket:
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FakeCommandSource:
    def __init__(self, commands: list[str]) -> None:
        self.commands = list(commands)

    async def read_command(self) -> str:
        if not self.commands:
            raise AssertionError("fake command source exhausted")
        return self.commands.pop(0)


class FakeMicrophone:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.started = 0
        self.stopped = 0
        self.closed = 0

    async def start(self) -> None:
        self.started += 1

    async def read_chunk(self) -> bytes:
        if not self.chunks:
            return b""
        return self.chunks.pop(0)

    async def stop(self) -> None:
        self.stopped += 1

    async def close(self) -> None:
        self.closed += 1


def server_event(event_type: str, *, sequence: int, session_id: UUID, turn_id: UUID | None) -> str:
    return json.dumps(
        {
            "protocol": "bionic-head-stream-v1",
            "type": event_type,
            "event_id": f"event-{sequence}",
            "session_id": str(session_id),
            "turn_id": str(turn_id) if turn_id is not None else None,
            "sequence": sequence,
            "generation_epoch": 0,
            "timestamp": "2026-06-26T00:00:00Z",
            "payload": {
                "session_id": str(session_id),
                "turn_id": str(turn_id) if turn_id is not None else None,
                "generation_epoch": 0,
            },
        }
    )


@pytest.mark.asyncio
async def test_interactive_enter_enter_sends_audio_turn(monkeypatch, tmp_path) -> None:
    ready = server_event("server.session.ready", sequence=1, session_id=SESSION_ID, turn_id=None)
    done = server_event("server.pipeline.done", sequence=2, session_id=SESSION_ID, turn_id=TURN_ID)
    websocket = FakeWebSocket([ready, done])
    mic = FakeMicrophone([b"\x01\x02" * 320, b""])

    monkeypatch.setattr(interactive, "uuid4", lambda: TURN_ID)
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        SimpleNamespace(connect=lambda url: FakeConnect(websocket)),
    )

    terminal = await interactive.run_interactive_demo(
        url="ws://127.0.0.1:8005/pipeline/stream",
        output_dir=tmp_path,
        command_source=FakeCommandSource(["", "", "q"]),
        microphone=mic,
        play_audio=False,
        chunk_ms=40,
        sample_rate=16000,
    )

    sent_json = [json.loads(message) for message in websocket.sent if isinstance(message, str)]
    assert terminal == "server.pipeline.done"
    assert [event["type"] for event in sent_json] == [
        "client.session.start",
        "client.audio.start",
        "client.audio.chunk",
        "client.audio.end",
    ]
    assert websocket.sent[3] == b"\x01\x02" * 320
    assert mic.started == 1
    assert mic.stopped == 1
```

- [ ] **Step 2: Run red test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_interactive_demo_client.py::test_interactive_enter_enter_sends_audio_turn -q
```

Expected: FAIL because `run_interactive_demo` does not exist.

- [ ] **Step 3: Implement minimal interactive session**

Implement:

- `CommandSource` protocol;
- `MicrophoneInput` protocol;
- `run_interactive_demo`;
- client sequence handling;
- receiver task that passes server JSON/binary to `LocalDemoReceiver`;
- recording loop that sends audio chunks until stopped.

Use `client_event`, `LocalDemoReceiver`, `AudioPlaybackEngine`, `FacePlaybackEngine`, `MemoryAudioSink`, `SoundDeviceAudioSink`, and `PlaybackMetrics` from `scripts.local_demo_client`.

- [ ] **Step 4: Verify test passes**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_interactive_demo_client.py::test_interactive_enter_enter_sends_audio_turn -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/interactive_demo_client.py tests/unit/test_interactive_demo_client.py
git commit -m "feat: stream microphone chunks from interactive client"
```

---

## Task 14.3: Manual cancel command

**Files:**
- Modify: `scripts/interactive_demo_client.py`
- Modify: `tests/unit/test_interactive_demo_client.py`

**Interfaces:**
- Consumes: `run_interactive_demo`
- Produces: `c` sends `client.turn.cancel`

- [ ] **Step 1: Write failing cancel test**

Add a test where commands are `["c", "q"]` and assert a `client.turn.cancel` event is sent after session ready.

- [ ] **Step 2: Run red test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_interactive_demo_client.py::test_interactive_c_sends_turn_cancel -q
```

Expected: FAIL until command handling is implemented.

- [ ] **Step 3: Implement `c` handling**

When command is `"c"`:

- mark client interrupt sent through receiver metrics;
- send `client.turn.cancel`;
- increment a manual cancel count in interactive summary state.

- [ ] **Step 4: Verify tests pass**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_interactive_demo_client.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/interactive_demo_client.py tests/unit/test_interactive_demo_client.py
git commit -m "feat: add manual interrupt command to interactive client"
```

---

## Task 14.4: Real sounddevice microphone backend and docs

**Files:**
- Modify: `scripts/interactive_demo_client.py`
- Modify: `docs/operations/local-demo-client.md`

**Interfaces:**
- Produces: `SoundDeviceMicrophoneInput`
- Produces: documented command for real mic demo.

- [ ] **Step 1: Add tests for missing optional dependency**

Add a test that monkeypatches imports so constructing `SoundDeviceMicrophoneInput` raises `SystemExit` with a friendly message if `sounddevice` is missing.

- [ ] **Step 2: Run red test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_interactive_demo_client.py::test_sounddevice_microphone_requires_optional_dependency -q
```

Expected: FAIL until backend exists.

- [ ] **Step 3: Implement `SoundDeviceMicrophoneInput`**

Use `sounddevice.InputStream` with:

```text
samplerate = 16000
channels = 1
dtype = int16
blocksize = chunk_samples_for_ms(16000, chunk_ms)
```

Move callback frames into an `asyncio.Queue[bytes]` using the running loop.

- [ ] **Step 4: Update CLI main**

`main()` should instantiate `SoundDeviceMicrophoneInput` and call `asyncio.run(run_interactive_demo(...))`.

- [ ] **Step 5: Update docs**

Add:

```bash
.venv/bin/python -m pip install -e ".[client,client-audio]"
.venv/bin/python scripts/interactive_demo_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --output-dir /tmp/bionic-interactive-demo \
  --chunk-ms 40 \
  --play-audio
```

Document keys:

```text
Enter: start/stop recording
c: interrupt current playback
q: quit
```

- [ ] **Step 6: Verify focused tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_interactive_demo_client.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/interactive_demo_client.py tests/unit/test_interactive_demo_client.py docs/operations/local-demo-client.md
git commit -m "feat: add real microphone backend for interactive demo"
```

---

## Task 14.5: Final verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_interactive_demo_client.py tests/unit/test_local_demo_client.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: PASS with integration skips only.

- [ ] **Step 3: Run CLI help**

```bash
.venv/bin/python scripts/interactive_demo_client.py --help
```

Expected: usage text includes `--url` and `--output-dir`.

- [ ] **Step 4: Inspect git state**

```bash
git status --short --branch
git log --oneline -8
```

Expected: branch is `task14-interactive-mic-client`; working tree is clean after commits.

---

## Task 14.6: Backend selector refinement

**Files:**
- Modify: `scripts/interactive_demo_client.py`
- Modify: `tests/unit/test_interactive_demo_client.py`
- Modify: `docs/operations/local-demo-client.md`
- Modify: `docs/superpowers/specs/2026-06-26-task14-interactive-mic-client-design.md`

- [x] Add `FakeMicBackend` as a public no-hardware microphone backend.
- [x] Add `create_microphone_backend("sounddevice"|"fake", ...)`.
- [x] Add `create_audio_sink("sounddevice"|"null")`.
- [x] Add parser flags:

```text
--mic-backend sounddevice|fake
--audio-backend sounddevice|null
```

- [x] Preserve `--play-audio/--no-play-audio` compatibility.
- [x] Document the real run command and fake/null smoke command.
- [x] Document that Task 14 has no AEC and headphones are recommended.
