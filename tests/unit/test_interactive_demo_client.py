import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import scripts.interactive_demo_client as interactive


SESSION_ID = UUID("00000000-0000-0000-0000-000000000021")
TURN_ID = UUID("00000000-0000-0000-0000-000000000022")


class FakeConnect:
    def __init__(self, websocket) -> None:
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class CommandAwareFakeWebSocket:
    def __init__(self, *, terminal_event: str = "server.pipeline.done") -> None:
        self.sent: list[str | bytes] = []
        self._ready_sent = False
        self._terminal_sent = False
        self._terminal_event = terminal_event

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        if not self._ready_sent:
            self._ready_sent = True
            return server_event("server.session.ready", sequence=1, session_id=SESSION_ID, turn_id=None)
        while not self._terminal_sent:
            sent_types = [
                json.loads(message)["type"]
                for message in self.sent
                if isinstance(message, str)
            ]
            if "client.audio.end" in sent_types or "client.turn.cancel" in sent_types:
                self._terminal_sent = True
                return server_event(
                    self._terminal_event,
                    sequence=2,
                    session_id=SESSION_ID,
                    turn_id=TURN_ID,
                    generation_epoch=1 if self._terminal_event == "server.turn.cancelled" else 0,
                )
            await asyncio_sleep()
        raise AssertionError("fake websocket exhausted")


async def asyncio_sleep() -> None:
    import asyncio

    await asyncio.sleep(0)


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


def server_event(
    event_type: str,
    *,
    sequence: int,
    session_id: UUID,
    turn_id: UUID | None,
    generation_epoch: int = 0,
) -> str:
    return json.dumps(
        {
            "protocol": "bionic-head-stream-v1",
            "type": event_type,
            "event_id": str(uuid4()),
            "session_id": str(session_id),
            "turn_id": str(turn_id) if turn_id is not None else None,
            "sequence": sequence,
            "generation_epoch": generation_epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "session_id": str(session_id),
                "turn_id": str(turn_id) if turn_id is not None else None,
                "generation_epoch": generation_epoch,
            },
        }
    )


def sent_json_messages(websocket: CommandAwareFakeWebSocket) -> list[dict[str, object]]:
    return [json.loads(message) for message in websocket.sent if isinstance(message, str)]


def test_build_parser_accepts_required_interactive_args() -> None:
    parser = interactive.build_parser()

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
    assert interactive.chunk_samples_for_ms(16000, 40) == 640


def test_interactive_demo_client_help_runs_when_executed_by_path_with_src_pythonpath() -> None:
    env = {
        **os.environ,
        "PYTHONPATH": "src",
    }

    result = subprocess.run(
        [sys.executable, "scripts/interactive_demo_client.py", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--url" in result.stdout
    assert "--output-dir" in result.stdout


@pytest.mark.asyncio
async def test_interactive_enter_enter_sends_audio_turn(monkeypatch, tmp_path) -> None:
    websocket = CommandAwareFakeWebSocket()
    mic = FakeMicrophone([b"\x01\x02" * 320, b""])
    ids = iter([SESSION_ID, TURN_ID])

    monkeypatch.setattr(interactive, "uuid4", lambda: next(ids))
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

    events = sent_json_messages(websocket)
    assert terminal == "server.pipeline.done"
    assert [event["type"] for event in events] == [
        "client.session.start",
        "client.audio.start",
        "client.audio.chunk",
        "client.audio.end",
    ]
    assert websocket.sent[3] == b"\x01\x02" * 320
    assert mic.started == 1
    assert mic.stopped == 1
    assert mic.closed == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["client_mic_recording_started_count"] == 1
    assert summary["client_mic_recording_stopped_count"] == 1
    assert summary["client_mic_chunks_sent"] == 1
    assert summary["client_mic_bytes_sent"] == 640


@pytest.mark.asyncio
async def test_interactive_cancel_command_sends_turn_cancel(monkeypatch, tmp_path) -> None:
    websocket = CommandAwareFakeWebSocket(terminal_event="server.turn.cancelled")
    ids = iter([SESSION_ID, TURN_ID])

    monkeypatch.setattr(interactive, "uuid4", lambda: next(ids))
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        SimpleNamespace(connect=lambda url: FakeConnect(websocket)),
    )

    terminal = await interactive.run_interactive_demo(
        url="ws://127.0.0.1:8005/pipeline/stream",
        output_dir=tmp_path,
        command_source=FakeCommandSource(["c", "q"]),
        microphone=FakeMicrophone([]),
        play_audio=False,
        chunk_ms=40,
        sample_rate=16000,
    )

    events = sent_json_messages(websocket)
    assert terminal == "server.turn.cancelled"
    assert [event["type"] for event in events] == [
        "client.session.start",
        "client.turn.cancel",
    ]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["client_manual_cancel_count"] == 1
    assert summary["client_interrupt_sent_ms"] is not None


def test_sounddevice_microphone_requires_optional_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)

    with pytest.raises(SystemExit, match="sounddevice is required for microphone input"):
        interactive.SoundDeviceMicrophoneInput(sample_rate=16000, chunk_ms=40)
