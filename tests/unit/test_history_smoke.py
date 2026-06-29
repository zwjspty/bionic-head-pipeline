from pathlib import Path
from array import array
from datetime import datetime, timezone
import json
import subprocess
import sys
from types import SimpleNamespace
from uuid import UUID, uuid4
import wave

import pytest

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


def test_history_smoke_parser_accepts_real_mode_args() -> None:
    import scripts.history_smoke as history_smoke_script

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
    import scripts.history_smoke as history_smoke_script

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
    assert [
        event["payload"]["reason"]
        for event in websocket.sent_json()
        if event["type"] == "client.audio.end"
    ] == ["client_end", "client_end"]
    assert report.success is True
    assert report.turns[0].asr_text == "我叫小张。"
    assert report.turns[1].asr_text == "我叫什么？"
    assert report.turns[1].history_turn_count_before == 1
    assert report.turns[1].reply_contains_expected is True
