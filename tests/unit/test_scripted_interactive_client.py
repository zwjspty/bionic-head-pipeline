import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import scripts.interactive_demo_client as interactive
from bionic_head.client.scripted import ScriptedAction, build_scripted_actions


SESSION_ID = UUID("00000000-0000-0000-0000-000000000031")
TURN_1_ID = UUID("00000000-0000-0000-0000-000000000032")
TURN_2_ID = UUID("00000000-0000-0000-0000-000000000033")
WAV_1 = b"RIFF1111WAVE"
WAV_2 = b"RIFF2222WAVE"


def test_scripted_controller_generates_two_turn_cancel_flow() -> None:
    actions = build_scripted_actions(turn_count=2, cancel_first_turn=True)

    assert actions == [
        ScriptedAction.START_RECORDING,
        ScriptedAction.STOP_RECORDING,
        ScriptedAction.WAIT_FOR_PLAYBACK,
        ScriptedAction.CANCEL,
        ScriptedAction.WAIT_FOR_TURN_CANCELLED,
        ScriptedAction.START_RECORDING,
        ScriptedAction.STOP_RECORDING,
        ScriptedAction.WAIT_FOR_PIPELINE_DONE,
        ScriptedAction.QUIT,
    ]


class FakeConnect:
    def __init__(self, websocket) -> None:
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class ScriptedFakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str | bytes] = []
        self._ready_sent = False
        self._stage = "turn1_tts"
        self._sequence = 2

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        if not self._ready_sent:
            self._ready_sent = True
            return server_event("server.session.ready", sequence=1, turn_id=None, generation_epoch=0)

        while True:
            sent_types = [event["type"] for event in self.sent_json()]
            if self._stage == "turn1_tts" and sent_types.count("client.audio.end") >= 1:
                self._stage = "turn1_tts_binary"
                return server_tts_event(
                    sequence=self._next_sequence(),
                    turn_id=TURN_1_ID,
                    generation_epoch=0,
                    chunk_id="turn1-chunk",
                    byte_length=len(WAV_1),
                )
            if self._stage == "turn1_tts_binary":
                self._stage = "turn1_ue5"
                return WAV_1
            if self._stage == "turn1_ue5":
                self._stage = "wait_cancel"
                return server_ue5_event(
                    sequence=self._next_sequence(),
                    turn_id=TURN_1_ID,
                    generation_epoch=0,
                    chunk_id="turn1-face",
                )
            if self._stage == "wait_cancel" and "client.turn.cancel" in sent_types:
                self._stage = "turn1_cancelled"
                return server_event(
                    "server.playback.stop",
                    sequence=self._next_sequence(),
                    turn_id=TURN_1_ID,
                    generation_epoch=1,
                )
            if self._stage == "turn1_cancelled":
                self._stage = "turn2_tts"
                return server_event(
                    "server.turn.cancelled",
                    sequence=self._next_sequence(),
                    turn_id=TURN_1_ID,
                    generation_epoch=1,
                )
            if self._stage == "turn2_tts" and sent_types.count("client.audio.end") >= 2:
                self._stage = "turn2_tts_binary"
                return server_tts_event(
                    sequence=self._next_sequence(),
                    turn_id=TURN_2_ID,
                    generation_epoch=1,
                    chunk_id="turn2-chunk",
                    byte_length=len(WAV_2),
                )
            if self._stage == "turn2_tts_binary":
                self._stage = "turn2_ue5"
                return WAV_2
            if self._stage == "turn2_ue5":
                self._stage = "done"
                return server_ue5_event(
                    sequence=self._next_sequence(),
                    turn_id=TURN_2_ID,
                    generation_epoch=1,
                    chunk_id="turn2-face",
                )
            if self._stage == "done":
                self._stage = "exhausted"
                return server_event(
                    "server.pipeline.done",
                    sequence=self._next_sequence(),
                    turn_id=TURN_2_ID,
                    generation_epoch=1,
                )
            await asyncio_sleep()

    def sent_json(self) -> list[dict[str, object]]:
        return [json.loads(message) for message in self.sent if isinstance(message, str)]

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence


async def asyncio_sleep() -> None:
    import asyncio

    await asyncio.sleep(0)


def server_event(
    event_type: str,
    *,
    sequence: int,
    turn_id: UUID | None,
    generation_epoch: int,
) -> str:
    return json.dumps(
        {
            "protocol": "bionic-head-stream-v1",
            "type": event_type,
            "event_id": str(uuid4()),
            "session_id": str(SESSION_ID),
            "turn_id": str(turn_id) if turn_id is not None else None,
            "sequence": sequence,
            "generation_epoch": generation_epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "session_id": str(SESSION_ID),
                "turn_id": str(turn_id) if turn_id is not None else None,
                "generation_epoch": generation_epoch,
            },
        }
    )


def server_tts_event(
    *,
    sequence: int,
    turn_id: UUID,
    generation_epoch: int,
    chunk_id: str,
    byte_length: int,
) -> str:
    envelope = json.loads(
        server_event(
            "server.tts.audio",
            sequence=sequence,
            turn_id=turn_id,
            generation_epoch=generation_epoch,
        )
    )
    envelope["payload"].update(
        {
            "chunk_id": chunk_id,
            "segment_id": chunk_id,
            "format": "wav",
            "byte_length": byte_length,
        }
    )
    return json.dumps(envelope)


def server_ue5_event(*, sequence: int, turn_id: UUID, generation_epoch: int, chunk_id: str) -> str:
    envelope = json.loads(
        server_event(
            "server.ue5.frames",
            sequence=sequence,
            turn_id=turn_id,
            generation_epoch=generation_epoch,
        )
    )
    envelope["payload"].update(
        {
            "chunk_id": chunk_id,
            "segment_id": chunk_id,
            "start_frame_index": 0,
            "frame_count": 1,
            "fps": 30,
            "frames": [{"frame_index": 0, "time_seconds": 0.0, "weights": [0.0] * 52}],
        }
    )
    return json.dumps(envelope)


@pytest.mark.asyncio
async def test_scripted_mode_runs_two_fake_turns_and_writes_report(monkeypatch, tmp_path) -> None:
    websocket = ScriptedFakeWebSocket()
    ids = iter([SESSION_ID, TURN_1_ID, TURN_2_ID])

    monkeypatch.setattr(interactive, "uuid4", lambda: next(ids))
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        SimpleNamespace(connect=lambda url: FakeConnect(websocket)),
    )

    terminal = await interactive.run_scripted_demo(
        url="ws://127.0.0.1:8005/pipeline/stream",
        output_dir=tmp_path,
        scripted_turns=2,
        scripted_cancel_after_ms=0,
        chunk_ms=40,
        sample_rate=16000,
        audio_backend="null",
    )

    sent_types = [event["type"] for event in websocket.sent_json()]
    report = json.loads((tmp_path / "interaction_report.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert terminal == "server.pipeline.done"
    assert sent_types[0] == "client.session.start"
    assert sent_types.count("client.audio.start") == 2
    assert sent_types.count("client.audio.chunk") >= 10
    assert sent_types.count("client.audio.end") == 2
    assert sent_types.count("client.turn.cancel") == 1
    assert sent_types.index("client.turn.cancel") > sent_types.index("client.audio.end")
    assert sent_types.index("client.turn.cancel") < sent_types.index("client.audio.start", 2)
    assert "mode" not in websocket.sent_json()[0]["payload"]
    assert [
        event["payload"].get("reason")
        for event in websocket.sent_json()
        if event["type"] == "client.audio.end"
    ] == ["client_end", "client_end"]
    assert report["success"] is True
    assert report["mode"] == "scripted"
    assert report["turn_count"] == 2
    assert report["completed_turn_count"] == 1
    assert report["cancelled_turn_count"] == 1
    assert report["playback_stop_count"] == 1
    assert report["old_generation_audio_play_count"] == 0
    assert report["old_generation_face_display_count"] == 0
    assert report["client_interrupt_to_audio_stop_ms"] is not None
    assert summary["tts_chunks"] == 2
    assert summary["ue5_chunks"] == 2
