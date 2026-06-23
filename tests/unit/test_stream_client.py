from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from scripts.stream_client import ClientReceiver, ProtocolError


SESSION_ID = UUID("00000000-0000-0000-0000-000000000001")
TURN_ID = UUID("00000000-0000-0000-0000-000000000002")


def test_receiver_pairs_tts_metadata_with_next_binary(tmp_path) -> None:
    now = iter([10.0, 10.125, 10.250])
    receiver = ClientReceiver(tmp_path, session_id=SESSION_ID, turn_id=TURN_ID, clock=lambda: next(now))

    receiver.accept_json(
        server_event(
            event_type="server.tts.audio",
            sequence=1,
            payload={"chunk_id": "0", "byte_length": 4, "format": "wav"},
        )
    )
    receiver.accept_binary(b"RIFF")

    assert (tmp_path / "tts/0.wav").read_bytes() == b"RIFF"
    assert receiver.summary["tts_chunks"] == 1
    assert receiver.summary["event_first_ms"]["server.tts.audio"] == 125.0
    assert receiver.summary["first_tts_binary_ms"] == 250.0
    event = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert '"_client_received_ms": 125.0' in event


def test_binary_length_mismatch_is_rejected(tmp_path) -> None:
    receiver = ClientReceiver(tmp_path, session_id=SESSION_ID, turn_id=TURN_ID)
    receiver.accept_json(
        server_event(
            event_type="server.tts.audio",
            sequence=1,
            payload={"chunk_id": "0", "byte_length": 4, "format": "wav"},
        )
    )

    with pytest.raises(ProtocolError):
        receiver.accept_binary(b"bad")


def test_sequence_gap_is_rejected(tmp_path) -> None:
    receiver = ClientReceiver(tmp_path, session_id=SESSION_ID, turn_id=TURN_ID)
    receiver.accept_json(server_event(event_type="server.state", sequence=1, payload={"state": "listening"}))

    with pytest.raises(ProtocolError):
        receiver.accept_json(server_event(event_type="server.state", sequence=3, payload={"state": "thinking"}))


def test_cancel_clears_pending_playback(tmp_path) -> None:
    now = iter([30.0, 30.111])
    receiver = ClientReceiver(tmp_path, session_id=SESSION_ID, turn_id=TURN_ID, clock=lambda: next(now))
    receiver.pending_segments["0"] = object()
    receiver.pending_ue5_chunks["0"] = object()

    receiver.accept_json(
        server_event(
            event_type="server.turn.cancelled",
            sequence=1,
            payload={},
        )
    )

    assert receiver.pending_segments == {}
    assert receiver.pending_ue5_chunks == {}
    assert receiver.terminal_event == "server.turn.cancelled"
    assert receiver.summary["event_counts"]["server.turn.cancelled"] == 1
    assert receiver.summary["terminal_event_ms"] == 111.0


def test_ue5_frames_are_saved_and_gap_is_rejected(tmp_path) -> None:
    now = iter([20.0, 20.333, 20.500])
    receiver = ClientReceiver(tmp_path, session_id=SESSION_ID, turn_id=TURN_ID, clock=lambda: next(now))
    receiver.accept_json(
        server_event(
            event_type="server.ue5.frames",
            sequence=1,
            payload={
                "chunk_id": "face-0000",
                "start_frame_index": 0,
                "frame_count": 1,
                "frames": [{"frame_index": 0, "time_seconds": 0.0, "weights": [0.0] * 52}],
            },
        )
    )

    assert (tmp_path / "ue5/face-0000.json").exists()
    assert receiver.summary["event_first_ms"]["server.ue5.frames"] == 333.0

    with pytest.raises(ProtocolError):
        receiver.accept_json(
            server_event(
                event_type="server.ue5.frames",
                sequence=2,
                payload={
                    "chunk_id": "face-0001",
                    "start_frame_index": 2,
                    "frame_count": 1,
                    "frames": [{"frame_index": 2, "time_seconds": 0.0, "weights": [0.0] * 52}],
                },
            )
        )


def server_event(event_type: str, sequence: int, payload: dict[str, object]) -> dict[str, object]:
    return {
        "protocol": "bionic-head-stream-v1",
        "type": event_type,
        "event_id": str(uuid4()),
        "session_id": str(SESSION_ID),
        "turn_id": str(TURN_ID),
        "sequence": sequence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "session_id": str(SESSION_ID),
            "turn_id": str(TURN_ID),
            **payload,
        },
    }
