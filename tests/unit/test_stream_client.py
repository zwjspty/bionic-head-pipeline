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


def test_playback_stop_clears_pending_buffers_and_tracks_epoch(tmp_path) -> None:
    now = iter([40.0, 40.100])
    receiver = ClientReceiver(tmp_path, session_id=SESSION_ID, turn_id=TURN_ID, clock=lambda: next(now))
    receiver.pending_tts = object()
    receiver.pending_segments["0"] = object()
    receiver.pending_ue5_chunks["0"] = object()

    receiver.accept_json(
        server_event(
            event_type="server.playback.stop",
            sequence=1,
            generation_epoch=3,
            payload={},
        )
    )

    assert receiver.pending_tts is None
    assert receiver.pending_segments == {}
    assert receiver.pending_ue5_chunks == {}
    assert receiver.terminal_event is None
    assert receiver.summary["playback_stop_count"] == 1
    assert receiver.summary["latest_generation_epoch"] == 3
    assert receiver.summary["event_first_ms"]["server.playback.stop"] == 100.0


def test_stale_lower_epoch_event_is_counted_but_not_applied(tmp_path) -> None:
    now = iter([50.0, 50.100, 50.200])
    receiver = ClientReceiver(tmp_path, session_id=SESSION_ID, turn_id=TURN_ID, clock=lambda: next(now))
    receiver.accept_json(
        server_event(
            event_type="server.playback.stop",
            sequence=1,
            generation_epoch=2,
            payload={},
        )
    )

    receiver.accept_json(
        server_event(
            event_type="server.tts.audio",
            sequence=2,
            generation_epoch=1,
            payload={"chunk_id": "old", "byte_length": 4, "format": "wav"},
        )
    )

    assert receiver.pending_tts is None
    assert receiver.summary["stale_drop_count"] == 1


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


def test_receiver_records_segment_face_timing_from_tts_and_ue5_events(tmp_path) -> None:
    now = iter([60.0, 60.100, 60.250, 60.480])
    receiver = ClientReceiver(tmp_path, session_id=SESSION_ID, turn_id=TURN_ID, clock=lambda: next(now))
    receiver.accept_json(
        server_event(
            event_type="server.tts.audio",
            sequence=1,
            payload={"chunk_id": "chunk-0001", "byte_length": 4, "format": "wav"},
        )
    )
    receiver.accept_binary(b"RIFF")
    receiver.accept_json(
        server_event(
            event_type="server.ue5.frames",
            sequence=2,
            payload={
                "chunk_id": "chunk-0001-0000",
                "segment_id": "chunk-0001",
                "segment_index": 1,
                "start_frame_index": 0,
                "frame_count": 1,
                "timing": {
                    "face_total_ms": 375.0,
                    "ue5_first_frame_after_tts_ms": 380.0,
                    "e2e_first_visible_face_ms": 480.0,
                    "face_stitch_enabled": True,
                    "face_stitch_applied": True,
                    "face_stitch_reset": False,
                    "face_stitch_overlap_frames": 8.0,
                    "face_stitch_actual_overlap_frames": 5.0,
                    "face_boundary_delta_before": 0.4,
                    "face_boundary_delta_after": 0.1,
                    "eye_continuity_enabled": True,
                    "eye_continuity_applied": True,
                    "eye_continuity_reset": False,
                    "eye_smooth_channel_count": 2.0,
                    "blink_channel_count": 1.0,
                    "eye_continuity_overlap_frames": 6.0,
                    "eye_continuity_actual_overlap_frames": 4.0,
                    "eye_boundary_delta_before": 0.3,
                    "eye_boundary_delta_after": 0.2,
                    "blink_enabled": True,
                    "blink_applied_count": 1.0,
                    "blink_frame_count": 3.0,
                    "blink_reset_count": 0.0,
                    "eye_global_frame_start": 10.0,
                    "eye_global_frame_end": 20.0,
                },
                "frames": [{"frame_index": 0, "time_seconds": 0.0, "weights": [0.0] * 52}],
            },
        )
    )

    segments = receiver.summary["segments"]
    assert segments["chunk-0001"]["tts_audio_event_ms"] == 100.0
    assert segments["chunk-0001"]["tts_binary_ms"] == 250.0
    assert segments["chunk-0001"]["ue5_first_frame_ms"] == 480.0
    assert segments["chunk-0001"]["ue5_first_frame_after_tts_ms"] == 380.0
    assert segments["chunk-0001"]["face_total_ms"] == 375.0
    assert segments["chunk-0001"]["face_stitch_enabled"] is True
    assert segments["chunk-0001"]["face_stitch_applied"] is True
    assert segments["chunk-0001"]["face_stitch_reset"] is False
    assert segments["chunk-0001"]["face_stitch_actual_overlap_frames"] == 5.0
    assert segments["chunk-0001"]["face_boundary_delta_after"] == 0.1
    assert segments["chunk-0001"]["eye_continuity_enabled"] is True
    assert segments["chunk-0001"]["eye_continuity_applied"] is True
    assert segments["chunk-0001"]["eye_continuity_reset"] is False
    assert segments["chunk-0001"]["eye_smooth_channel_count"] == 2.0
    assert segments["chunk-0001"]["eye_continuity_actual_overlap_frames"] == 4.0
    assert segments["chunk-0001"]["eye_boundary_delta_after"] == 0.2
    assert segments["chunk-0001"]["blink_enabled"] is True
    assert segments["chunk-0001"]["blink_applied_count"] == 1.0
    assert segments["chunk-0001"]["blink_frame_count"] == 3.0
    assert segments["chunk-0001"]["blink_reset_count"] == 0.0
    assert receiver.summary["e2e_first_visible_face_ms"] == 480.0


def test_receiver_counts_stale_face_as_old_turn_leak(tmp_path) -> None:
    now = iter([70.0, 70.100, 70.200])
    receiver = ClientReceiver(tmp_path, session_id=SESSION_ID, turn_id=TURN_ID, clock=lambda: next(now))
    receiver.accept_json(
        server_event(
            event_type="server.playback.stop",
            sequence=1,
            generation_epoch=2,
            payload={},
        )
    )
    receiver.accept_json(
        server_event(
            event_type="server.ue5.frames",
            sequence=2,
            generation_epoch=1,
            payload={
                "chunk_id": "old-0000",
                "segment_id": "old",
                "start_frame_index": 0,
                "frame_count": 1,
                "frames": [{"frame_index": 0, "time_seconds": 0.0, "weights": [0.0] * 52}],
            },
        )
    )

    assert receiver.summary["stale_drop_count"] == 1
    assert receiver.summary["stale_face_drop_count"] == 1
    assert receiver.summary["old_turn_face_leak_count"] == 1


def server_event(
    event_type: str,
    sequence: int,
    payload: dict[str, object],
    generation_epoch: int = 0,
) -> dict[str, object]:
    return {
        "protocol": "bionic-head-stream-v1",
        "type": event_type,
        "event_id": str(uuid4()),
        "session_id": str(SESSION_ID),
        "turn_id": str(TURN_ID),
        "sequence": sequence,
        "generation_epoch": generation_epoch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "session_id": str(SESSION_ID),
            "turn_id": str(TURN_ID),
            "generation_epoch": generation_epoch,
            **payload,
        },
    }
