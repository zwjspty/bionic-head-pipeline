from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import itertools
import os

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.skipif(
    os.environ.get("BIONIC_HEAD_RUN_REAL_EMOTALK") != "1",
    reason="set BIONIC_HEAD_RUN_REAL_EMOTALK=1 to run the real stream EmoTalk sidecar smoke test",
)


TERMINAL_TYPES = {"server.pipeline.done", "server.pipeline.error", "server.turn.cancelled"}


@pytest.mark.integration
def test_real_stream_emotalk_sidecar_emits_ue5_timing(real_app, real_settings, chinese_pcm: bytes) -> None:
    assert real_settings.adapters.audio2face.provider == "emotalk_sidecar"

    events, binaries = _run_ws_turn(real_app, chinese_pcm)
    event_types = [event["type"] for event in events]

    assert event_types[-1] == "server.pipeline.done"
    assert binaries
    assert "server.tts.audio" in event_types
    assert "server.ue5.frames" in event_types
    assert event_types.index("server.tts.audio") < event_types.index("server.ue5.frames")

    ue5_event = next(event for event in events if event["type"] == "server.ue5.frames")
    payload = ue5_event["payload"]
    timing = payload.get("timing")
    assert payload["segment_id"]
    assert payload["turn_id"] == ue5_event["turn_id"]
    assert payload["generation_epoch"] == ue5_event["generation_epoch"]
    assert isinstance(timing, dict)
    assert timing["face_total_ms"] >= 0
    assert timing["ue5_first_frame_after_tts_ms"] >= timing["face_start_after_tts_ms"]
    assert "face_stitch_enabled" in timing
    assert "face_stitch_overlap_frames" in timing
    assert payload["frame_count"] > 0
    assert payload["channel_count"] == 52


def _run_ws_turn(app, pcm: bytes) -> tuple[list[dict[str, object]], list[bytes]]:
    session_id = uuid4()
    turn_id = uuid4()
    sequence = itertools.count(1)
    events: list[dict[str, object]] = []
    binaries: list[bytes] = []
    with TestClient(app) as client:
        with client.websocket_connect("/pipeline/stream") as ws:
            ws.send_json(_client_event("client.session.start", session_id, None, next(sequence), {}))
            events.append(ws.receive_json())
            ws.send_json(_client_event("client.audio.start", session_id, turn_id, next(sequence), {}))
            for chunk in _pcm_chunks(pcm, chunk_ms=40):
                ws.send_json(
                    _client_event(
                        "client.audio.chunk",
                        session_id,
                        turn_id,
                        next(sequence),
                        {
                            "byte_length": len(chunk),
                            "duration_ms": int(len(chunk) / 2 / 16000 * 1000),
                        },
                    )
                )
                ws.send_bytes(chunk)
            ws.send_json(
                _client_event(
                    "client.audio.end",
                    session_id,
                    turn_id,
                    next(sequence),
                    {"reason": "client_end"},
                )
            )
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] == "server.tts.audio":
                    binaries.append(ws.receive_bytes())
                if event["type"] in TERMINAL_TYPES:
                    return events, binaries


def _pcm_chunks(pcm: bytes, *, chunk_ms: int):
    bytes_per_ms = 16000 * 2 // 1000
    chunk_size = bytes_per_ms * chunk_ms
    minimum_size = bytes_per_ms * 20
    for index in range(0, len(pcm), chunk_size):
        chunk = pcm[index : index + chunk_size]
        if len(chunk) < minimum_size:
            chunk = chunk + b"\x00" * (minimum_size - len(chunk))
        yield chunk


def _client_event(event_type: str, session_id, turn_id, sequence: int, payload: dict[str, object]) -> dict[str, object]:
    return {
        "protocol": "bionic-head-stream-v1",
        "type": event_type,
        "event_id": str(uuid4()),
        "session_id": str(session_id),
        "turn_id": str(turn_id) if turn_id is not None else None,
        "sequence": sequence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
