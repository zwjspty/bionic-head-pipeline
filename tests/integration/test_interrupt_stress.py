from __future__ import annotations

from datetime import datetime, timezone
from itertools import count
from uuid import uuid4

from fastapi.testclient import TestClient

from bionic_head.api.app import create_app
from scripts.interrupt_stress import summarize_events


def test_mock_websocket_repeated_interrupts_do_not_leak_old_face_events(
    mock_settings,
    tmp_path,
    speech_pcm,
) -> None:
    repeats = 5
    settings = mock_settings.model_copy(deep=True)
    settings.storage.root = tmp_path / "interrupt-stress-data"
    settings.mock.latency_ms.face = 100
    app = create_app(settings)
    session_id = uuid4()
    sequence = count(1)
    events: list[dict[str, object]] = []

    with TestClient(app).websocket_connect("/pipeline/stream") as ws:
        ws.send_json(_client_event("client.session.start", session_id, None, next(sequence), {}))
        _receive_until(ws, events, lambda event: event["type"] == "server.session.ready")

        current_turn = uuid4()
        _send_full_turn(ws, session_id, current_turn, sequence, speech_pcm)
        _receive_until(
            ws,
            events,
            lambda event: event["type"] == "server.tts.audio"
            and event["turn_id"] == str(current_turn),
        )

        for _ in range(repeats):
            next_turn = uuid4()
            ws.send_json(_client_event("client.audio.start", session_id, next_turn, next(sequence), {}))
            _send_chunk(ws, session_id, next_turn, sequence, speech_pcm, duration_ms=100)
            _receive_until(
                ws,
                events,
                lambda event: event["type"] == "server.state"
                and event["turn_id"] == str(next_turn),
            )
            current_turn = next_turn
            ws.send_json(
                _client_event(
                    "client.audio.end",
                    session_id,
                    current_turn,
                    next(sequence),
                    {"reason": "client_end"},
                )
            )
            _receive_until(
                ws,
                events,
                lambda event: event["type"] == "server.tts.audio"
                and event["turn_id"] == str(current_turn),
            )

    summary = summarize_events(events)
    assert summary["playback_stop_count"] == repeats
    assert summary["old_turn_face_leak_count"] == 0
    assert summary["strictly_increasing_sequences"] is True


def _send_full_turn(ws, session_id, turn_id, sequence, speech_pcm: bytes) -> None:
    ws.send_json(_client_event("client.audio.start", session_id, turn_id, next(sequence), {}))
    _send_chunk(ws, session_id, turn_id, sequence, speech_pcm, duration_ms=100)
    ws.send_json(
        _client_event(
            "client.audio.end",
            session_id,
            turn_id,
            next(sequence),
            {"reason": "client_end"},
        )
    )


def _send_chunk(ws, session_id, turn_id, sequence, payload: bytes, *, duration_ms: int) -> None:
    ws.send_json(
        _client_event(
            "client.audio.chunk",
            session_id,
            turn_id,
            next(sequence),
            {"byte_length": len(payload), "duration_ms": duration_ms},
        )
    )
    ws.send_bytes(payload)


def _receive_until(ws, events: list[dict[str, object]], predicate) -> dict[str, object]:
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] == "server.tts.audio":
            ws.receive_bytes()
        if predicate(event):
            return event


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
