from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import itertools
import json

from fastapi.testclient import TestClient


TERMINAL_TYPES = {"server.pipeline.done", "server.pipeline.error", "server.turn.cancelled"}


def test_websocket_mock_turn_reaches_done(app, speech_pcm) -> None:
    session_id = uuid4()
    turn_id = uuid4()
    sequence = itertools.count(1)
    with TestClient(app).websocket_connect("/pipeline/stream") as ws:
        ws.send_json(client_event("client.session.start", session_id, None, next(sequence), {}))
        ready = ws.receive_json()
        ws.send_json(client_event("client.audio.start", session_id, turn_id, next(sequence), {}))
        ws.send_json(
            client_event(
                "client.audio.chunk",
                session_id,
                turn_id,
                next(sequence),
                {"byte_length": len(speech_pcm), "duration_ms": 100},
            )
        )
        ws.send_bytes(speech_pcm)
        ws.send_json(client_event("client.audio.end", session_id, turn_id, next(sequence), {}))
        events, binaries = receive_until_terminal(ws)

    assert ready["type"] == "server.session.ready"
    assert events[-1]["type"] == "server.pipeline.done"
    assert binaries
    assert strictly_increasing_sequences([ready, *events])


def test_chunk_metadata_followed_by_json_is_protocol_violation(app, speech_pcm) -> None:
    session_id = uuid4()
    turn_id = uuid4()
    sequence = itertools.count(1)
    with TestClient(app).websocket_connect("/pipeline/stream") as ws:
        ws.send_json(client_event("client.session.start", session_id, None, next(sequence), {}))
        ws.receive_json()
        ws.send_json(client_event("client.audio.start", session_id, turn_id, next(sequence), {}))
        ws.send_json(
            client_event(
                "client.audio.chunk",
                session_id,
                turn_id,
                next(sequence),
                {"byte_length": len(speech_pcm), "duration_ms": 100},
            )
        )
        ws.send_json(client_event("client.audio.end", session_id, turn_id, next(sequence), {}))
        events, _ = receive_until_terminal(ws)

    assert events[-1]["type"] == "server.pipeline.error"
    assert events[-1]["payload"]["error"]["code"] == "protocol_violation"


def test_binary_without_pending_metadata_is_protocol_violation(app, speech_pcm) -> None:
    session_id = uuid4()
    turn_id = uuid4()
    sequence = itertools.count(1)
    with TestClient(app).websocket_connect("/pipeline/stream") as ws:
        ws.send_json(client_event("client.session.start", session_id, None, next(sequence), {}))
        ws.receive_json()
        ws.send_json(client_event("client.audio.start", session_id, turn_id, next(sequence), {}))
        ws.send_bytes(speech_pcm)
        events, _ = receive_until_terminal(ws)

    assert events[-1]["type"] == "server.pipeline.error"
    assert events[-1]["payload"]["error"]["code"] == "protocol_violation"


def test_binary_length_mismatch_is_protocol_violation(app, speech_pcm) -> None:
    session_id = uuid4()
    turn_id = uuid4()
    sequence = itertools.count(1)
    with TestClient(app).websocket_connect("/pipeline/stream") as ws:
        ws.send_json(client_event("client.session.start", session_id, None, next(sequence), {}))
        ws.receive_json()
        ws.send_json(client_event("client.audio.start", session_id, turn_id, next(sequence), {}))
        ws.send_json(
            client_event(
                "client.audio.chunk",
                session_id,
                turn_id,
                next(sequence),
                {"byte_length": len(speech_pcm) + 2, "duration_ms": 100},
            )
        )
        ws.send_bytes(speech_pcm)
        events, _ = receive_until_terminal(ws)

    assert events[-1]["type"] == "server.pipeline.error"
    assert events[-1]["payload"]["error"]["code"] == "protocol_violation"


def test_explicit_cancel_emits_cancelled(app) -> None:
    session_id = uuid4()
    turn_id = uuid4()
    sequence = itertools.count(1)
    with TestClient(app).websocket_connect("/pipeline/stream") as ws:
        ws.send_json(client_event("client.session.start", session_id, None, next(sequence), {}))
        ws.receive_json()
        ws.send_json(client_event("client.audio.start", session_id, turn_id, next(sequence), {}))
        ws.send_json(client_event("client.turn.cancel", session_id, turn_id, next(sequence), {}))
        events, _ = receive_until_terminal(ws)

    assert [event["type"] for event in events if event["type"] in TERMINAL_TYPES] == ["server.turn.cancelled"]


def test_second_simultaneous_session_receives_limit_error(app) -> None:
    first_session = uuid4()
    second_session = uuid4()
    with TestClient(app).websocket_connect("/pipeline/stream") as ws1:
        ws1.send_json(client_event("client.session.start", first_session, None, 1, {}))
        assert ws1.receive_json()["type"] == "server.session.ready"
        with TestClient(app).websocket_connect("/pipeline/stream") as ws2:
            ws2.send_json(client_event("client.session.start", second_session, None, 1, {}))
            response = ws2.receive_json()

    assert response["type"] == "server.pipeline.error"
    assert response["payload"]["error"]["code"] == "session_limit_reached"


def client_event(event_type: str, session_id, turn_id, sequence: int, payload: dict[str, object]) -> dict[str, object]:
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


def receive_until_terminal(ws):
    events = []
    binaries = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] == "server.tts.audio":
            binaries.append(ws.receive_bytes())
        if event["type"] in TERMINAL_TYPES:
            return events, binaries


def strictly_increasing_sequences(events: list[dict[str, object]]) -> bool:
    sequences = [int(event["sequence"]) for event in events]
    return sequences == sorted(sequences) and len(sequences) == len(set(sequences))
