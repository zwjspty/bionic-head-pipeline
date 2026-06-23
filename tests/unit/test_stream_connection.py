from __future__ import annotations

import itertools
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from bionic_head.api.app import create_app


def test_client_cancel_emits_playback_stop_before_turn_cancelled(mock_settings, tmp_path) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.storage.root = tmp_path / "stream-data"
    app = create_app(settings)
    client = TestClient(app)
    session_id = uuid4()
    turn_id = uuid4()
    sequence = itertools.count(1)

    with client.websocket_connect("/pipeline/stream") as ws:
        ws.send_json(_client_event("client.session.start", session_id, None, next(sequence), {}))
        assert ws.receive_json()["type"] == "server.session.ready"

        ws.send_json(_client_event("client.audio.start", session_id, turn_id, next(sequence), {}))
        state = ws.receive_json()
        assert state["type"] == "server.state"
        assert state["generation_epoch"] == 0

        ws.send_json(
            _client_event(
                "client.turn.cancel",
                session_id,
                turn_id,
                next(sequence),
                {"reason": "test_cancel"},
            )
        )

        playback_stop = ws.receive_json()
        cancelled = ws.receive_json()

    assert playback_stop["type"] == "server.playback.stop"
    assert playback_stop["turn_id"] == str(turn_id)
    assert playback_stop["generation_epoch"] == 1
    assert playback_stop["payload"]["generation_epoch"] == 1
    assert cancelled["type"] == "server.turn.cancelled"
    assert cancelled["generation_epoch"] == 1


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
