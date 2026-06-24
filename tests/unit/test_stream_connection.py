from __future__ import annotations

import itertools
import asyncio
from array import array
from datetime import datetime, timezone
import sys
from uuid import uuid4

from fastapi.testclient import TestClient

from bionic_head.adapters.registry import AdapterRegistry
from bionic_head.api.app import create_app
from bionic_head.core.state import TurnHandle, TurnState, TurnStateMachine
from bionic_head.protocol.connection import StreamConnection
from bionic_head.protocol.events import EventEnvelope, EventFactory


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


def test_session_start_can_prewarm_emotalk_sidecar_before_ready(mock_settings, tmp_path) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.storage.root = tmp_path / "stream-data"
    settings.adapters.audio2face.provider = "emotalk_sidecar"
    settings.providers.emotalk_sidecar.sidecar_command = [
        sys.executable,
        "-m",
        "bionic_head.emotalk_fake_sidecar",
    ]
    settings.providers.emotalk_sidecar.prewarm_on_startup = False
    settings.providers.emotalk_sidecar.prewarm_on_session_start = True
    app = create_app(settings)
    session_id = uuid4()

    with TestClient(app) as client:
        assert client.app.state.startup_diagnostics == []
        with client.websocket_connect("/pipeline/stream") as ws:
            ws.send_json(_client_event("client.session.start", session_id, None, 1, {}))
            ready = ws.receive_json()

            assert ready["type"] == "server.session.ready"
            assert client.app.state.container.registry.audio2face.process_start_count == 1
            assert client.app.state.container.registry.audio2face.call_count == 0


async def test_barge_in_audio_start_does_not_stop_playback_before_speech(
    mock_settings,
    tmp_path,
) -> None:
    connection, old_turn_id = _thinking_connection(mock_settings, tmp_path)
    new_turn_id = uuid4()

    await connection._start_turn(
        _client_envelope("client.audio.start", connection.session_id, new_turn_id, 1, {})
    )

    assert old_turn_id == connection.current_turn.turn_id
    assert not _sent_type(connection.websocket, "server.playback.stop")


async def test_high_rms_barge_in_triggers_playback_stop_after_threshold(
    mock_settings,
    tmp_path,
) -> None:
    connection, old_turn_id = _thinking_connection(mock_settings, tmp_path)
    new_turn_id = uuid4()
    speech = _pcm_chunk(amplitude=5000, duration_ms=40)

    await connection._start_turn(
        _client_envelope("client.audio.start", connection.session_id, new_turn_id, 1, {})
    )
    await _send_candidate_chunk(connection, new_turn_id, sequence=2, payload=speech, duration_ms=40)
    assert not _sent_type(connection.websocket, "server.playback.stop")

    await _send_candidate_chunk(connection, new_turn_id, sequence=3, payload=speech, duration_ms=40)

    sent_types = [event["type"] for event in connection.websocket.sent_json]
    assert sent_types[-3:] == [
        "server.playback.stop",
        "server.turn.cancelled",
        "server.state",
    ]
    assert connection.websocket.sent_json[-3]["turn_id"] == str(old_turn_id)
    assert connection.websocket.sent_json[-1]["turn_id"] == str(new_turn_id)
    assert connection.websocket.sent_json[-1]["generation_epoch"] == 1
    assert connection.current_turn.turn_id == new_turn_id
    assert connection.state_machine.state is TurnState.LISTENING
    await _cleanup_connection(connection)


async def test_low_rms_barge_in_candidate_does_not_interrupt(
    mock_settings,
    tmp_path,
) -> None:
    connection, old_turn_id = _thinking_connection(mock_settings, tmp_path)
    new_turn_id = uuid4()
    quiet = _pcm_chunk(amplitude=100, duration_ms=40)

    await connection._start_turn(
        _client_envelope("client.audio.start", connection.session_id, new_turn_id, 1, {})
    )
    await _send_candidate_chunk(connection, new_turn_id, sequence=2, payload=quiet, duration_ms=40)
    await _send_candidate_chunk(connection, new_turn_id, sequence=3, payload=quiet, duration_ms=40)

    assert connection.current_turn.turn_id == old_turn_id
    assert not _sent_type(connection.websocket, "server.playback.stop")


async def test_turn_cancel_schedules_best_effort_provider_cancel_without_blocking_stop(
    mock_settings,
    tmp_path,
) -> None:
    connection, _old_turn_id = _thinking_connection(mock_settings, tmp_path)
    cancel_log: list[str] = []
    connection.container.registry = AdapterRegistry(
        asr=_CancelRecordingAdapter("asr", cancel_log),
        llm=_CancelRecordingAdapter("llm", cancel_log),
        tts=_CancelRecordingAdapter("tts", cancel_log, fail=True),
        audio2face=_CancelRecordingAdapter("audio2face", cancel_log),
        ue5=_CancelRecordingAdapter("ue5", cancel_log),
    )

    await connection._cancel_current_turn(emit=True)

    sent_types = [event["type"] for event in connection.websocket.sent_json]
    assert sent_types[-2:] == ["server.playback.stop", "server.turn.cancelled"]
    assert len(cancel_log) < 5

    provider_cancel_tasks = list(connection._provider_cancel_tasks)
    assert provider_cancel_tasks
    await asyncio.gather(*provider_cancel_tasks)

    assert cancel_log == ["asr", "llm", "tts", "audio2face", "ue5"]
    results = connection.container.last_provider_cancel_results
    assert [result.stage for result in results] == ["asr", "llm", "tts", "audio2face", "ue5"]
    assert [result.ok for result in results] == [True, True, False, True, True]
    tts_result = results[2]
    assert tts_result.provider == "tts"
    assert tts_result.error_code == "provider_failed"


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent_json: list[dict[str, object]] = []
        self.sent_bytes: list[bytes] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent_json.append(payload)

    async def send_bytes(self, payload: bytes) -> None:
        self.sent_bytes.append(payload)


class _CancelRecordingAdapter:
    def __init__(
        self,
        stage: str,
        log: list[str],
        *,
        fail: bool = False,
    ) -> None:
        self.name = stage
        self._stage = stage
        self._log = log
        self._fail = fail

    async def cancel(self, turn_id) -> None:
        await asyncio.sleep(0.05)
        self._log.append(self._stage)
        if self._fail:
            raise RuntimeError(f"{self._stage} cancel failed")


def _thinking_connection(mock_settings, tmp_path):
    settings = mock_settings.model_copy(deep=True)
    settings.storage.root = tmp_path / "stream-data"
    app = create_app(settings)
    websocket = _FakeWebSocket()
    connection = StreamConnection(websocket, app.state.container)
    session_id = uuid4()
    old_turn_id = uuid4()
    connection.session_id = session_id
    connection.event_factory = EventFactory(
        session_id=session_id,
        generation_epoch_getter=lambda: connection.generation_epoch,
    )
    connection.current_turn = TurnHandle(
        session_id=session_id,
        turn_id=old_turn_id,
        generation_epoch=connection.generation_epoch,
        generation_epoch_getter=lambda: connection.generation_epoch,
    )
    connection.state_machine = TurnStateMachine()
    connection.state_machine.transition(TurnState.LISTENING)
    connection.state_machine.transition(TurnState.THINKING)
    return connection, old_turn_id


async def _cleanup_connection(connection: StreamConnection) -> None:
    if connection.watchdog_task is not None:
        connection.watchdog_task.cancel()
        await asyncio.sleep(0)


async def _send_candidate_chunk(
    connection: StreamConnection,
    turn_id,
    *,
    sequence: int,
    payload: bytes,
    duration_ms: int,
) -> None:
    await connection._handle_audio_chunk_metadata(
        _client_envelope(
            "client.audio.chunk",
            connection.session_id,
            turn_id,
            sequence,
            {"byte_length": len(payload), "duration_ms": duration_ms},
        )
    )
    await connection._handle_binary(payload)


def _pcm_chunk(*, amplitude: int, duration_ms: int) -> bytes:
    sample_count = 16000 * duration_ms // 1000
    return array("h", [amplitude, -amplitude] * (sample_count // 2)).tobytes()


def _sent_type(websocket: _FakeWebSocket, event_type: str) -> bool:
    return any(event["type"] == event_type for event in websocket.sent_json)


def _client_envelope(event_type: str, session_id, turn_id, sequence: int, payload: dict[str, object]) -> EventEnvelope:
    return EventEnvelope.model_validate(
        _client_event(event_type, session_id, turn_id, sequence, payload)
    )


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
