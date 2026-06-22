from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import itertools

import pytest
from fastapi.testclient import TestClient


TERMINAL_TYPES = {"server.pipeline.done", "server.pipeline.error", "server.turn.cancelled"}


@pytest.mark.integration
def test_real_offline_pipeline(real_app, chinese_wav: Path) -> None:
    response = post_audio(real_app, chinese_wav)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["asr"]["text"]
    assert body["llm"]["reply"]
    assert body["audio"]["duration_seconds"] > 0
    assert body["face"]["channel_count"] == 52


@pytest.mark.integration
def test_real_pseudo_streaming_pipeline(real_app, chinese_pcm: bytes) -> None:
    events, binaries = run_ws_turn(real_app, chinese_pcm)

    assert terminal_types(events) == ["server.pipeline.done"]
    assert any(event["type"] == "server.segment.ready" for event in events)
    assert binaries


def post_audio(app, wav_path: Path):
    with TestClient(app) as client:
        return client.post(
            "/pipeline/audio",
            files={"audio": ("input.wav", wav_path.read_bytes(), "audio/wav")},
        )


def run_ws_turn(app, pcm: bytes):
    session_id = uuid4()
    turn_id = uuid4()
    sequence = itertools.count(1)
    events = []
    binaries = []
    with TestClient(app).websocket_connect("/pipeline/stream") as ws:
        ws.send_json(client_event("client.session.start", session_id, None, next(sequence), {}))
        ready = ws.receive_json()
        events.append(ready)
        ws.send_json(client_event("client.audio.start", session_id, turn_id, next(sequence), {}))
        for chunk in pcm_chunks(pcm, chunk_ms=40):
            ws.send_json(
                client_event(
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
            client_event(
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
                break
    return events, binaries


def pcm_chunks(pcm: bytes, *, chunk_ms: int):
    bytes_per_ms = 16000 * 2 // 1000
    chunk_size = bytes_per_ms * chunk_ms
    minimum_size = bytes_per_ms * 20
    for index in range(0, len(pcm), chunk_size):
        chunk = pcm[index : index + chunk_size]
        if len(chunk) < minimum_size:
            chunk = chunk + b"\x00" * (minimum_size - len(chunk))
        yield chunk


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


def terminal_types(events: list[dict[str, object]]) -> list[str]:
    return [event["type"] for event in events if event["type"] in TERMINAL_TYPES]
