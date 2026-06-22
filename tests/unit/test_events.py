from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import asyncio

import pytest
from pydantic import ValidationError

from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.protocol.events import (
    ClientAudioChunkPayload,
    ClientSequenceValidator,
    EventEnvelope,
    EventFactory,
    EventType,
)


def test_event_sequence_is_monotonic() -> None:
    factory = EventFactory(session_id=uuid4())
    turn_id = uuid4()

    first = factory.server("server.state", turn_id, {"state": "IDLE"})
    second = factory.server("server.pong", turn_id, {})

    assert first.protocol == "bionic-head-stream-v1"
    assert (first.sequence, second.sequence) == (1, 2)
    assert first.timestamp.tzinfo is not None


@pytest.mark.asyncio
async def test_concurrent_server_event_sequences_are_unique() -> None:
    factory = EventFactory(session_id=uuid4())
    turn_id = uuid4()

    async def build_event(index: int):
        await asyncio.sleep(0)
        return factory.server(EventType.SERVER_PONG, turn_id, {"index": index})

    events = await asyncio.gather(*(build_event(index) for index in range(10)))

    assert sorted(event.sequence for event in events) == list(range(1, 11))


def test_client_payload_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ClientAudioChunkPayload(byte_length=3200, duration_ms=100, extra=True)


def test_client_sequence_validator_rejects_gaps() -> None:
    session_id = uuid4()
    validator = ClientSequenceValidator()

    validator.validate(_client_event(session_id, sequence=1))
    validator.validate(_client_event(session_id, sequence=2))
    with pytest.raises(PipelineException) as raised:
        validator.validate(_client_event(session_id, sequence=4))

    assert raised.value.code is ErrorCode.PROTOCOL_VIOLATION


def test_turn_events_require_turn_id() -> None:
    with pytest.raises(ValidationError):
        EventEnvelope(
            protocol="bionic-head-stream-v1",
            type="client.audio.start",
            event_id=uuid4(),
            session_id=uuid4(),
            turn_id=None,
            sequence=1,
            timestamp=datetime.now(timezone.utc),
            payload={},
        )


def _client_event(session_id, *, sequence: int) -> EventEnvelope:
    return EventEnvelope(
        protocol="bionic-head-stream-v1",
        type="client.ping",
        event_id=uuid4(),
        session_id=session_id,
        turn_id=None,
        sequence=sequence,
        timestamp=datetime.now(timezone.utc),
        payload={},
    )
