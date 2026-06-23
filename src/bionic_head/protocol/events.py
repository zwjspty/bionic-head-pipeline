from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from itertools import count
from collections.abc import Callable
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bionic_head.domain.errors import ErrorCode, PipelineException

try:  # Python 3.10 fallback for local verification
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        pass


class EventType(StrEnum):
    CLIENT_SESSION_START = "client.session.start"
    CLIENT_AUDIO_START = "client.audio.start"
    CLIENT_AUDIO_CHUNK = "client.audio.chunk"
    CLIENT_AUDIO_END = "client.audio.end"
    CLIENT_TURN_CANCEL = "client.turn.cancel"
    CLIENT_PING = "client.ping"

    SERVER_SESSION_READY = "server.session.ready"
    SERVER_STATE = "server.state"
    SERVER_ASR_FINAL = "server.asr.final"
    SERVER_LLM_TOKEN = "server.llm.token"
    SERVER_LLM_CHUNK = "server.llm.chunk"
    SERVER_TTS_AUDIO = "server.tts.audio"
    SERVER_FACE_FRAMES = "server.face.frames"
    SERVER_UE5_FRAMES = "server.ue5.frames"
    SERVER_SEGMENT_READY = "server.segment.ready"
    SERVER_PLAYBACK_STOP = "server.playback.stop"
    SERVER_TURN_CANCELLED = "server.turn.cancelled"
    SERVER_PIPELINE_DONE = "server.pipeline.done"
    SERVER_PIPELINE_ERROR = "server.pipeline.error"
    SERVER_PONG = "server.pong"


class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: Literal["bionic-head-stream-v1"] = "bionic-head-stream-v1"
    type: EventType
    event_id: UUID
    session_id: UUID
    turn_id: UUID | None
    sequence: int = Field(ge=1)
    generation_epoch: int | None = Field(default=None, ge=0)
    timestamp: datetime
    payload: dict[str, object]

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_turn_scope(self) -> "EventEnvelope":
        session_level = {
            EventType.CLIENT_SESSION_START,
            EventType.SERVER_SESSION_READY,
            EventType.CLIENT_PING,
            EventType.SERVER_PONG,
            EventType.SERVER_PIPELINE_ERROR,
        }
        if self.turn_id is None and self.type not in session_level:
            raise ValueError(f"{self.type.value} requires turn_id")
        return self


class _StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClientSessionStartPayload(_StrictPayload):
    client_name: str | None = None


class ClientAudioStartPayload(_StrictPayload):
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    sample_width_bytes: Literal[2] = 2


class ClientAudioChunkPayload(_StrictPayload):
    byte_length: int = Field(ge=1)
    duration_ms: int = Field(ge=1)


class ClientAudioEndPayload(_StrictPayload):
    reason: Literal["client_end", "silence_timeout", "max_duration"] = "client_end"


class ClientCancelPayload(_StrictPayload):
    reason: str | None = None


class ClientPingPayload(_StrictPayload):
    ping_id: str | None = None


class EventFactory:
    def __init__(
        self,
        *,
        session_id: UUID,
        generation_epoch_getter: Callable[[], int] | None = None,
    ) -> None:
        self.session_id = session_id
        self._sequence = count(start=1)
        self._generation_epoch_getter = generation_epoch_getter

    def server(
        self,
        event_type: EventType | str,
        turn_id: UUID | None,
        payload: dict[str, object],
    ) -> EventEnvelope:
        generation_epoch = self._generation_epoch()
        return EventEnvelope(
            type=EventType(event_type),
            event_id=uuid4(),
            session_id=self.session_id,
            turn_id=turn_id,
            sequence=next(self._sequence),
            generation_epoch=generation_epoch,
            timestamp=datetime.now(timezone.utc),
            payload={
                "session_id": self.session_id,
                "turn_id": turn_id,
                "generation_epoch": generation_epoch,
                **payload,
            },
        )

    def _generation_epoch(self) -> int:
        if self._generation_epoch_getter is None:
            return 0
        return self._generation_epoch_getter()


class ClientSequenceValidator:
    def __init__(self) -> None:
        self._next_sequence = 1

    def validate(self, envelope: EventEnvelope) -> None:
        if not envelope.type.value.startswith("client."):
            raise self._protocol_violation("Expected a client event")
        if envelope.sequence != self._next_sequence:
            raise self._protocol_violation(
                f"Expected client sequence {self._next_sequence}, got {envelope.sequence}"
            )
        self._next_sequence += 1

    def _protocol_violation(self, message: str) -> PipelineException:
        return PipelineException(
            code=ErrorCode.PROTOCOL_VIOLATION,
            stage="websocket",
            provider=None,
            retryable=False,
            message=message,
        )
