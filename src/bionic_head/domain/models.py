from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bionic_head.core.cancellation import CancellationToken

try:  # Python 3.10 fallback for local verification
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        pass


class Emotion(StrEnum):
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    THINKING = "thinking"
    CALM = "calm"


def _validate_weight_vector(values: list[float], *, label: str) -> list[float]:
    if len(values) != 52:
        raise ValueError(f"{label} must contain exactly 52 values")
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{label} values must be finite numbers")
    return values


class AudioStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_rate: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    duration_seconds: float
    rms: float
    peak: float


class AudioFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    sample_width_bytes: Literal[2] = 2


class ASRResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    language: str
    confidence: float | None = None
    audio: AudioStats


class LLMResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str
    emotion: Emotion
    intensity: float = Field(ge=0.0, le=1.0)


class LLMEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["token", "final"]
    text: str = ""
    result: LLMResult | None = None


class AudioArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    sample_rate: int
    channels: int
    sample_width_bytes: int
    duration_seconds: float
    byte_length: int


class FaceArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path | None = None
    frames: list[list[float]]
    fps: int = 30
    channel_count: int = 52
    frame_count: int
    auxiliary_paths: list[Path] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)

    @field_validator("frames")
    @classmethod
    def validate_frames(cls, value: list[list[float]]) -> list[list[float]]:
        for frame in value:
            _validate_weight_vector(frame, label="face frame")
        return value

    @model_validator(mode="after")
    def validate_frame_count(self) -> "FaceArtifact":
        if self.frame_count != len(self.frames):
            raise ValueError("frame_count must match the number of frames")
        return self


class UE5Frame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int
    time_seconds: float
    weights: list[float]

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: list[float]) -> list[float]:
        return _validate_weight_vector(value, label="UE5 frame weights")


class UE5Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: Literal["bionic-head-ue5-v1"] = "bionic-head-ue5-v1"
    format: Literal["morpheus_52_raw"] = "morpheus_52_raw"
    fps: int = 30
    channel_count: Literal[52] = 52
    channels: list[str]
    frame_count: int
    frames: list[UE5Frame]

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: list[str]) -> list[str]:
        if len(value) != 52:
            raise ValueError("channels must contain exactly 52 names")
        return value

    @model_validator(mode="after")
    def validate_frame_count(self) -> "UE5Payload":
        if self.frame_count != len(self.frames):
            raise ValueError("frame_count must match the number of frames")
        return self


class DiagnosticResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: str
    provider: str
    available: bool
    latency_ms: float
    message: str


class PipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    turn_id: UUID
    asr: ASRResult
    llm: LLMResult
    audio: AudioArtifact
    face: FaceArtifact
    ue5: UE5Payload
    timeline: dict[str, object]


@dataclass(frozen=True)
class TurnContext:
    session_id: UUID
    turn_id: UUID
    artifact_dir: Path
    cancellation: CancellationToken
    generation_epoch: int = 0
