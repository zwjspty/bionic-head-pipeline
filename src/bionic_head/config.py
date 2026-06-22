from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from bionic_head.domain.models import Emotion


class ServerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8000
    max_active_sessions: int = Field(default=1, ge=1)


class StreamSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    silence_timeout_ms: int = Field(default=1000, ge=1)
    max_turn_duration_seconds: int = Field(default=30, ge=1)
    silence_rms_threshold: float = Field(default=0.01, ge=0.0)
    input_sample_rate: Literal[16000] = 16000
    input_channels: Literal[1] = 1
    input_sample_width_bytes: Literal[2] = 2
    sentence_max_chars: int = Field(default=80, ge=1)
    sentence_max_wait_ms: int = Field(default=500, ge=1)


class RetentionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_runs: int = Field(default=100, ge=1)


class LimitsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    morpheus_max_concurrency: int = Field(default=1, ge=1)
    subprocess_terminate_grace_seconds: int = Field(default=2, ge=1)


class AdapterSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "mock"
    timeout_seconds: float = Field(default=5, gt=0)


class AdaptersSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asr: AdapterSettings = Field(default_factory=AdapterSettings)
    llm: AdapterSettings = Field(default_factory=AdapterSettings)
    tts: AdapterSettings = Field(default_factory=AdapterSettings)
    audio2face: AdapterSettings = Field(default_factory=AdapterSettings)
    ue5: AdapterSettings = Field(default_factory=AdapterSettings)


class LatencySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asr: int = Field(default=0, ge=0)
    llm_first_token: int = Field(default=0, ge=0)
    llm_token: int = Field(default=0, ge=0)
    tts: int = Field(default=0, ge=0)
    face: int = Field(default=0, ge=0)


class MockSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latency_ms: LatencySettings = Field(default_factory=LatencySettings)
    fail_stage: str | None = None
    timeout_stage: str | None = None
    asr_text: str = "你好"
    reply: str = "你好！很高兴见到你。"
    emotion: Emotion = Emotion.FRIENDLY
    intensity: float = Field(default=0.8, ge=0.0, le=1.0)


class StorageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path = Path("data")


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerSettings = Field(default_factory=ServerSettings)
    stream: StreamSettings = Field(default_factory=StreamSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)
    limits: LimitsSettings = Field(default_factory=LimitsSettings)
    adapters: AdaptersSettings = Field(default_factory=AdaptersSettings)
    mock: MockSettings = Field(default_factory=MockSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)


def load_settings(path: Path) -> AppSettings:
    return AppSettings.model_validate_json(path.read_text(encoding="utf-8"))
