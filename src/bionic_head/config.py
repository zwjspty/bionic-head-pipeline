from __future__ import annotations

from pathlib import Path

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator
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
    sentence_min_chars: int = Field(default=8, ge=1)
    sentence_max_chars: int = Field(default=24, ge=1)
    sentence_max_wait_ms: int = Field(default=500, ge=1)


class VadSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Literal["rms"] = "rms"
    interrupt_min_speech_ms: int = Field(default=80, ge=1)
    interrupt_rms_threshold: float = Field(default=0.02, ge=0.0)


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


class CommandSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executable: str = ""
    args: list[str] = Field(default_factory=list)
    cwd: Path | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)


class FasterWhisperSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "zh"


class OllamaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: AnyHttpUrl = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b"
    timeout_seconds: float = Field(default=120.0, gt=0)
    keep_alive: str | int | None = "30m"
    num_ctx: int | None = Field(default=2048, ge=1)
    num_predict: int | None = Field(default=96, ge=1)
    temperature: float | None = Field(default=0.3, ge=0.0)
    prewarm: bool = True


class PiperSettings(CommandSettings):
    runtime: Literal["cli", "python"] = "cli"
    model_path: Path | None = None
    config_path: Path | None = None
    use_cuda: bool = False
    speaker_id: int | None = None
    length_scale: float | None = Field(default=None, gt=0)
    noise_scale: float | None = Field(default=None, ge=0)
    noise_w_scale: float | None = Field(default=None, ge=0)
    normalize_audio: bool = True
    volume: float = Field(default=1.0, gt=0)

    @field_validator("model_path", "config_path", mode="before")
    @classmethod
    def _empty_path_is_unknown(cls, value: object) -> object:
        if value == "":
            return None
        return value


class MorpheusSettings(CommandSettings):
    output_npy_glob: str = "*.npy"
    output_json_glob: str = "*.json"


class EmoTalkSettings(CommandSettings):
    output_npy_glob: str = "*.npy"
    output_json_glob: str = "*.json"


class EmoTalkSidecarSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sidecar_command: list[str] = Field(default_factory=list)
    sample_rate: Literal[16000] = 16000
    fps: int = Field(default=30, ge=1)
    timeout_seconds: float = Field(default=10.0, gt=0)
    channel_count: Literal[52] = 52
    output_npy_name: str = "face.npy"


class ProvidersSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faster_whisper: FasterWhisperSettings = Field(default_factory=FasterWhisperSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    piper: PiperSettings = Field(default_factory=PiperSettings)
    morpheus: MorpheusSettings = Field(default_factory=MorpheusSettings)
    emotalk: EmoTalkSettings = Field(default_factory=EmoTalkSettings)
    emotalk_sidecar: EmoTalkSidecarSettings = Field(default_factory=EmoTalkSidecarSettings)


class StorageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path = Path("data")


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerSettings = Field(default_factory=ServerSettings)
    stream: StreamSettings = Field(default_factory=StreamSettings)
    vad: VadSettings = Field(default_factory=VadSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)
    limits: LimitsSettings = Field(default_factory=LimitsSettings)
    adapters: AdaptersSettings = Field(default_factory=AdaptersSettings)
    mock: MockSettings = Field(default_factory=MockSettings)
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)


def load_settings(path: Path) -> AppSettings:
    return AppSettings.model_validate_json(path.read_text(encoding="utf-8"))
