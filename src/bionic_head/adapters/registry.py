from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar
from uuid import UUID
import asyncio

from bionic_head.adapters.mock import (
    MockASRAdapter,
    MockAudio2FaceAdapter,
    MockLLMAdapter,
    MockTTSAdapter,
    MockUE5Adapter,
    map_mock_runtime_error,
)
from bionic_head.adapters.protocols import (
    ASRAdapter,
    Audio2FaceAdapter,
    LLMAdapter,
    TTSAdapter,
    UE5Adapter,
)
from bionic_head.config import AdapterSettings, AppSettings
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import (
    ASRResult,
    AudioArtifact,
    DiagnosticResult,
    Emotion,
    FaceArtifact,
    LLMEvent,
    LLMResult,
    TurnContext,
    UE5Payload,
)


T = TypeVar("T")


def _provider_unavailable(provider: str) -> PipelineException:
    return PipelineException(
        code=ErrorCode.PROVIDER_UNAVAILABLE,
        stage="startup",
        provider=provider,
        retryable=False,
        message=f"Provider is not available in P0 registry: {provider}",
    )


def _timeout(stage: str, provider: str) -> PipelineException:
    return PipelineException(
        code=ErrorCode.PROVIDER_TIMEOUT,
        stage=stage,
        provider=provider,
        retryable=True,
        message=f"Provider timed out during {stage}",
    )


def _provider_failed(stage: str, provider: str, exc: Exception) -> PipelineException:
    if provider == "mock":
        return map_mock_runtime_error(exc, stage=stage)
    if isinstance(exc, PipelineException):
        return exc
    return PipelineException(
        code=ErrorCode.PROVIDER_FAILED,
        stage=stage,
        provider=provider,
        retryable=True,
        message=f"Provider failed during {stage}",
    )


async def _call_with_timeout(
    call: Callable[[], Awaitable[T]],
    *,
    stage: str,
    provider: str,
    timeout_seconds: float,
) -> T:
    try:
        return await asyncio.wait_for(call(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise _timeout(stage, provider) from exc
    except asyncio.CancelledError:
        raise
    except PipelineException:
        raise
    except Exception as exc:
        raise _provider_failed(stage, provider, exc) from exc


class _BaseWrapper:
    def __init__(self, inner: object, settings: AdapterSettings, stage: str) -> None:
        self._inner = inner
        self._settings = settings
        self._stage = stage

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    @property
    def name(self) -> str:
        return self._inner.name  # type: ignore[attr-defined]

    @property
    def call_count(self) -> int:
        return self._inner.call_count  # type: ignore[attr-defined]

    async def diagnostics(self) -> DiagnosticResult:
        return await _call_with_timeout(
            self._inner.diagnostics,  # type: ignore[attr-defined]
            stage=f"{self._stage}.diagnostics",
            provider=self.name,
            timeout_seconds=self._settings.timeout_seconds,
        )

    async def cancel(self, turn_id: UUID) -> None:
        await _call_with_timeout(
            lambda: self._inner.cancel(turn_id),  # type: ignore[attr-defined]
            stage=f"{self._stage}.cancel",
            provider=self.name,
            timeout_seconds=self._settings.timeout_seconds,
        )


class _ASRWrapper(_BaseWrapper):
    async def transcribe(self, audio_path: Path, context: TurnContext) -> ASRResult:
        return await _call_with_timeout(
            lambda: self._inner.transcribe(audio_path, context),  # type: ignore[attr-defined]
            stage="asr",
            provider=self.name,
            timeout_seconds=self._settings.timeout_seconds,
        )


class _LLMWrapper(_BaseWrapper):
    async def chat(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> LLMResult:
        return await _call_with_timeout(
            lambda: self._inner.chat(text, history, context),  # type: ignore[attr-defined]
            stage="llm",
            provider=self.name,
            timeout_seconds=self._settings.timeout_seconds,
        )

    async def _stream_with_timeout(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> AsyncIterator[LLMEvent]:
        iterator = self._inner.chat_stream(text, history, context)  # type: ignore[attr-defined]
        while True:
            try:
                yield await asyncio.wait_for(
                    iterator.__anext__(),
                    timeout=self._settings.timeout_seconds,
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                raise _timeout("llm", self.name) from exc
            except asyncio.CancelledError:
                raise
            except PipelineException:
                raise
            except Exception as exc:
                raise _provider_failed("llm", self.name, exc) from exc

    def chat_stream(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> AsyncIterator[LLMEvent]:
        return self._stream_with_timeout(text, history, context)


class _TTSWrapper(_BaseWrapper):
    async def synthesize(
        self,
        text: str,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> AudioArtifact:
        return await _call_with_timeout(
            lambda: self._inner.synthesize(text, emotion, intensity, context),  # type: ignore[attr-defined]
            stage="tts",
            provider=self.name,
            timeout_seconds=self._settings.timeout_seconds,
        )


class _Audio2FaceWrapper(_BaseWrapper):
    async def drive(
        self,
        audio: AudioArtifact,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> FaceArtifact:
        return await _call_with_timeout(
            lambda: self._inner.drive(audio, emotion, intensity, context),  # type: ignore[attr-defined]
            stage="audio2face",
            provider=self.name,
            timeout_seconds=self._settings.timeout_seconds,
        )


class _UE5Wrapper(_BaseWrapper):
    async def format(self, face: FaceArtifact, context: TurnContext) -> UE5Payload:
        return await _call_with_timeout(
            lambda: self._inner.format(face, context),  # type: ignore[attr-defined]
            stage="ue5",
            provider=self.name,
            timeout_seconds=self._settings.timeout_seconds,
        )


@dataclass(frozen=True)
class AdapterRegistry:
    asr: ASRAdapter
    llm: LLMAdapter
    tts: TTSAdapter
    audio2face: Audio2FaceAdapter
    ue5: UE5Adapter


def _ensure_mock(settings: AdapterSettings) -> None:
    if settings.provider != "mock":
        raise _provider_unavailable(settings.provider)


def _build_llm(settings: AppSettings) -> LLMAdapter:
    if settings.adapters.llm.provider == "mock":
        return MockLLMAdapter(settings.mock, settings.adapters.llm)
    if settings.adapters.llm.provider == "ollama":
        from bionic_head.adapters.ollama import OllamaLLMAdapter

        return OllamaLLMAdapter(settings.providers.ollama)
    raise _provider_unavailable(settings.adapters.llm.provider)


def _build_asr(settings: AppSettings) -> ASRAdapter:
    if settings.adapters.asr.provider == "mock":
        return MockASRAdapter(settings.mock, settings.adapters.asr)
    if settings.adapters.asr.provider == "faster-whisper":
        from bionic_head.adapters.faster_whisper import FasterWhisperASRAdapter

        return FasterWhisperASRAdapter(settings.providers.faster_whisper)
    raise _provider_unavailable(settings.adapters.asr.provider)


def _build_tts(settings: AppSettings) -> TTSAdapter:
    if settings.adapters.tts.provider == "mock":
        return MockTTSAdapter(settings.mock, settings.adapters.tts)
    if settings.adapters.tts.provider == "piper":
        from bionic_head.adapters.piper import PiperTTSAdapter

        return PiperTTSAdapter.from_settings(
            settings.providers.piper,
            grace_seconds=settings.limits.subprocess_terminate_grace_seconds,
        )
    raise _provider_unavailable(settings.adapters.tts.provider)


def build_registry(settings: AppSettings) -> AdapterRegistry:
    _ensure_mock(settings.adapters.audio2face)
    _ensure_mock(settings.adapters.ue5)

    return AdapterRegistry(
        asr=_ASRWrapper(
            _build_asr(settings),
            settings.adapters.asr,
            "asr",
        ),
        llm=_LLMWrapper(
            _build_llm(settings),
            settings.adapters.llm,
            "llm",
        ),
        tts=_TTSWrapper(
            _build_tts(settings),
            settings.adapters.tts,
            "tts",
        ),
        audio2face=_Audio2FaceWrapper(
            MockAudio2FaceAdapter(settings.mock, settings.adapters.audio2face),
            settings.adapters.audio2face,
            "audio2face",
        ),
        ue5=_UE5Wrapper(
            MockUE5Adapter(settings.mock, settings.adapters.ue5),
            settings.adapters.ue5,
            "ue5",
        ),
    )
