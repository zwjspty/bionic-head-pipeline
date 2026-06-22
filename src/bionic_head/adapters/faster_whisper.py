from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Callable
from time import perf_counter
from uuid import UUID

from bionic_head.config import FasterWhisperSettings
from bionic_head.core.audio import inspect_wav
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import ASRResult, DiagnosticResult, TurnContext


ModelFactory = Callable[..., object]


def _asr_error(
    *,
    code: ErrorCode,
    message: str,
    retryable: bool,
) -> PipelineException:
    return PipelineException(
        code=code,
        stage="asr",
        provider="faster-whisper",
        retryable=retryable,
        message=message,
    )


def _provider_unavailable() -> PipelineException:
    return _asr_error(
        code=ErrorCode.PROVIDER_UNAVAILABLE,
        message="faster-whisper is not installed",
        retryable=False,
    )


def _provider_failed() -> PipelineException:
    return _asr_error(
        code=ErrorCode.PROVIDER_FAILED,
        message="faster-whisper transcription failed",
        retryable=True,
    )


def _no_speech() -> PipelineException:
    return _asr_error(
        code=ErrorCode.NO_SPEECH_DETECTED,
        message="No speech detected",
        retryable=True,
    )


def _default_model_factory(**kwargs: object) -> object:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise _provider_unavailable() from exc

    return WhisperModel(**kwargs)


class FasterWhisperASRAdapter:
    name = "faster-whisper"

    def __init__(
        self,
        settings: FasterWhisperSettings,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.settings = settings
        self._model_factory = model_factory or _default_model_factory
        self._model: object | None = None
        self._model_lock = asyncio.Lock()

    async def transcribe(self, audio_path, context: TurnContext) -> ASRResult:
        context.cancellation.raise_if_cancelled()
        model = await self._get_model()
        context.cancellation.raise_if_cancelled()

        try:
            segments, info = await asyncio.to_thread(
                model.transcribe,  # type: ignore[attr-defined]
                str(audio_path),
                language=self.settings.language,
                vad_filter=True,
            )
        except asyncio.CancelledError:
            raise
        except PipelineException:
            raise
        except Exception as exc:
            raise _provider_failed() from exc

        context.cancellation.raise_if_cancelled()
        parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        text = " ".join(parts)
        if not text:
            raise _no_speech()

        return ASRResult(
            text=text,
            language=getattr(info, "language", self.settings.language) or self.settings.language,
            confidence=getattr(info, "language_probability", None),
            audio=inspect_wav(audio_path),
        )

    async def _get_model(self) -> object:
        if self._model is not None:
            return self._model

        async with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                self._model = await asyncio.to_thread(
                    self._model_factory,
                    model_size_or_path=self.settings.model,
                    device=self.settings.device,
                    compute_type=self.settings.compute_type,
                )
            except asyncio.CancelledError:
                raise
            except PipelineException:
                raise
            except Exception as exc:
                raise _provider_failed() from exc
            return self._model

    async def diagnostics(self) -> DiagnosticResult:
        started = perf_counter()
        available = importlib.util.find_spec("faster_whisper") is not None
        if not available:
            return DiagnosticResult(
                adapter="asr",
                provider=self.name,
                available=False,
                latency_ms=(perf_counter() - started) * 1000.0,
                message="faster-whisper dependency is not installed",
            )

        return DiagnosticResult(
            adapter="asr",
            provider=self.name,
            available=True,
            latency_ms=(perf_counter() - started) * 1000.0,
            message=(
                "faster-whisper dependency available "
                f"(model={self.settings.model}, device={self.settings.device}, "
                f"compute_type={self.settings.compute_type})"
            ),
        )

    async def cancel(self, turn_id: UUID) -> None:
        await asyncio.sleep(0)
