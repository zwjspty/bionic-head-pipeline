from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import string
import sys
import wave
from pathlib import Path
from time import perf_counter
from uuid import UUID

from bionic_head.config import PiperSettings
from bionic_head.core.process import run_command
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import AudioArtifact, DiagnosticResult, Emotion, TurnContext


ALLOWED_TEMPLATE_FIELDS = {"model_path", "output_path", "text"}


def _tts_error(
    *,
    code: ErrorCode,
    message: str,
    retryable: bool,
) -> PipelineException:
    return PipelineException(
        code=code,
        stage="tts",
        provider="piper",
        retryable=retryable,
        message=message,
    )


def _invalid_request(message: str) -> PipelineException:
    return _tts_error(code=ErrorCode.INVALID_REQUEST, message=message, retryable=False)


def _output_invalid(message: str) -> PipelineException:
    return _tts_error(
        code=ErrorCode.OUTPUT_VALIDATION_FAILED,
        message=message,
        retryable=False,
    )


def _map_process_error(exc: PipelineException) -> PipelineException:
    if exc.code is ErrorCode.PROVIDER_TIMEOUT:
        return _tts_error(
            code=ErrorCode.PROVIDER_TIMEOUT,
            message="Piper synthesis timed out",
            retryable=True,
        )
    if exc.code is ErrorCode.PROVIDER_UNAVAILABLE:
        return _tts_error(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message="Piper command is unavailable",
            retryable=False,
        )
    return _tts_error(
        code=ErrorCode.PROVIDER_FAILED,
        message="Piper synthesis failed",
        retryable=True,
    )


def _template_fields(args: list[str]) -> set[str]:
    fields: set[str] = set()
    formatter = string.Formatter()
    for arg in args:
        try:
            parsed = formatter.parse(arg)
            for _, field_name, format_spec, conversion in parsed:
                if field_name is None:
                    continue
                if format_spec or conversion:
                    raise _invalid_request("Piper command templates cannot use format modifiers")
                if field_name not in ALLOWED_TEMPLATE_FIELDS:
                    raise _invalid_request(f"Unknown Piper command template variable: {field_name}")
                fields.add(field_name)
        except ValueError as exc:
            raise _invalid_request("Invalid Piper command template") from exc
    return fields


def _executable_available(executable: str) -> bool:
    if not executable:
        return False
    path = Path(executable)
    if path.is_absolute() or os.sep in executable:
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(executable) is not None


def _piper_python_api_available() -> bool:
    if "piper" in sys.modules:
        return True
    return importlib.util.find_spec("piper") is not None


def _audio_artifact_from_piper_wav(path: Path) -> AudioArtifact:
    if not path.exists():
        raise _output_invalid("Piper did not write an output WAV")
    if path.stat().st_size <= 0:
        raise _output_invalid("Piper wrote an empty output WAV")

    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()
            frames = wav.readframes(frame_count)
    except (wave.Error, OSError, EOFError) as exc:
        raise _output_invalid("Piper output is not a valid WAV") from exc

    if channels != 1:
        raise _output_invalid("Piper WAV output must be mono")
    if sample_width != 2:
        raise _output_invalid("Piper WAV output must be signed 16-bit PCM")
    if frame_count <= 0 or not frames:
        raise _output_invalid("Piper WAV output must contain audio frames")
    if len(frames) % sample_width != 0:
        raise _output_invalid("Piper WAV frame data is not aligned")

    return AudioArtifact(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        duration_seconds=frame_count / float(sample_rate),
        byte_length=path.stat().st_size,
    )


class PiperTTSAdapter:
    name = "piper"

    def __init__(
        self,
        *,
        executable: str,
        args: list[str],
        model_path: str | Path | None,
        timeout_seconds: float,
        grace_seconds: float,
        cwd: str | Path | None = None,
        runtime: str = "cli",
        config_path: str | Path | None = None,
        use_cuda: bool = False,
        speaker_id: int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w_scale: float | None = None,
        normalize_audio: bool = True,
        volume: float = 1.0,
    ) -> None:
        self.executable = executable
        self.args = args
        self.model_path = Path(model_path) if model_path is not None else None
        self.timeout_seconds = timeout_seconds
        self.grace_seconds = grace_seconds
        self.cwd = Path(cwd) if cwd is not None else None
        self.runtime = runtime
        self.config_path = Path(config_path) if config_path is not None else None
        self.use_cuda = use_cuda
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.normalize_audio = normalize_audio
        self.volume = volume
        self.call_count = 0
        self._voice: object | None = None
        self._synthesis_lock = asyncio.Lock()
        if self.runtime not in {"cli", "python"}:
            raise _invalid_request(f"Unsupported Piper runtime: {runtime}")
        self._fields = _template_fields(args) if self.runtime == "cli" else set()
        if self.runtime == "cli" and "output_path" not in self._fields:
            raise _invalid_request("Piper command template must include {output_path}")

    @classmethod
    def from_settings(
        cls,
        settings: PiperSettings,
        *,
        grace_seconds: float,
    ) -> "PiperTTSAdapter":
        return cls(
            executable=settings.executable,
            args=list(settings.args),
            model_path=settings.model_path,
            cwd=settings.cwd,
            timeout_seconds=settings.timeout_seconds,
            grace_seconds=grace_seconds,
            runtime=settings.runtime,
            config_path=settings.config_path,
            use_cuda=settings.use_cuda,
            speaker_id=settings.speaker_id,
            length_scale=settings.length_scale,
            noise_scale=settings.noise_scale,
            noise_w_scale=settings.noise_w_scale,
            normalize_audio=settings.normalize_audio,
            volume=settings.volume,
        )

    async def synthesize(
        self,
        text: str,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> AudioArtifact:
        context.cancellation.raise_if_cancelled()
        self.call_count += 1
        output_path = context.artifact_dir / "tts" / f"piper_{self.call_count:04d}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.runtime == "python":
            await self._synthesize_with_python_runtime(text, output_path)
            context.cancellation.raise_if_cancelled()
            return _audio_artifact_from_piper_wav(output_path)

        command_args = [self.executable, *self._render_args(text, output_path)]
        stdin = None if "text" in self._fields else text.encode("utf-8")

        try:
            await run_command(
                args=command_args,
                cwd=self.cwd,
                stdin=stdin,
                timeout_seconds=self.timeout_seconds,
                cancellation=context.cancellation,
                grace_seconds=self.grace_seconds,
            )
        except asyncio.CancelledError:
            raise
        except PipelineException as exc:
            raise _map_process_error(exc) from exc

        context.cancellation.raise_if_cancelled()
        return _audio_artifact_from_piper_wav(output_path)

    async def _synthesize_with_python_runtime(self, text: str, output_path: Path) -> None:
        try:
            async with self._synthesis_lock:
                await asyncio.wait_for(
                    asyncio.to_thread(self._synthesize_python_sync, text, output_path),
                    timeout=self.timeout_seconds,
                )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as exc:
            raise _tts_error(
                code=ErrorCode.PROVIDER_TIMEOUT,
                message="Piper synthesis timed out",
                retryable=True,
            ) from exc
        except ImportError as exc:
            raise _tts_error(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message="Piper Python API is unavailable",
                retryable=False,
            ) from exc
        except PipelineException:
            raise
        except Exception as exc:
            raise _tts_error(
                code=ErrorCode.PROVIDER_FAILED,
                message="Piper synthesis failed",
                retryable=True,
            ) from exc

    def _synthesize_python_sync(self, text: str, output_path: Path) -> None:
        from piper import SynthesisConfig

        voice = self._get_python_voice_sync()
        with wave.open(str(output_path), "wb") as wav:
            voice.synthesize_wav(
                text,
                wav,
                syn_config=SynthesisConfig(
                    speaker_id=self.speaker_id,
                    length_scale=self.length_scale,
                    noise_scale=self.noise_scale,
                    noise_w_scale=self.noise_w_scale,
                    normalize_audio=self.normalize_audio,
                    volume=self.volume,
                ),
            )

    def _get_python_voice_sync(self) -> object:
        if self._voice is not None:
            return self._voice
        if self.model_path is None:
            raise _tts_error(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message="Piper model path is not configured",
                retryable=False,
            )
        if not self.model_path.exists():
            raise _tts_error(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message="Piper model file is missing",
                retryable=False,
            )
        if self.config_path is not None and not self.config_path.exists():
            raise _tts_error(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message="Piper config file is missing",
                retryable=False,
            )

        from piper import PiperVoice

        self._voice = PiperVoice.load(
            self.model_path,
            config_path=self.config_path,
            use_cuda=self.use_cuda,
        )
        return self._voice

    def _render_args(self, text: str, output_path: Path) -> list[str]:
        values = {
            "model_path": str(self.model_path) if self.model_path is not None else "",
            "output_path": str(output_path),
            "text": text,
        }
        return [arg.format(**values) for arg in self.args]

    async def diagnostics(self) -> DiagnosticResult:
        started = perf_counter()
        if self.runtime == "python":
            if not _piper_python_api_available():
                return self._diagnostic(
                    available=False,
                    started=started,
                    message="Piper Python API is unavailable",
                )
            if self.model_path is None:
                return self._diagnostic(
                    available=False,
                    started=started,
                    message="Piper model path is not configured",
                )
            if not self.model_path.exists():
                return self._diagnostic(
                    available=False,
                    started=started,
                    message="Piper model file is missing",
                )
            if self.config_path is not None and not self.config_path.exists():
                return self._diagnostic(
                    available=False,
                    started=started,
                    message="Piper config file is missing",
                )
            return self._diagnostic(
                available=True,
                started=started,
                message="Piper Python provider configuration is ready",
            )

        if not _executable_available(self.executable):
            return self._diagnostic(
                available=False,
                started=started,
                message="Piper executable is unavailable",
            )
        if self.model_path is None:
            return self._diagnostic(
                available=False,
                started=started,
                message="Piper model path is not configured",
            )
        if not self.model_path.exists():
            return self._diagnostic(
                available=False,
                started=started,
                message="Piper model file is missing",
            )
        if self.cwd is not None and not self.cwd.exists():
            return self._diagnostic(
                available=False,
                started=started,
                message="Piper working directory is missing",
            )
        return self._diagnostic(
            available=True,
            started=started,
            message="Piper provider configuration is ready",
        )

    def _diagnostic(
        self,
        *,
        available: bool,
        started: float,
        message: str,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            adapter="tts",
            provider=self.name,
            available=available,
            latency_ms=(perf_counter() - started) * 1000.0,
            message=message,
        )

    async def cancel(self, turn_id: UUID) -> None:
        await asyncio.sleep(0)
