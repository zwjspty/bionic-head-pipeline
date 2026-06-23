from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import string
from pathlib import Path
from time import perf_counter
from uuid import UUID

import numpy as np

from bionic_head.config import MorpheusSettings
from bionic_head.core.process import run_command
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import AudioArtifact, DiagnosticResult, Emotion, FaceArtifact, TurnContext


ALLOWED_TEMPLATE_FIELDS = {"input_path", "output_dir", "emotion", "intensity"}


def _executable_available(executable: str) -> bool:
    if not executable:
        return False
    path = Path(executable)
    if path.is_absolute() or os.sep in executable:
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(executable) is not None


class MorpheusAudio2FaceAdapter:
    name = "morpheus"
    label = "Morpheus"
    output_prefix = "morpheus"
    _shared_semaphore = asyncio.Semaphore(1)

    def __init__(
        self,
        *,
        executable: str,
        args: list[str],
        output_npy_glob: str,
        output_json_glob: str,
        timeout_seconds: float,
        grace_seconds: float,
        cwd: str | Path | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.executable = executable
        self.args = args
        self.output_npy_glob = output_npy_glob
        self.output_json_glob = output_json_glob
        self.timeout_seconds = timeout_seconds
        self.grace_seconds = grace_seconds
        self.cwd = Path(cwd) if cwd is not None else None
        self.call_count = 0
        self._fields = self._template_fields(args)
        self._semaphore = semaphore or self._shared_semaphore
        if "input_path" not in self._fields or "output_dir" not in self._fields:
            raise self._invalid_request(
                f"{self.label} command template must include {{input_path}} and {{output_dir}}"
            )

    @classmethod
    def from_settings(
        cls,
        settings: MorpheusSettings,
        *,
        grace_seconds: float,
    ) -> "MorpheusAudio2FaceAdapter":
        return cls(
            executable=settings.executable,
            args=list(settings.args),
            cwd=settings.cwd,
            output_npy_glob=settings.output_npy_glob,
            output_json_glob=settings.output_json_glob,
            timeout_seconds=settings.timeout_seconds,
            grace_seconds=grace_seconds,
        )

    async def drive(
        self,
        audio: AudioArtifact,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> FaceArtifact:
        context.cancellation.raise_if_cancelled()
        self.call_count += 1
        output_dir = context.artifact_dir / "face" / f"{self.output_prefix}_{self.call_count:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        command_args = [
            self.executable,
            *self._render_args(
                input_path=audio.path,
                output_dir=output_dir,
                emotion=emotion,
                intensity=intensity,
            ),
        ]

        try:
            async with self._semaphore:
                await run_command(
                    args=command_args,
                    cwd=self.cwd,
                    stdin=None,
                    timeout_seconds=self.timeout_seconds,
                    cancellation=context.cancellation,
                    grace_seconds=self.grace_seconds,
                )
        except asyncio.CancelledError:
            raise
        except PipelineException as exc:
            raise self._map_process_error(exc) from exc

        context.cancellation.raise_if_cancelled()
        return self._load_face_artifact(output_dir, audio)

    def _audio2face_error(
        self,
        *,
        code: ErrorCode,
        message: str,
        retryable: bool,
    ) -> PipelineException:
        return PipelineException(
            code=code,
            stage="audio2face",
            provider=self.name,
            retryable=retryable,
            message=message,
        )

    def _invalid_request(self, message: str) -> PipelineException:
        return self._audio2face_error(
            code=ErrorCode.INVALID_REQUEST,
            message=message,
            retryable=False,
        )

    def _output_invalid(self, message: str) -> PipelineException:
        return self._audio2face_error(
            code=ErrorCode.OUTPUT_VALIDATION_FAILED,
            message=message,
            retryable=False,
        )

    def _map_process_error(self, exc: PipelineException) -> PipelineException:
        if exc.code is ErrorCode.PROVIDER_TIMEOUT:
            return self._audio2face_error(
                code=ErrorCode.PROVIDER_TIMEOUT,
                message=f"{self.label} processing timed out",
                retryable=True,
            )
        if exc.code is ErrorCode.PROVIDER_UNAVAILABLE:
            return self._audio2face_error(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message=f"{self.label} command is unavailable",
                retryable=False,
            )
        return self._audio2face_error(
            code=ErrorCode.PROVIDER_FAILED,
            message=f"{self.label} processing failed",
            retryable=True,
        )

    def _template_fields(self, args: list[str]) -> set[str]:
        fields: set[str] = set()
        formatter = string.Formatter()
        for arg in args:
            try:
                for _, field_name, format_spec, conversion in formatter.parse(arg):
                    if field_name is None:
                        continue
                    if format_spec or conversion:
                        raise self._invalid_request(
                            f"{self.label} templates cannot use format modifiers"
                        )
                    if field_name not in ALLOWED_TEMPLATE_FIELDS:
                        raise self._invalid_request(
                            f"Unknown {self.label} template variable: {field_name}"
                        )
                    fields.add(field_name)
            except ValueError as exc:
                raise self._invalid_request(f"Invalid {self.label} command template") from exc
        return fields

    def _render_args(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        emotion: Emotion,
        intensity: float,
    ) -> list[str]:
        values = {
            "input_path": str(input_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "emotion": emotion.value,
            "intensity": str(float(intensity)),
        }
        return [arg.format(**values) for arg in self.args]

    def _load_face_artifact(self, output_dir: Path, audio: AudioArtifact) -> FaceArtifact:
        npy_paths = sorted(output_dir.glob(self.output_npy_glob))
        if not npy_paths:
            raise self._output_invalid(f"{self.label} did not write an npy output")
        if len(npy_paths) > 1:
            raise self._output_invalid(f"{self.label} wrote multiple npy outputs")

        npy_path = npy_paths[0]
        try:
            array = np.load(npy_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise self._output_invalid(f"{self.label} npy output could not be loaded") from exc

        if array.ndim != 2 or array.shape[1] != 52:
            raise self._output_invalid(f"{self.label} output must have shape [N, 52]")
        if array.shape[0] <= 0:
            raise self._output_invalid(f"{self.label} output must contain at least one frame")
        if not np.isfinite(array).all():
            raise self._output_invalid(f"{self.label} output weights must be finite")

        json_paths = sorted(output_dir.glob(self.output_json_glob))
        fps = self._fps_from_json(json_paths) or 30
        frames = array.astype(float).tolist()
        warnings = self._quality_warnings(
            audio_duration=audio.duration_seconds,
            frame_count=len(frames),
            fps=fps,
        )
        return FaceArtifact(
            path=npy_path,
            frames=frames,
            fps=fps,
            channel_count=52,
            frame_count=len(frames),
            auxiliary_paths=json_paths,
            quality_warnings=warnings,
        )

    def _fps_from_json(self, paths: list[Path]) -> int | None:
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            fps = payload.get("fps")
            if isinstance(fps, (int, float)) and math.isfinite(float(fps)) and fps > 0:
                return int(fps)
        return None

    def _quality_warnings(
        self,
        *,
        audio_duration: float,
        frame_count: int,
        fps: int,
    ) -> list[str]:
        frame_duration = frame_count / float(fps)
        tolerance = max(0.1, 1.0 / float(fps))
        if abs(frame_duration - audio_duration) > tolerance:
            return [
                f"{self.label} frame duration differs from audio duration "
                f"({frame_duration:.3f}s vs {audio_duration:.3f}s)"
            ]
        return []

    async def diagnostics(self) -> DiagnosticResult:
        started = perf_counter()
        if not _executable_available(self.executable):
            return self._diagnostic(
                available=False,
                started=started,
                message=f"{self.label} executable is unavailable",
            )
        if any(arg == "" for arg in self.args):
            return self._diagnostic(
                available=False,
                started=started,
                message=f"{self.label} command contains an empty argument",
            )
        if self.cwd is not None and not self.cwd.exists():
            return self._diagnostic(
                available=False,
                started=started,
                message=f"{self.label} project directory is missing",
            )
        return self._diagnostic(
            available=True,
            started=started,
            message=f"{self.label} provider configuration is ready",
        )

    def _diagnostic(
        self,
        *,
        available: bool,
        started: float,
        message: str,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            adapter="audio2face",
            provider=self.name,
            available=available,
            latency_ms=(perf_counter() - started) * 1000.0,
            message=message,
        )

    async def cancel(self, turn_id: UUID) -> None:
        await asyncio.sleep(0)
