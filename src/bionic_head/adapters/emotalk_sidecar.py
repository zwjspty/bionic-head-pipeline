from __future__ import annotations

import asyncio
import json
import shutil
import struct
import time
import wave
from pathlib import Path
from uuid import UUID

import numpy as np

from bionic_head.config import EmoTalkSidecarSettings
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import AudioArtifact, DiagnosticResult, Emotion, FaceArtifact, TurnContext
from bionic_head.sidecar_protocol import (
    HEADER_PREFIX_SIZE,
    SidecarProtocolError,
    SidecarRequest,
    decode_message,
    decode_response,
    encode_request,
)


class SidecarProcessError(PipelineException):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            stage="audio2face",
            provider="emotalk_sidecar",
            retryable=retryable,
            message=message,
        )


class EmoTalkSidecarAudio2FaceAdapter:
    name = "emotalk_sidecar"
    _close_timeout_seconds = 0.2
    _stderr_tail_limit = 8192

    def __init__(
        self,
        *,
        sidecar_command: list[str],
        sample_rate: int,
        fps: int,
        timeout_seconds: float,
        channel_count: int = 52,
        output_npy_name: str = "face.npy",
    ) -> None:
        self.sidecar_command = list(sidecar_command)
        self.sample_rate = sample_rate
        self.fps = fps
        self.timeout_seconds = timeout_seconds
        self.channel_count = channel_count
        self.output_npy_name = output_npy_name
        self.call_count = 0
        self.process_start_count = 0
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = ""
        self._request_lock = asyncio.Lock()
        self._process_lock = asyncio.Lock()

    @classmethod
    def from_settings(
        cls,
        settings: EmoTalkSidecarSettings,
    ) -> "EmoTalkSidecarAudio2FaceAdapter":
        return cls(
            sidecar_command=list(settings.sidecar_command),
            sample_rate=settings.sample_rate,
            fps=settings.fps,
            timeout_seconds=settings.timeout_seconds,
            channel_count=settings.channel_count,
            output_npy_name=settings.output_npy_name,
        )

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        return self._process

    @property
    def process_pid(self) -> int | None:
        if self._process is None:
            return None
        return self._process.pid

    async def drive(
        self,
        audio: AudioArtifact,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> FaceArtifact:
        del emotion, intensity
        context.cancellation.raise_if_cancelled()

        request = self._build_request(audio.path, context)

        async with self._request_lock:
            self.call_count += 1
            output_dir = (
                context.artifact_dir / "face" / f"emotalk_sidecar_{self.call_count:04d}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                response = await asyncio.wait_for(
                    self._transact(request),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                pid = self.process_pid
                await self.close()
                raise PipelineException(
                    code=ErrorCode.PROVIDER_TIMEOUT,
                    stage="audio2face",
                    provider=self.name,
                    retryable=True,
                    message=f"EmoTalk sidecar timed out pid={pid}",
                ) from exc

        context.cancellation.raise_if_cancelled()
        return self._write_face_artifact(output_dir, request, response)

    async def diagnostics(self) -> DiagnosticResult:
        started = time.perf_counter()
        executable = self.sidecar_command[0] if self.sidecar_command else ""
        if not executable:
            return self._diagnostic(
                started=started,
                available=False,
                message="EmoTalk sidecar command not configured",
            )
        if not self._command_executable_available(executable):
            return self._diagnostic(
                started=started,
                available=False,
                message=f"EmoTalk sidecar executable is unavailable: {executable}",
            )
        return self._diagnostic(
            started=started,
            available=True,
            message="EmoTalk sidecar provider configuration ready",
        )

    async def cancel(self, turn_id: UUID) -> None:
        del turn_id
        await self.close()

    async def close(self) -> None:
        async with self._process_lock:
            process = self._process
            stderr_task = self._stderr_task
            self._process = None
            self._stderr_task = None

            if process is not None:
                if process.stdin is not None and not process.stdin.is_closing():
                    process.stdin.close()
                    wait_closed = getattr(process.stdin, "wait_closed", None)
                    if wait_closed is not None:
                        try:
                            await wait_closed()
                        except Exception:
                            pass

                try:
                    await asyncio.wait_for(process.wait(), timeout=self._close_timeout_seconds)
                except asyncio.TimeoutError:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=self._close_timeout_seconds)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()

            if stderr_task is not None:
                if process is None or process.returncode is not None:
                    await asyncio.gather(stderr_task, return_exceptions=True)
                else:
                    stderr_task.cancel()
                    await asyncio.gather(stderr_task, return_exceptions=True)

    def _build_request(self, audio_path: Path, context: TurnContext) -> SidecarRequest:
        pcm = self._wav_to_pcm16(audio_path)
        return SidecarRequest(
            session_id=str(context.session_id),
            turn_id=str(context.turn_id),
            generation_epoch=getattr(context, "generation_epoch", 0),
            sample_rate=self.sample_rate,
            channels=1,
            dtype="int16",
            num_samples=len(pcm) // 2,
            fps=self.fps,
            audio=pcm,
        )

    def _wav_to_pcm16(self, path: Path) -> bytes:
        try:
            with wave.open(str(path), "rb") as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                frame_count = wav.getnframes()
                frames = wav.readframes(frame_count)
        except (wave.Error, OSError, EOFError) as exc:
            raise self._output_invalid("Invalid WAV audio for EmoTalk sidecar") from exc

        if sample_width != 2:
            raise self._output_invalid("WAV must be signed 16-bit PCM for EmoTalk sidecar")
        if channels <= 0 or frame_count <= 0 or not frames:
            raise self._output_invalid("WAV must contain audio frames")
        if len(frames) % (sample_width * channels) != 0:
            raise self._output_invalid("WAV frame data is not aligned to sample width")

        samples = np.frombuffer(frames, dtype=np.int16)
        if samples.size == 0:
            raise self._output_invalid("WAV must contain samples")
        if channels > 1:
            samples = samples.reshape(-1, channels).astype(np.float64).mean(axis=1)
        else:
            samples = samples.astype(np.float64)
        if sample_rate != self.sample_rate:
            samples = self._resample(samples, sample_rate, self.sample_rate)
        samples = np.clip(np.rint(samples), -32768, 32767).astype(np.int16)
        if samples.size == 0:
            raise self._output_invalid("Resampled WAV must contain samples")
        return samples.tobytes()

    def _resample(
        self,
        samples: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        if samples.size == 1:
            return samples.copy()
        target_count = max(1, int(round(samples.size * target_rate / float(source_rate))))
        source_positions = np.arange(samples.size, dtype=np.float64)
        target_positions = np.linspace(0.0, samples.size - 1, num=target_count, dtype=np.float64)
        return np.interp(target_positions, source_positions, samples)

    async def _transact(self, request: SidecarRequest):
        process = await self._ensure_process(request)
        if process.stdin is None or process.stdout is None:
            raise self._process_unavailable("EmoTalk sidecar stdio is unavailable", process)

        payload = encode_request(request)
        try:
            process.stdin.write(payload)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise self._process_unavailable(
                "EmoTalk sidecar stdin closed unexpectedly",
                process,
            ) from exc

        try:
            response_payload = await self._read_response_payload(process)
            try:
                response = decode_response(response_payload)
            except SidecarProtocolError as exc:
                raise self._output_invalid(f"Invalid EmoTalk sidecar response: {exc}") from exc

            self._validate_response(request, response, process)
            return response
        except PipelineException as exc:
            if exc.code is ErrorCode.OUTPUT_VALIDATION_FAILED:
                await self.close()
            raise

    async def _ensure_process(self, request: SidecarRequest) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process

        async with self._process_lock:
            if self._process is not None and self._process.returncode is None:
                return self._process

            if not self.sidecar_command:
                raise self._process_unavailable("EmoTalk sidecar command not configured", None)

            try:
                process = await asyncio.create_subprocess_exec(
                    *self.sidecar_command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except (FileNotFoundError, OSError) as exc:
                raise self._process_start_failed(request, "EmoTalk sidecar failed to start", exc) from exc
            self._process = process
            self.process_start_count += 1
            self._stderr_task = asyncio.create_task(self._drain_stderr(process))
            return process

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        while True:
            chunk = await process.stderr.read(1024)
            if not chunk:
                return
            decoded = chunk.decode("utf-8", errors="replace")
            self._stderr_tail = (self._stderr_tail + decoded)[-self._stderr_tail_limit :]

    async def _read_response_payload(self, process: asyncio.subprocess.Process) -> bytes:
        if process.stdout is None:
            raise self._process_unavailable("EmoTalk sidecar stdout is unavailable", process)

        try:
            prefix = await process.stdout.readexactly(HEADER_PREFIX_SIZE)
        except asyncio.IncompleteReadError as exc:
            if not exc.partial:
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass
                if process.returncode is not None:
                    raise self._process_unavailable(
                        "EmoTalk sidecar exited before writing a response",
                        process,
                    ) from exc
            raise self._output_invalid("EmoTalk sidecar truncated response header prefix") from exc

        header_len = struct.unpack(">I", prefix)[0]
        if header_len <= 0:
            raise self._output_invalid("EmoTalk sidecar response header length must be positive")

        try:
            header_bytes = await process.stdout.readexactly(header_len)
        except asyncio.IncompleteReadError as exc:
            raise self._output_invalid("EmoTalk sidecar truncated response header") from exc

        try:
            header, _ = decode_message(prefix + header_bytes)
            body_len = self._response_body_length(header)
        except SidecarProtocolError as exc:
            raise self._output_invalid(f"Invalid EmoTalk sidecar response header: {exc}") from exc

        try:
            body = await process.stdout.readexactly(body_len)
        except asyncio.IncompleteReadError as exc:
            raise self._output_invalid("EmoTalk sidecar truncated response body") from exc

        return prefix + header_bytes + body

    def _response_body_length(self, header: dict[str, object]) -> int:
        ok = header.get("ok")
        if not isinstance(ok, bool):
            raise SidecarProtocolError("ok must be bool")
        if not ok:
            return 0

        frame_count = header.get("frame_count")
        channel_count = header.get("channel_count")
        dtype = header.get("dtype")
        if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
            raise SidecarProtocolError("frame_count must be positive integer")
        if not isinstance(channel_count, int) or isinstance(channel_count, bool) or channel_count <= 0:
            raise SidecarProtocolError("channel_count must be positive integer")
        if dtype != "float32":
            raise SidecarProtocolError("dtype must be float32")
        return frame_count * channel_count * 4

    def _validate_response(self, request: SidecarRequest, response, process: asyncio.subprocess.Process) -> None:
        if not response.ok:
            mismatches = []
            if response.session_id is not None and response.session_id != request.session_id:
                mismatches.append(
                    f"session_id mismatch: request={request.session_id} response={response.session_id}"
                )
            if response.turn_id is not None and response.turn_id != request.turn_id:
                mismatches.append(f"turn_id mismatch: request={request.turn_id} response={response.turn_id}")
            if (
                response.generation_epoch is not None
                and response.generation_epoch != request.generation_epoch
            ):
                mismatches.append(
                    f"generation_epoch mismatch: request={request.generation_epoch} response={response.generation_epoch}"
                )
            if mismatches:
                raise self._output_invalid(", ".join(mismatches))

            raise PipelineException(
                code=ErrorCode.PROVIDER_FAILED,
                stage="audio2face",
                provider=self.name,
                retryable=True,
                message=(
                    "EmoTalk sidecar returned failure "
                    f"session_id={request.session_id} "
                    f"turn_id={request.turn_id} "
                    f"generation_epoch={request.generation_epoch} "
                    f"pid={process.pid} "
                    f"error_code={response.error_code} "
                    f"error_message={response.error_message}"
                ),
            )

        if response.dtype != "float32":
            raise self._output_invalid("EmoTalk sidecar response dtype must be float32")
        if response.channel_count != self.channel_count:
            raise self._output_invalid(
                f"EmoTalk sidecar response channel_count must be {self.channel_count}"
            )
        if response.frame_count <= 0:
            raise self._output_invalid("EmoTalk sidecar response must contain frames")
        if response.session_id != request.session_id:
            raise self._output_invalid("EmoTalk sidecar session_id did not match request")
        if response.turn_id != request.turn_id:
            raise self._output_invalid("EmoTalk sidecar turn_id did not match request")
        if response.generation_epoch != request.generation_epoch:
            raise self._output_invalid("EmoTalk sidecar generation_epoch did not match request")

        frames = np.frombuffer(response.frames, dtype=np.float32)
        if frames.size != response.frame_count * self.channel_count:
            raise self._output_invalid("EmoTalk sidecar frames length did not match header")
        if not np.isfinite(frames).all():
            raise self._output_invalid("EmoTalk sidecar frames must be finite")

    def _write_face_artifact(self, output_dir: Path, request: SidecarRequest, response) -> FaceArtifact:
        frames = np.frombuffer(response.frames, dtype=np.float32).copy().reshape(
            response.frame_count,
            self.channel_count,
        )
        npy_path = output_dir / self.output_npy_name
        np.save(npy_path, frames, allow_pickle=False)

        meta_path = output_dir / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "provider": self.name,
                    "sidecar_pid": self.process_pid,
                    "request": {
                        "session_id": request.session_id,
                        "turn_id": request.turn_id,
                        "generation_epoch": request.generation_epoch,
                        "sample_rate": request.sample_rate,
                        "num_samples": request.num_samples,
                        "fps": request.fps,
                    },
                    "response": {
                        "session_id": response.session_id,
                        "turn_id": response.turn_id,
                        "generation_epoch": response.generation_epoch,
                        "frame_count": response.frame_count,
                        "channel_count": response.channel_count,
                        "fps": response.fps,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return FaceArtifact(
            path=npy_path,
            frames=frames.astype(float).tolist(),
            fps=response.fps,
            channel_count=self.channel_count,
            frame_count=response.frame_count,
            auxiliary_paths=[meta_path],
        )

    def _process_unavailable(
        self,
        message: str,
        process: asyncio.subprocess.Process | None,
    ) -> SidecarProcessError:
        pid = process.pid if process is not None else None
        stderr_tail = self._stderr_tail.strip()
        suffix = f" pid={pid}" if pid is not None else ""
        if stderr_tail:
            suffix = f"{suffix} stderr={stderr_tail[-256:]}"
        return SidecarProcessError(f"{message}{suffix}")

    def _process_start_failed(
        self,
        request: SidecarRequest,
        message: str,
        exc: Exception,
    ) -> SidecarProcessError:
        detail = f"{message}: session_id={request.session_id}, turn_id={request.turn_id}, generation_epoch={request.generation_epoch}, pid=None, error={exc!s}"
        return SidecarProcessError(detail)

    def _output_invalid(self, message: str) -> PipelineException:
        return PipelineException(
            code=ErrorCode.OUTPUT_VALIDATION_FAILED,
            stage="audio2face",
            provider=self.name,
            retryable=False,
            message=message,
        )

    def _diagnostic(
        self,
        *,
        started: float,
        available: bool,
        message: str,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            adapter="audio2face",
            provider=self.name,
            available=available,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            message=message,
        )

    def _command_executable_available(self, executable: str) -> bool:
        path = Path(executable)
        if path.is_absolute() or "/" in executable:
            return path.exists() and path.is_file()
        return shutil.which(executable) is not None
