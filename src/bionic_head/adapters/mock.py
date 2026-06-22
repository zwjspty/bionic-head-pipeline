from __future__ import annotations

from array import array
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from uuid import UUID
import asyncio
import json
import math
import wave

from bionic_head.config import AdapterSettings, MockSettings
from bionic_head.core.audio import audio_artifact_from_wav, inspect_wav
from bionic_head.core.ue5 import build_ue5_payload
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


def _stage_matches(configured: str | None, stage: str) -> bool:
    if configured is None:
        return False
    aliases = {
        "audio2face": {"audio2face", "face"},
        "face": {"audio2face", "face"},
        "llm": {"llm", "llm_first_token", "llm_token"},
    }
    return configured in aliases.get(stage, {stage})


def _latency_ms(settings: MockSettings, stage: str) -> int:
    if stage == "llm":
        return settings.latency_ms.llm_first_token
    if stage == "audio2face":
        return settings.latency_ms.face
    if stage == "ue5":
        return 0
    return int(getattr(settings.latency_ms, stage))


def _provider_failed(stage: str, message: str) -> PipelineException:
    return PipelineException(
        code=ErrorCode.PROVIDER_FAILED,
        stage=stage,
        provider="mock",
        retryable=True,
        message=message,
    )


@dataclass
class _MockBase:
    settings: MockSettings
    adapter_settings: AdapterSettings
    adapter_name: str
    stage: str

    name: str = "mock"
    call_count: int = 0
    call_counts: dict[str, int] = field(default_factory=dict)

    def __getattr__(self, name: str) -> int:
        if name.endswith("_call_count"):
            return self.call_counts.get(name[: -len("_call_count")], 0)
        raise AttributeError(name)

    def _record_call(self, method: str) -> None:
        self.call_count += 1
        self.call_counts[method] = self.call_counts.get(method, 0) + 1

    async def _before(self, context: TurnContext) -> None:
        context.cancellation.raise_if_cancelled()
        if _stage_matches(self.settings.fail_stage, self.stage):
            raise RuntimeError(f"Configured mock failure for {self.stage}")

        latency = _latency_ms(self.settings, self.stage) / 1000.0
        if latency:
            await asyncio.sleep(latency)
        context.cancellation.raise_if_cancelled()

        if _stage_matches(self.settings.timeout_stage, self.stage):
            await asyncio.sleep(self.adapter_settings.timeout_seconds + 1.0)

        context.cancellation.raise_if_cancelled()

    async def diagnostics(self) -> DiagnosticResult:
        started = perf_counter()
        await asyncio.sleep(0)
        return DiagnosticResult(
            adapter=self.adapter_name,
            provider=self.name,
            available=True,
            latency_ms=(perf_counter() - started) * 1000.0,
            message=f"{self.name} provider ready",
        )

    async def cancel(self, turn_id: UUID) -> None:
        await asyncio.sleep(0)


class MockASRAdapter(_MockBase):
    def __init__(self, settings: MockSettings, adapter_settings: AdapterSettings) -> None:
        super().__init__(settings, adapter_settings, "asr", "asr")

    async def transcribe(self, audio_path: Path, context: TurnContext) -> ASRResult:
        await self._before(context)
        self._record_call("transcribe")
        stats = inspect_wav(audio_path)
        return ASRResult(
            text=self.settings.asr_text,
            language="zh",
            confidence=1.0,
            audio=stats,
        )


class MockLLMAdapter(_MockBase):
    def __init__(self, settings: MockSettings, adapter_settings: AdapterSettings) -> None:
        super().__init__(settings, adapter_settings, "llm", "llm")

    async def chat(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> LLMResult:
        await self._before(context)
        self._record_call("chat")
        return LLMResult(
            reply=self.settings.reply,
            emotion=self.settings.emotion,
            intensity=self.settings.intensity,
        )

    async def _chat_stream(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> AsyncIterator[LLMEvent]:
        await self._before(context)
        self._record_call("chat_stream")
        for character in self.settings.reply:
            context.cancellation.raise_if_cancelled()
            token_latency = self.settings.latency_ms.llm_token / 1000.0
            if token_latency:
                await asyncio.sleep(token_latency)
            yield LLMEvent(kind="token", text=character)
        yield LLMEvent(
            kind="final",
            result=LLMResult(
                reply=self.settings.reply,
                emotion=self.settings.emotion,
                intensity=self.settings.intensity,
            ),
        )

    def chat_stream(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> AsyncIterator[LLMEvent]:
        return self._chat_stream(text, history, context)


class MockTTSAdapter(_MockBase):
    def __init__(self, settings: MockSettings, adapter_settings: AdapterSettings) -> None:
        super().__init__(settings, adapter_settings, "tts", "tts")

    async def synthesize(
        self,
        text: str,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> AudioArtifact:
        await self._before(context)
        self._record_call("synthesize")
        output_path = context.artifact_dir / "tts" / f"mock_tts_{self.call_count:04d}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = 16000
        duration_seconds = 0.25
        total_samples = int(sample_rate * duration_seconds)
        amplitude = max(400, int(1400 * max(0.0, min(1.0, intensity))))
        samples = array(
            "h",
            (
                int(amplitude * math.sin(2.0 * math.pi * 220.0 * index / sample_rate))
                for index in range(total_samples)
            ),
        )
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(samples.tobytes())

        return audio_artifact_from_wav(output_path)


class MockAudio2FaceAdapter(_MockBase):
    def __init__(self, settings: MockSettings, adapter_settings: AdapterSettings) -> None:
        super().__init__(settings, adapter_settings, "audio2face", "audio2face")

    async def drive(
        self,
        audio: AudioArtifact,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> FaceArtifact:
        await self._before(context)
        self._record_call("drive")
        frame_count = max(1, round(audio.duration_seconds * 30))
        frames = [
            [
                round((((frame_index + 1) * (channel_index + 3)) % 100) / 100.0 * intensity, 6)
                for channel_index in range(52)
            ]
            for frame_index in range(frame_count)
        ]
        output_path = context.artifact_dir / "face" / f"mock_face_{self.call_count:04d}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"fps": 30, "frames": frames}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return FaceArtifact(
            path=output_path,
            frames=frames,
            fps=30,
            channel_count=52,
            frame_count=frame_count,
        )


class MockUE5Adapter(_MockBase):
    def __init__(self, settings: MockSettings, adapter_settings: AdapterSettings) -> None:
        super().__init__(settings, adapter_settings, "ue5", "ue5")

    async def format(self, face: FaceArtifact, context: TurnContext) -> UE5Payload:
        await self._before(context)
        self._record_call("format")
        return build_ue5_payload(face.frames, fps=face.fps)


def map_mock_runtime_error(exc: Exception, *, stage: str) -> PipelineException:
    if isinstance(exc, PipelineException):
        return exc
    return _provider_failed(stage, f"Mock provider failed during {stage}")
