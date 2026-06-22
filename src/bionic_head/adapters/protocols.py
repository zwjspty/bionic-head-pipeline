from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID

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


@runtime_checkable
class ASRAdapter(Protocol):
    name: str

    async def transcribe(self, audio_path: Path, context: TurnContext) -> ASRResult:
        raise NotImplementedError

    async def diagnostics(self) -> DiagnosticResult:
        raise NotImplementedError

    async def cancel(self, turn_id: UUID) -> None:
        raise NotImplementedError


@runtime_checkable
class LLMAdapter(Protocol):
    name: str

    async def chat(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> LLMResult:
        raise NotImplementedError

    def chat_stream(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> AsyncIterator[LLMEvent]:
        raise NotImplementedError

    async def diagnostics(self) -> DiagnosticResult:
        raise NotImplementedError

    async def cancel(self, turn_id: UUID) -> None:
        raise NotImplementedError


@runtime_checkable
class TTSAdapter(Protocol):
    name: str

    async def synthesize(
        self,
        text: str,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> AudioArtifact:
        raise NotImplementedError

    async def diagnostics(self) -> DiagnosticResult:
        raise NotImplementedError

    async def cancel(self, turn_id: UUID) -> None:
        raise NotImplementedError


@runtime_checkable
class Audio2FaceAdapter(Protocol):
    name: str

    async def drive(
        self,
        audio: AudioArtifact,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> FaceArtifact:
        raise NotImplementedError

    async def diagnostics(self) -> DiagnosticResult:
        raise NotImplementedError

    async def cancel(self, turn_id: UUID) -> None:
        raise NotImplementedError


@runtime_checkable
class UE5Adapter(Protocol):
    name: str

    async def format(self, face: FaceArtifact, context: TurnContext) -> UE5Payload:
        raise NotImplementedError

    async def diagnostics(self) -> DiagnosticResult:
        raise NotImplementedError

    async def cancel(self, turn_id: UUID) -> None:
        raise NotImplementedError
