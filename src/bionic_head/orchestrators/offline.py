from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID
import asyncio
import shutil

from bionic_head.adapters.registry import AdapterRegistry
from bionic_head.config import AppSettings
from bionic_head.core.artifacts import ArtifactStore
from bionic_head.core.audio import inspect_wav
from bionic_head.core.timeline import Timeline
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import PipelineResult, TurnContext


CommitCallback = Callable[[], None]
CommitIfCurrent = Callable[[UUID, UUID, CommitCallback], Awaitable[bool]]


@dataclass
class OfflineOrchestrator:
    settings: AppSettings
    registry: AdapterRegistry
    store: ArtifactStore
    commit_if_current: CommitIfCurrent

    async def run(self, input_path: Path, context: TurnContext) -> PipelineResult:
        timeline = Timeline()
        copied = context.artifact_dir / "input.wav"

        try:
            timeline.mark("started")
            context.artifact_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, copied)
            context.cancellation.raise_if_cancelled()

            stats = inspect_wav(copied)
            if stats.rms <= self.settings.stream.silence_rms_threshold:
                raise PipelineException(
                    code=ErrorCode.NO_SPEECH_DETECTED,
                    stage="audio",
                    provider=None,
                    retryable=True,
                    message="No speech detected",
                )

            with timeline.stage("asr", self.registry.asr.name):
                asr = await self.registry.asr.transcribe(copied, context)
            with timeline.stage("llm", self.registry.llm.name):
                llm = await self.registry.llm.chat(asr.text, [], context)
            with timeline.stage("tts", self.registry.tts.name):
                audio = await self.registry.tts.synthesize(
                    llm.reply,
                    llm.emotion,
                    llm.intensity,
                    context,
                )
            with timeline.stage("audio2face", self.registry.audio2face.name):
                face = await self.registry.audio2face.drive(
                    audio,
                    llm.emotion,
                    llm.intensity,
                    context,
                )
            with timeline.stage("ue5", self.registry.ue5.name):
                ue5 = await self.registry.ue5.format(face, context)

            context.cancellation.raise_if_cancelled()
            timeline.mark("completed")
            snapshot = timeline.snapshot()
            result = PipelineResult(
                session_id=context.session_id,
                turn_id=context.turn_id,
                asr=asr,
                llm=llm,
                audio=audio,
                face=face,
                ue5=ue5,
                timeline=snapshot,
            )

            self._write_success_artifacts(context, result)
            await self.store.publish_latest(
                pipeline=result.model_dump(mode="json"),
                ue5=ue5.model_dump(mode="json"),
                commit_if_current=lambda callback: self.commit_if_current(
                    context.session_id,
                    context.turn_id,
                    callback,
                ),
            )
            return result
        except asyncio.CancelledError:
            timeline.mark("cancelled")
            self._write_timeline(context, timeline)
            raise
        except BaseException:
            self._write_timeline(context, timeline)
            raise

    def _write_success_artifacts(self, context: TurnContext, result: PipelineResult) -> None:
        self.store.write_json(
            context.artifact_dir / "asr.json",
            result.asr.model_dump(mode="json"),
        )
        self.store.write_json(
            context.artifact_dir / "llm.json",
            result.llm.model_dump(mode="json"),
        )
        self.store.write_json(
            context.artifact_dir / "tts" / "result.json",
            result.audio.model_dump(mode="json"),
        )
        self.store.write_json(
            context.artifact_dir / "face" / "result.json",
            result.face.model_dump(mode="json"),
        )
        self.store.write_json(
            context.artifact_dir / "ue5" / "result.json",
            result.ue5.model_dump(mode="json"),
        )
        self.store.write_json(context.artifact_dir / "timeline.json", result.timeline)

    def _write_timeline(self, context: TurnContext, timeline: Timeline) -> None:
        self.store.write_json(context.artifact_dir / "timeline.json", timeline.snapshot())
