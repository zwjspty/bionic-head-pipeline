from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
import asyncio
import shutil

from bionic_head.adapters.registry import AdapterRegistry
from bionic_head.config import AppSettings
from bionic_head.core.artifacts import ArtifactStore
from bionic_head.core.audio import inspect_wav
from bionic_head.core.sentences import SentenceBuffer
from bionic_head.core.state import TurnHandle
from bionic_head.core.timeline import Timeline
from bionic_head.core.ue5 import chunk_ue5_frames
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import (
    ASRResult,
    AudioArtifact,
    FaceArtifact,
    LLMResult,
    PipelineResult,
    TurnContext,
    UE5Payload,
)
from bionic_head.protocol.events import EventEnvelope, EventFactory, EventType


EmitJSON = Callable[[EventEnvelope], Awaitable[None]]
EmitBinaryPair = Callable[[EventEnvelope, bytes], Awaitable[None]]


@dataclass
class _StreamArtifacts:
    asr: ASRResult | None = None
    llm: LLMResult | None = None
    audio: AudioArtifact | None = None
    face: FaceArtifact | None = None
    ue5: UE5Payload | None = None


@dataclass
class StreamOrchestrator:
    settings: AppSettings
    registry: AdapterRegistry
    store: ArtifactStore

    async def run(
        self,
        input_wav: Path,
        turn: TurnHandle,
        emit_json: EmitJSON,
        emit_binary_pair: EmitBinaryPair,
        event_factory: EventFactory | None = None,
    ) -> None:
        factory = event_factory or EventFactory(session_id=turn.session_id)
        timeline = Timeline()
        marks: set[str] = set()
        artifacts = _StreamArtifacts()
        turn_dir = self.store.create_turn(turn.session_id, turn.turn_id)
        context = TurnContext(
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            artifact_dir=turn_dir,
            cancellation=turn.cancellation,
        )

        def mark_once(name: str) -> None:
            if name not in marks:
                timeline.mark(name)
                marks.add(name)

        try:
            copied = turn_dir / "input.wav"
            shutil.copy2(input_wav, copied)
            mark_once("audio_end")
            self._ensure_current(turn)
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
                self._ensure_current(turn)
                artifacts.asr = await self.registry.asr.transcribe(copied, context)
            mark_once("asr_final")
            await self._emit_json(
                turn,
                emit_json,
                factory.server(
                    EventType.SERVER_ASR_FINAL,
                    turn.turn_id,
                    {
                        "text": artifacts.asr.text,
                        "language": artifacts.asr.language,
                        "confidence": artifacts.asr.confidence,
                    },
                ),
            )

            buffer = SentenceBuffer(
                max_chars=self.settings.stream.sentence_max_chars,
                min_chars=self.settings.stream.sentence_min_chars,
            )
            reply_parts: list[str] = []
            chunk_index = 0
            fallback_llm = LLMResult(
                reply="",
                emotion=self.settings.mock.emotion,
                intensity=self.settings.mock.intensity,
            )

            with timeline.stage("llm", self.registry.llm.name):
                iterator = self.registry.llm.chat_stream(artifacts.asr.text, [], context)
                while True:
                    try:
                        event = await asyncio.wait_for(
                            iterator.__anext__(),
                            timeout=self.settings.stream.sentence_max_wait_ms / 1000.0,
                        )
                    except asyncio.TimeoutError:
                        segment = buffer.flush()
                        if segment is not None:
                            chunk_index += 1
                            await self._process_segment(
                                segment,
                                chunk_index,
                                fallback_llm,
                                turn,
                                context,
                                factory,
                                timeline,
                                mark_once,
                                artifacts,
                                emit_json,
                                emit_binary_pair,
                            )
                        continue
                    except StopAsyncIteration:
                        break

                    self._ensure_current(turn)
                    if event.kind == "token":
                        mark_once("llm_first_token")
                        reply_parts.append(event.text)
                        await self._emit_json(
                            turn,
                            emit_json,
                            factory.server(
                                EventType.SERVER_LLM_TOKEN,
                                turn.turn_id,
                                {"text": event.text},
                            ),
                        )
                        for segment in buffer.push(event.text):
                            chunk_index += 1
                            await self._process_segment(
                                segment,
                                chunk_index,
                                fallback_llm,
                                turn,
                                context,
                                factory,
                                timeline,
                                mark_once,
                                artifacts,
                                emit_json,
                                emit_binary_pair,
                            )
                    elif event.kind == "final" and event.result is not None:
                        artifacts.llm = event.result
                        fallback_llm = event.result

            residual = buffer.flush()
            if residual is not None:
                chunk_index += 1
                await self._process_segment(
                    residual,
                    chunk_index,
                    fallback_llm,
                    turn,
                    context,
                    factory,
                    timeline,
                    mark_once,
                    artifacts,
                    emit_json,
                    emit_binary_pair,
                )

            if artifacts.llm is None:
                artifacts.llm = LLMResult(
                    reply="".join(reply_parts),
                    emotion=fallback_llm.emotion,
                    intensity=fallback_llm.intensity,
                )
            self._ensure_complete(artifacts)
            snapshot = timeline.snapshot()
            result = PipelineResult(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                asr=artifacts.asr,
                llm=artifacts.llm,
                audio=artifacts.audio,
                face=artifacts.face,
                ue5=artifacts.ue5,
                timeline=snapshot,
            )
            self._ensure_current(turn)
            self._write_success_artifacts(turn_dir, result)
            self._ensure_current(turn)
            await self.store.publish_latest(
                pipeline=result.model_dump(mode="json"),
                ue5=artifacts.ue5.model_dump(mode="json"),
                commit_if_current=turn.commit_if_current,
            )
            self._ensure_current(turn)
            if await turn.emit_terminal_once(EventType.SERVER_PIPELINE_DONE):
                await emit_json(factory.server(EventType.SERVER_PIPELINE_DONE, turn.turn_id, {}))
        except asyncio.CancelledError:
            mark_once("cancelled")
            if await turn.emit_terminal_once(EventType.SERVER_TURN_CANCELLED):
                await emit_json(factory.server(EventType.SERVER_TURN_CANCELLED, turn.turn_id, {}))
        except PipelineException as exc:
            mark_once("error")
            if await turn.emit_terminal_once(EventType.SERVER_PIPELINE_ERROR):
                await emit_json(
                    factory.server(
                        EventType.SERVER_PIPELINE_ERROR,
                        turn.turn_id,
                        {"error": exc.to_detail()},
                    )
                )
        except Exception:
            mark_once("error")
            error = PipelineException(
                code=ErrorCode.INTERNAL_ERROR,
                stage="stream",
                provider=None,
                retryable=True,
                message="Internal server error",
            )
            if await turn.emit_terminal_once(EventType.SERVER_PIPELINE_ERROR):
                await emit_json(
                    factory.server(
                        EventType.SERVER_PIPELINE_ERROR,
                        turn.turn_id,
                        {"error": error.to_detail()},
                    )
                )
        finally:
            self.store.write_json(turn_dir / "timeline.json", timeline.snapshot())

    async def _process_segment(
        self,
        segment: str,
        chunk_index: int,
        llm: LLMResult,
        turn: TurnHandle,
        context: TurnContext,
        factory: EventFactory,
        timeline: Timeline,
        mark_once: Callable[[str], None],
        artifacts: _StreamArtifacts,
        emit_json: EmitJSON,
        emit_binary_pair: EmitBinaryPair,
    ) -> None:
        chunk_id = f"chunk-{chunk_index:04d}"
        await self._emit_json(
            turn,
            emit_json,
            factory.server(
                EventType.SERVER_LLM_CHUNK,
                turn.turn_id,
                {"chunk_id": chunk_id, "text": segment},
            ),
        )

        with timeline.stage("tts", self.registry.tts.name):
            self._ensure_current(turn)
            audio = await self.registry.tts.synthesize(segment, llm.emotion, llm.intensity, context)
        artifacts.audio = audio
        mark_once("first_tts_ready")
        metadata = factory.server(
            EventType.SERVER_TTS_AUDIO,
            turn.turn_id,
            {
                "chunk_id": chunk_id,
                "format": "wav",
                "sample_rate": audio.sample_rate,
                "byte_length": audio.byte_length,
                "duration_seconds": audio.duration_seconds,
            },
        )
        await self._emit_binary_pair(turn, emit_binary_pair, metadata, audio.path.read_bytes())

        with timeline.stage("audio2face", self.registry.audio2face.name):
            self._ensure_current(turn)
            face = await self.registry.audio2face.drive(audio, llm.emotion, llm.intensity, context)
        artifacts.face = face
        mark_once("first_face_ready")
        await self._emit_json(
            turn,
            emit_json,
            factory.server(
                EventType.SERVER_FACE_FRAMES,
                turn.turn_id,
                {
                    "chunk_id": chunk_id,
                    "fps": face.fps,
                    "frame_count": face.frame_count,
                    "frames": face.frames,
                },
            ),
        )

        with timeline.stage("ue5", self.registry.ue5.name):
            self._ensure_current(turn)
            ue5 = await self.registry.ue5.format(face, context)
        artifacts.ue5 = ue5
        for chunk in chunk_ue5_frames(ue5, chunk_size=30, chunk_id=chunk_id):
            await self._emit_json(
                turn,
                emit_json,
                factory.server(EventType.SERVER_UE5_FRAMES, turn.turn_id, chunk),
            )

        mark_once("first_segment_ready")
        await self._emit_json(
            turn,
            emit_json,
            factory.server(
                EventType.SERVER_SEGMENT_READY,
                turn.turn_id,
                {"chunk_id": chunk_id},
            ),
        )

    async def _emit_json(self, turn: TurnHandle, emit_json: EmitJSON, envelope: EventEnvelope) -> None:
        async def operation() -> None:
            await emit_json(envelope)

        if not await turn.emit_if_current(operation):
            raise asyncio.CancelledError

    async def _emit_binary_pair(
        self,
        turn: TurnHandle,
        emit_binary_pair: EmitBinaryPair,
        envelope: EventEnvelope,
        binary: bytes,
    ) -> None:
        async def operation() -> None:
            await emit_binary_pair(envelope, binary)

        if not await turn.emit_if_current(operation):
            raise asyncio.CancelledError

    def _ensure_current(self, turn: TurnHandle) -> None:
        turn.cancellation.raise_if_cancelled()
        if not turn.is_current():
            raise asyncio.CancelledError

    def _ensure_complete(self, artifacts: _StreamArtifacts) -> None:
        if not all((artifacts.asr, artifacts.llm, artifacts.audio, artifacts.face, artifacts.ue5)):
            raise PipelineException(
                code=ErrorCode.OUTPUT_VALIDATION_FAILED,
                stage="stream",
                provider=None,
                retryable=False,
                message="Stream pipeline did not produce all required outputs",
            )

    def _write_success_artifacts(self, turn_dir: Path, result: PipelineResult) -> None:
        self.store.write_json(turn_dir / "asr.json", result.asr.model_dump(mode="json"))
        self.store.write_json(turn_dir / "llm.json", result.llm.model_dump(mode="json"))
        self.store.write_json(turn_dir / "tts" / "result.json", result.audio.model_dump(mode="json"))
        self.store.write_json(turn_dir / "face" / "result.json", result.face.model_dump(mode="json"))
        self.store.write_json(turn_dir / "ue5" / "result.json", result.ue5.model_dump(mode="json"))
        self.store.write_json(turn_dir / "timeline.json", result.timeline)
