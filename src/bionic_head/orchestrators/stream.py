from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic_ns
import asyncio
import shutil

from bionic_head.adapters.registry import AdapterRegistry
from bionic_head.config import AppSettings
from bionic_head.core.artifacts import ArtifactStore
from bionic_head.core.audio import inspect_wav
from bionic_head.core.history import ConversationHistoryStore
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
from bionic_head.eye_continuity import EyeContinuityMetrics, EyeContinuityProcessor
from bionic_head.face_stitcher import FaceSegmentStitcher, FaceStitchMetrics
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


@dataclass(frozen=True)
class _RawFaceSegmentResult:
    chunk_index: int
    chunk_id: str
    face: FaceArtifact
    segment_timing: "_StreamSegmentTiming"


@dataclass(frozen=True)
class _FaceSegmentResult:
    chunk_index: int
    face: FaceArtifact
    ue5: UE5Payload


@dataclass
class _StreamSegmentTiming:
    chunk_index: int
    chunk_id: str
    turn_id: str
    generation_epoch: int
    run_started_ns: int
    tts_ready_ns: int
    tts_audio_ready_ms: float
    face_start_after_tts_ms: float | None = None
    face_total_ms: float | None = None
    ue5_first_frame_after_tts_ms: float | None = None
    e2e_first_visible_face_ms: float | None = None
    face_stitch_enabled: bool | None = None
    face_stitch_applied: bool | None = None
    face_stitch_reset: bool | None = None
    face_stitch_overlap_frames: float | None = None
    face_stitch_actual_overlap_frames: float | None = None
    face_boundary_delta_before: float | None = None
    face_boundary_delta_after: float | None = None
    eye_continuity_enabled: bool | None = None
    eye_continuity_applied: bool | None = None
    eye_continuity_reset: bool | None = None
    eye_smooth_channel_count: float | None = None
    blink_channel_count: float | None = None
    eye_continuity_overlap_frames: float | None = None
    eye_continuity_actual_overlap_frames: float | None = None
    eye_boundary_delta_before: float | None = None
    eye_boundary_delta_after: float | None = None
    blink_enabled: bool | None = None
    blink_applied_count: float | None = None
    blink_frame_count: float | None = None
    blink_reset_count: float | None = None
    eye_global_frame_start: float | None = None
    eye_global_frame_end: float | None = None
    stale_dropped: bool = False

    @property
    def segment_id(self) -> str:
        return self.chunk_id

    def timing_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"tts_audio_ready_ms": self.tts_audio_ready_ms}
        optional = {
            "face_start_after_tts_ms": self.face_start_after_tts_ms,
            "face_total_ms": self.face_total_ms,
            "ue5_first_frame_after_tts_ms": self.ue5_first_frame_after_tts_ms,
            "e2e_first_visible_face_ms": self.e2e_first_visible_face_ms,
            "face_stitch_enabled": self.face_stitch_enabled,
            "face_stitch_applied": self.face_stitch_applied,
            "face_stitch_reset": self.face_stitch_reset,
            "face_stitch_overlap_frames": self.face_stitch_overlap_frames,
            "face_stitch_actual_overlap_frames": self.face_stitch_actual_overlap_frames,
            "face_boundary_delta_before": self.face_boundary_delta_before,
            "face_boundary_delta_after": self.face_boundary_delta_after,
            "eye_continuity_enabled": self.eye_continuity_enabled,
            "eye_continuity_applied": self.eye_continuity_applied,
            "eye_continuity_reset": self.eye_continuity_reset,
            "eye_smooth_channel_count": self.eye_smooth_channel_count,
            "blink_channel_count": self.blink_channel_count,
            "eye_continuity_overlap_frames": self.eye_continuity_overlap_frames,
            "eye_continuity_actual_overlap_frames": self.eye_continuity_actual_overlap_frames,
            "eye_boundary_delta_before": self.eye_boundary_delta_before,
            "eye_boundary_delta_after": self.eye_boundary_delta_after,
            "blink_enabled": self.blink_enabled,
            "blink_applied_count": self.blink_applied_count,
            "blink_frame_count": self.blink_frame_count,
            "blink_reset_count": self.blink_reset_count,
            "eye_global_frame_start": self.eye_global_frame_start,
            "eye_global_frame_end": self.eye_global_frame_end,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = value
        return payload

    def apply_stitch_metrics(
        self,
        metrics: FaceStitchMetrics,
        *,
        record_boundary_metrics: bool,
    ) -> None:
        self.face_stitch_enabled = metrics.enabled
        self.face_stitch_applied = metrics.applied
        self.face_stitch_reset = metrics.reset
        self.face_stitch_overlap_frames = float(metrics.overlap_frames)
        self.face_stitch_actual_overlap_frames = float(metrics.actual_overlap_frames)
        if record_boundary_metrics:
            self.face_boundary_delta_before = metrics.boundary_delta_before
            self.face_boundary_delta_after = metrics.boundary_delta_after

    def apply_eye_continuity_metrics(
        self,
        metrics: EyeContinuityMetrics,
        *,
        record_boundary_metrics: bool,
    ) -> None:
        self.eye_continuity_enabled = metrics.enabled
        self.eye_continuity_applied = metrics.applied
        self.eye_continuity_reset = metrics.reset
        self.eye_smooth_channel_count = float(metrics.smooth_channel_count)
        self.blink_channel_count = float(metrics.blink_channel_count)
        self.eye_continuity_overlap_frames = float(metrics.overlap_frames)
        self.eye_continuity_actual_overlap_frames = float(metrics.actual_overlap_frames)
        self.blink_enabled = metrics.blink_enabled
        self.blink_applied_count = float(metrics.blink_applied_count)
        self.blink_frame_count = float(metrics.blink_frame_count)
        self.blink_reset_count = float(metrics.blink_reset_count)
        self.eye_global_frame_start = float(metrics.global_frame_start)
        self.eye_global_frame_end = float(metrics.global_frame_end)
        if record_boundary_metrics:
            self.eye_boundary_delta_before = metrics.boundary_delta_before
            self.eye_boundary_delta_after = metrics.boundary_delta_after

    def snapshot(self) -> dict[str, object]:
        item: dict[str, object] = {
            "chunk_index": self.chunk_index,
            "segment_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "segment_id": self.segment_id,
            "turn_id": self.turn_id,
            "generation_epoch": self.generation_epoch,
            "tts_audio_ready_ms": self.tts_audio_ready_ms,
        }
        for key, value in self.timing_payload().items():
            item[key] = value
        if self.stale_dropped:
            item["stale_dropped"] = True
        return item


@dataclass
class _StreamTiming:
    run_started_ns: int
    segments: dict[str, _StreamSegmentTiming] = field(default_factory=dict)
    old_turn_face_leak_count: int = 0
    stale_face_drop_count: int = 0
    _stale_counted_segments: set[str] = field(default_factory=set)

    def elapsed_ms(self, at_ns: int | None = None) -> float:
        if at_ns is None:
            at_ns = monotonic_ns()
        return _duration_ms(self.run_started_ns, at_ns)

    def after_tts_ms(self, segment: _StreamSegmentTiming, at_ns: int | None = None) -> float:
        if at_ns is None:
            at_ns = monotonic_ns()
        return _duration_ms(segment.tts_ready_ns, at_ns)

    def add_segment(
        self,
        *,
        chunk_index: int,
        chunk_id: str,
        turn: TurnHandle,
        tts_ready_ns: int,
    ) -> _StreamSegmentTiming:
        segment = _StreamSegmentTiming(
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            turn_id=str(turn.turn_id),
            generation_epoch=turn.generation_epoch,
            run_started_ns=self.run_started_ns,
            tts_ready_ns=tts_ready_ns,
            tts_audio_ready_ms=self.elapsed_ms(tts_ready_ns),
        )
        self.segments[chunk_id] = segment
        return segment

    def record_stale_face_drop(self, chunk_id: str) -> None:
        if chunk_id in self._stale_counted_segments:
            return
        self._stale_counted_segments.add(chunk_id)
        self.stale_face_drop_count += 1
        segment = self.segments.get(chunk_id)
        if segment is not None:
            segment.stale_dropped = True

    def snapshot(self) -> dict[str, object]:
        return {
            "segments": [
                segment.snapshot()
                for segment in sorted(self.segments.values(), key=lambda item: item.chunk_index)
            ],
            "old_turn_face_leak_count": self.old_turn_face_leak_count,
            "stale_face_drop_count": self.stale_face_drop_count,
        }


def _duration_ms(start_ns: int, end_ns: int) -> float:
    return round((end_ns - start_ns) / 1_000_000.0, 3)


@dataclass
class StreamOrchestrator:
    settings: AppSettings
    registry: AdapterRegistry
    store: ArtifactStore
    history: ConversationHistoryStore | None = None

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
        stream_timing = _StreamTiming(run_started_ns=monotonic_ns())
        face_stitcher = FaceSegmentStitcher(
            enabled=self.settings.face_stitching.enabled,
            overlap_frames=self.settings.face_stitching.overlap_frames,
        )
        eye_continuity = EyeContinuityProcessor(
            enabled=self.settings.eye_continuity.enabled,
            eye_smooth_channel_indices=self.settings.eye_continuity.eye_smooth_channel_indices,
            blink_enabled=self.settings.eye_continuity.blink_enabled,
            blink_channel_indices=self.settings.eye_continuity.blink_channel_indices,
            overlap_frames=self.settings.eye_continuity.overlap_frames,
            blink_interval_min_sec=self.settings.eye_continuity.blink_interval_min_sec,
            blink_interval_max_sec=self.settings.eye_continuity.blink_interval_max_sec,
            blink_duration_frames=self.settings.eye_continuity.blink_duration_frames,
            blink_strength=self.settings.eye_continuity.blink_strength,
            seed=self.settings.eye_continuity.seed,
            reset_blink_on_new_turn=self.settings.eye_continuity.reset_blink_on_new_turn,
        )
        marks: set[str] = set()
        artifacts = _StreamArtifacts()
        face_tasks: dict[int, asyncio.Task[_RawFaceSegmentResult]] = {}
        emitted_face_results: list[_FaceSegmentResult] = []
        pending_llm_event: asyncio.Task[object] | None = None
        next_face_emit_index = 1
        turn_dir = self.store.create_turn(turn.session_id, turn.turn_id)
        context = TurnContext(
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            artifact_dir=turn_dir,
            cancellation=turn.cancellation,
            generation_epoch=turn.generation_epoch,
        )

        def mark_once(name: str) -> None:
            if name not in marks:
                timeline.mark(name)
                marks.add(name)

        def timeline_snapshot() -> dict[str, object]:
            snapshot = timeline.snapshot()
            snapshot["stream"] = stream_timing.snapshot()
            return snapshot

        async def schedule_segment(segment: str, chunk_index: int, llm: LLMResult) -> None:
            chunk_id, audio, segment_timing = await self._process_audio_segment(
                segment,
                chunk_index,
                llm,
                turn,
                context,
                factory,
                timeline,
                mark_once,
                artifacts,
                stream_timing,
                emit_json,
                emit_binary_pair,
            )
            face_tasks[chunk_index] = asyncio.create_task(
                self._drive_face_segment(
                    chunk_index,
                    chunk_id,
                    audio,
                    llm,
                    turn,
                    context,
                    timeline,
                    stream_timing,
                    segment_timing,
                )
            )
            emitted_face_results.extend(await drain_face_segments(block=False))

        async def drain_face_segments(*, block: bool) -> list[_FaceSegmentResult]:
            nonlocal next_face_emit_index
            emitted: list[_FaceSegmentResult] = []
            while True:
                task = face_tasks.get(next_face_emit_index)
                if task is None:
                    return emitted
                if task.done():
                    raw_result = task.result()
                elif block:
                    raw_result = await task
                else:
                    return emitted
                emitted.append(
                    await self._postprocess_and_emit_face_segment(
                        raw_result.chunk_index,
                        raw_result.chunk_id,
                        raw_result.face,
                        turn,
                        context,
                        factory,
                        timeline,
                        mark_once,
                        stream_timing,
                        raw_result.segment_timing,
                        face_stitcher,
                        eye_continuity,
                        emit_json,
                    )
                )
                next_face_emit_index += 1

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
            await self._emit_server_json(
                turn,
                emit_json,
                factory,
                EventType.SERVER_ASR_FINAL,
                {
                    "text": artifacts.asr.text,
                    "language": artifacts.asr.language,
                    "confidence": artifacts.asr.confidence,
                },
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
                    if pending_llm_event is None:
                        pending_llm_event = asyncio.create_task(iterator.__anext__())

                    done, _ = await asyncio.wait(
                        {pending_llm_event},
                        timeout=self.settings.stream.sentence_max_wait_ms / 1000.0,
                    )
                    if not done:
                        segment = buffer.flush()
                        if segment is not None:
                            chunk_index += 1
                            await schedule_segment(segment, chunk_index, fallback_llm)
                        continue

                    try:
                        event = pending_llm_event.result()
                    except StopAsyncIteration:
                        pending_llm_event = None
                        break
                    else:
                        pending_llm_event = None

                    self._ensure_current(turn)
                    if event.kind == "token":
                        mark_once("llm_first_token")
                        reply_parts.append(event.text)
                        await self._emit_server_json(
                            turn,
                            emit_json,
                            factory,
                            EventType.SERVER_LLM_TOKEN,
                            {"text": event.text},
                        )
                        for segment in buffer.push(event.text):
                            chunk_index += 1
                            await schedule_segment(segment, chunk_index, fallback_llm)
                    elif event.kind == "final" and event.result is not None:
                        artifacts.llm = event.result
                        fallback_llm = event.result

            residual = buffer.flush()
            if residual is not None:
                chunk_index += 1
                await schedule_segment(residual, chunk_index, fallback_llm)

            if artifacts.llm is None:
                artifacts.llm = LLMResult(
                    reply="".join(reply_parts),
                    emotion=fallback_llm.emotion,
                    intensity=fallback_llm.intensity,
                )
            if face_tasks:
                emitted_face_results.extend(await drain_face_segments(block=True))
                latest_face_result = max(emitted_face_results, key=lambda result: result.chunk_index)
                artifacts.face = latest_face_result.face
                artifacts.ue5 = latest_face_result.ue5
            self._ensure_complete(artifacts)
            snapshot = timeline_snapshot()
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
            if pending_llm_event is not None and not pending_llm_event.done():
                pending_llm_event.cancel()
                await asyncio.gather(pending_llm_event, return_exceptions=True)
            pending_face_tasks = [task for task in face_tasks.values() if not task.done()]
            for task in pending_face_tasks:
                task.cancel()
            if face_tasks:
                await asyncio.gather(*face_tasks.values(), return_exceptions=True)
            self.store.write_json(turn_dir / "timeline.json", timeline_snapshot())

    async def _process_audio_segment(
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
        stream_timing: _StreamTiming,
        emit_json: EmitJSON,
        emit_binary_pair: EmitBinaryPair,
    ) -> tuple[str, AudioArtifact, _StreamSegmentTiming]:
        chunk_id = f"chunk-{chunk_index:04d}"
        await self._emit_server_json(
            turn,
            emit_json,
            factory,
            EventType.SERVER_LLM_CHUNK,
            {"chunk_id": chunk_id, "text": segment},
        )

        with timeline.stage("tts", self.registry.tts.name):
            self._ensure_current(turn)
            audio = await self.registry.tts.synthesize(segment, llm.emotion, llm.intensity, context)
        artifacts.audio = audio
        tts_ready_ns = monotonic_ns()
        segment_timing = stream_timing.add_segment(
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            turn=turn,
            tts_ready_ns=tts_ready_ns,
        )
        mark_once("first_tts_ready")
        await self._emit_server_binary_pair(
            turn,
            emit_binary_pair,
            factory,
            EventType.SERVER_TTS_AUDIO,
            {
                "chunk_id": chunk_id,
                "format": "wav",
                "sample_rate": audio.sample_rate,
                "byte_length": audio.byte_length,
                "duration_seconds": audio.duration_seconds,
                "segment_id": segment_timing.segment_id,
                "segment_index": chunk_index,
                "timing": segment_timing.timing_payload(),
            },
            audio.path.read_bytes(),
        )
        return chunk_id, audio, segment_timing

    async def _drive_face_segment(
        self,
        chunk_index: int,
        chunk_id: str,
        audio: AudioArtifact,
        llm: LLMResult,
        turn: TurnHandle,
        context: TurnContext,
        timeline: Timeline,
        stream_timing: _StreamTiming,
        segment_timing: _StreamSegmentTiming,
    ) -> _RawFaceSegmentResult:
        try:
            face_started_ns = monotonic_ns()
            segment_timing.face_start_after_tts_ms = stream_timing.after_tts_ms(
                segment_timing,
                face_started_ns,
            )
            with timeline.stage("audio2face", self.registry.audio2face.name):
                self._ensure_current(turn)
                face = await self.registry.audio2face.drive(audio, llm.emotion, llm.intensity, context)
            segment_timing.face_total_ms = _duration_ms(face_started_ns, monotonic_ns())
            return _RawFaceSegmentResult(
                chunk_index=chunk_index,
                chunk_id=chunk_id,
                face=face,
                segment_timing=segment_timing,
            )
        except asyncio.CancelledError:
            stream_timing.record_stale_face_drop(chunk_id)
            raise

    async def _postprocess_and_emit_face_segment(
        self,
        chunk_index: int,
        chunk_id: str,
        face: FaceArtifact,
        turn: TurnHandle,
        context: TurnContext,
        factory: EventFactory,
        timeline: Timeline,
        mark_once: Callable[[str], None],
        stream_timing: _StreamTiming,
        segment_timing: _StreamSegmentTiming,
        face_stitcher: FaceSegmentStitcher,
        eye_continuity: EyeContinuityProcessor,
        emit_json: EmitJSON,
    ) -> _FaceSegmentResult:
        try:
            self._ensure_current(turn)
            stitched_frames, stitch_metrics = face_stitcher.stitch(
                face.frames,
                session_id=str(turn.session_id),
                turn_id=str(turn.turn_id),
                generation_epoch=turn.generation_epoch,
                segment_index=chunk_index,
            )
            segment_timing.apply_stitch_metrics(
                stitch_metrics,
                record_boundary_metrics=self.settings.face_stitching.record_boundary_metrics,
            )
            if stitched_frames != face.frames:
                face = face.model_copy(
                    update={
                        "frames": stitched_frames,
                        "frame_count": len(stitched_frames),
                    }
                )
            eye_frames, eye_metrics = eye_continuity.process(
                face.frames,
                session_id=str(turn.session_id),
                turn_id=str(turn.turn_id),
                generation_epoch=turn.generation_epoch,
                segment_index=chunk_index,
                fps=face.fps,
            )
            segment_timing.apply_eye_continuity_metrics(
                eye_metrics,
                record_boundary_metrics=self.settings.eye_continuity.record_boundary_metrics,
            )
            if eye_frames != face.frames:
                face = face.model_copy(
                    update={
                        "frames": eye_frames,
                        "frame_count": len(eye_frames),
                    }
                )
            mark_once("first_face_ready")
            await self._emit_server_json(
                turn,
                emit_json,
                factory,
                EventType.SERVER_FACE_FRAMES,
                {
                    "chunk_id": chunk_id,
                    "segment_id": segment_timing.segment_id,
                    "segment_index": chunk_index,
                    "fps": face.fps,
                    "frame_count": face.frame_count,
                    "timing": segment_timing.timing_payload(),
                    "frames": face.frames,
                },
            )

            with timeline.stage("ue5", self.registry.ue5.name):
                self._ensure_current(turn)
                ue5 = await self.registry.ue5.format(face, context)
            for chunk in chunk_ue5_frames(ue5, chunk_size=30, chunk_id=chunk_id):
                if segment_timing.ue5_first_frame_after_tts_ms is None:
                    first_ue5_ns = monotonic_ns()
                    segment_timing.ue5_first_frame_after_tts_ms = stream_timing.after_tts_ms(
                        segment_timing,
                        first_ue5_ns,
                    )
                    segment_timing.e2e_first_visible_face_ms = stream_timing.elapsed_ms(first_ue5_ns)
                chunk["segment_id"] = segment_timing.segment_id
                chunk["segment_index"] = chunk_index
                chunk["turn_id"] = str(turn.turn_id)
                chunk["generation_epoch"] = turn.generation_epoch
                chunk["timing"] = segment_timing.timing_payload()
                await self._emit_server_json(
                    turn,
                    emit_json,
                    factory,
                    EventType.SERVER_UE5_FRAMES,
                    chunk,
                )

            mark_once("first_segment_ready")
            await self._emit_server_json(
                turn,
                emit_json,
                factory,
                EventType.SERVER_SEGMENT_READY,
                {
                    "chunk_id": chunk_id,
                    "segment_id": segment_timing.segment_id,
                    "segment_index": chunk_index,
                    "timing": segment_timing.timing_payload(),
                },
            )
            return _FaceSegmentResult(chunk_index=chunk_index, face=face, ue5=ue5)
        except asyncio.CancelledError:
            stream_timing.record_stale_face_drop(chunk_id)
            raise

    async def _emit_server_json(
        self,
        turn: TurnHandle,
        emit_json: EmitJSON,
        factory: EventFactory,
        event_type: EventType,
        payload: dict[str, object],
    ) -> None:
        async def operation() -> None:
            envelope = factory.server(event_type, turn.turn_id, payload)
            await emit_json(envelope)

        if not await turn.emit_if_current(operation):
            raise asyncio.CancelledError

    async def _emit_server_binary_pair(
        self,
        turn: TurnHandle,
        emit_binary_pair: EmitBinaryPair,
        factory: EventFactory,
        event_type: EventType,
        payload: dict[str, object],
        binary: bytes,
    ) -> None:
        async def operation() -> None:
            envelope = factory.server(event_type, turn.turn_id, payload)
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
