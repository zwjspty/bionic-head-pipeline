from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from bionic_head.adapters.registry import AdapterRegistry, build_registry
from bionic_head.domain.models import (
    AudioArtifact,
    DiagnosticResult,
    Emotion,
    FaceArtifact,
    LLMEvent,
    LLMResult,
    TurnContext,
)


class _OutOfOrderAudio2FaceAdapter:
    name = "out-of-order-face"

    def __init__(self) -> None:
        self.call_count = 0

    async def drive(
        self,
        audio: AudioArtifact,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> FaceArtifact:
        self.call_count += 1
        call_index = self.call_count
        if call_index == 1:
            await asyncio.sleep(0.05)
        frame = [float(call_index)] * 52
        return FaceArtifact(
            frames=[frame],
            fps=30,
            channel_count=52,
            frame_count=1,
        )

    async def diagnostics(self) -> DiagnosticResult:
        return DiagnosticResult(
            adapter="audio2face",
            provider=self.name,
            available=True,
            latency_ms=0.0,
            message="test adapter ready",
        )

    async def cancel(self, turn_id) -> None:
        await asyncio.sleep(0)


class _StalingAudio2FaceAdapter:
    name = "staling-face"

    def __init__(self, stale_turn) -> None:
        self._stale_turn = stale_turn
        self.call_count = 0

    async def drive(
        self,
        audio: AudioArtifact,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> FaceArtifact:
        self.call_count += 1
        self._stale_turn()
        await asyncio.sleep(0.01)
        return FaceArtifact(
            frames=[[1.0] * 52],
            fps=30,
            channel_count=52,
            frame_count=1,
        )

    async def diagnostics(self) -> DiagnosticResult:
        return DiagnosticResult(
            adapter="audio2face",
            provider=self.name,
            available=True,
            latency_ms=0.0,
            message="test adapter ready",
        )

    async def cancel(self, turn_id) -> None:
        await asyncio.sleep(0)


class _ConstantSequenceAudio2FaceAdapter:
    name = "constant-sequence-face"

    def __init__(self) -> None:
        self.call_count = 0

    async def drive(
        self,
        audio: AudioArtifact,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> FaceArtifact:
        self.call_count += 1
        value = float(self.call_count - 1)
        frames = [[value] * 52 for _ in range(4)]
        return FaceArtifact(
            frames=frames,
            fps=30,
            channel_count=52,
            frame_count=len(frames),
        )

    async def diagnostics(self) -> DiagnosticResult:
        return DiagnosticResult(
            adapter="audio2face",
            provider=self.name,
            available=True,
            latency_ms=0.0,
            message="test adapter ready",
        )

    async def cancel(self, turn_id) -> None:
        await asyncio.sleep(0)


class _DelayedTokenLLMAdapter:
    name = "delayed-token-llm"

    def __init__(self) -> None:
        self.call_count = 0

    async def chat(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> LLMResult:
        return LLMResult(reply="你好", emotion=Emotion.FRIENDLY, intensity=0.8)

    async def _chat_stream(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> AsyncIterator[LLMEvent]:
        self.call_count += 1
        yield LLMEvent(kind="token", text="你")
        await asyncio.sleep(0.05)
        yield LLMEvent(kind="token", text="好")
        yield LLMEvent(
            kind="final",
            result=LLMResult(reply="你好", emotion=Emotion.FRIENDLY, intensity=0.8),
        )

    def chat_stream(
        self,
        text: str,
        history: list[dict[str, str]],
        context: TurnContext,
    ) -> AsyncIterator[LLMEvent]:
        return self._chat_stream(text, history, context)

    async def diagnostics(self) -> DiagnosticResult:
        return DiagnosticResult(
            adapter="llm",
            provider=self.name,
            available=True,
            latency_ms=0.0,
            message="test adapter ready",
        )

    async def cancel(self, turn_id) -> None:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_stream_emits_audio_before_face_then_segment_ready(stream_harness) -> None:
    await stream_harness.run()

    types = stream_harness.json_types
    assert types.index("server.asr.final") < types.index("server.llm.token")
    assert types.index("server.tts.audio") < types.index("server.face.frames")
    assert types.index("server.face.frames") < types.index("server.segment.ready")
    assert types[-1] == "server.pipeline.done"
    assert stream_harness.terminal_types == ["server.pipeline.done"]
    assert len(stream_harness.binary_frames) >= 1
    assert len(stream_harness.binary_frames) == types.count("server.tts.audio")


@pytest.mark.asyncio
async def test_stream_records_face_segment_timing_and_ue5_payload_timing(stream_harness) -> None:
    await stream_harness.run()

    timeline_path = (
        stream_harness.store.runs
        / str(stream_harness.turn.session_id)
        / str(stream_harness.turn.turn_id)
        / "timeline.json"
    )
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    stream = timeline["stream"]
    segments = stream["segments"]
    assert stream["old_turn_face_leak_count"] == 0
    assert stream["stale_face_drop_count"] == 0
    assert len(segments) >= 1
    first = segments[0]
    assert first["segment_id"] == first["chunk_id"] == "chunk-0001"
    assert first["turn_id"] == str(stream_harness.turn.turn_id)
    assert first["generation_epoch"] == 0
    for key in [
        "tts_audio_ready_ms",
        "face_start_after_tts_ms",
        "face_total_ms",
        "ue5_first_frame_after_tts_ms",
        "e2e_first_visible_face_ms",
    ]:
        assert isinstance(first[key], float)
        assert first[key] >= 0.0

    ue5 = next(
        envelope
        for envelope in stream_harness.json_envelopes
        if envelope.type.value == "server.ue5.frames"
    )
    assert ue5.payload["segment_id"] == "chunk-0001"
    assert ue5.payload["segment_index"] == 1
    assert ue5.payload["timing"]["face_total_ms"] == first["face_total_ms"]


@pytest.mark.asyncio
async def test_stream_applies_face_stitching_to_second_segment_and_records_boundary_metrics(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.mock.reply = "第一段内容已经足够。第二段内容也足够。"
    settings.stream.sentence_min_chars = 4
    settings.stream.sentence_max_chars = 12
    settings.face_stitching.enabled = True
    settings.face_stitching.overlap_frames = 2
    registry = build_registry(settings)
    registry = AdapterRegistry(
        asr=registry.asr,
        llm=registry.llm,
        tts=registry.tts,
        audio2face=_ConstantSequenceAudio2FaceAdapter(),
        ue5=registry.ue5,
    )
    harness = stream_harness_factory(settings=settings, registry=registry)

    await harness.run()

    timeline_path = (
        harness.store.runs
        / str(harness.turn.session_id)
        / str(harness.turn.turn_id)
        / "timeline.json"
    )
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    segments = timeline["stream"]["segments"]
    stitched_segments = [
        segment for segment in segments if segment.get("face_stitch_applied") is True
    ]

    assert stitched_segments
    assert stitched_segments[0]["face_stitch_actual_overlap_frames"] == 2.0
    assert stitched_segments[0]["face_boundary_delta_before"] == pytest.approx(1.0)
    assert stitched_segments[0]["face_boundary_delta_after"] == pytest.approx(0.5)
    assert timeline["stream"]["old_turn_face_leak_count"] == 0

    ue5_payloads = [
        envelope.payload
        for envelope in harness.json_envelopes
        if envelope.type.value == "server.ue5.frames"
    ]
    assert any(payload["timing"].get("face_stitch_applied") is True for payload in ue5_payloads)


@pytest.mark.asyncio
async def test_stream_does_not_block_later_tts_on_slow_face(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.mock.reply = "第一段内容已经足够。第二段内容也足够。"
    settings.mock.latency_ms.face = 100
    settings.stream.sentence_min_chars = 4
    settings.stream.sentence_max_chars = 12
    registry = build_registry(settings)
    harness = stream_harness_factory(settings=settings, registry=registry)

    await harness.run()

    types = harness.json_types
    tts_indexes = [index for index, event_type in enumerate(types) if event_type == "server.tts.audio"]
    first_face_index = types.index("server.face.frames")
    assert len(tts_indexes) >= 2
    assert tts_indexes[1] < first_face_index


@pytest.mark.asyncio
async def test_stream_llm_timeout_flushes_without_cancelling_later_tokens(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.stream.sentence_max_wait_ms = 10
    settings.stream.sentence_min_chars = 1
    registry = build_registry(settings)
    registry = AdapterRegistry(
        asr=registry.asr,
        llm=_DelayedTokenLLMAdapter(),
        tts=registry.tts,
        audio2face=registry.audio2face,
        ue5=registry.ue5,
    )
    harness = stream_harness_factory(settings=settings, registry=registry)

    await harness.run()

    tokens = [
        str(envelope.payload["text"])
        for envelope in harness.json_envelopes
        if envelope.type.value == "server.llm.token"
    ]
    assert tokens == ["你", "好"]
    assert harness.terminal_types == ["server.pipeline.done"]


@pytest.mark.asyncio
async def test_stream_latest_uses_highest_chunk_when_face_finishes_out_of_order(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.mock.reply = "第一段内容已经足够。第二段内容也足够。"
    settings.stream.sentence_min_chars = 4
    settings.stream.sentence_max_chars = 12
    registry = build_registry(settings)
    registry = AdapterRegistry(
        asr=registry.asr,
        llm=registry.llm,
        tts=registry.tts,
        audio2face=_OutOfOrderAudio2FaceAdapter(),
        ue5=registry.ue5,
    )
    harness = stream_harness_factory(settings=settings, registry=registry)

    await harness.run()

    latest = json.loads((harness.store.latest / "latest_pipeline.json").read_text(encoding="utf-8"))
    latest_ue5 = json.loads((harness.store.latest / "latest_ue5_blendshape.json").read_text(encoding="utf-8"))
    assert latest["face"]["frames"][0][0] == 2.0
    assert latest_ue5["frames"][0]["weights"][0] == 2.0


@pytest.mark.asyncio
async def test_stream_background_face_failure_emits_error_without_latest(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.mock.fail_stage = "audio2face"
    harness = stream_harness_factory(settings=settings, registry=build_registry(settings))

    await harness.run()

    assert harness.terminal_types == ["server.pipeline.error"]
    assert "server.pipeline.done" not in harness.json_types
    assert not (harness.store.latest / "latest_pipeline.json").exists()
    assert not (harness.store.latest / "latest_ue5_blendshape.json").exists()


@pytest.mark.asyncio
async def test_stream_cancel_after_tts_suppresses_background_face_and_latest(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.mock.latency_ms.face = 50
    harness = stream_harness_factory(settings=settings, registry=build_registry(settings))
    emit_binary_pair = harness.emit_binary_pair

    async def emit_then_cancel(envelope, binary: bytes) -> None:
        await emit_binary_pair(envelope, binary)
        asyncio.create_task(harness.turn.cancel())

    harness.emit_binary_pair = emit_then_cancel

    await harness.run()

    assert harness.terminal_types == ["server.turn.cancelled"]
    assert "server.face.frames" not in harness.json_types
    assert "server.ue5.frames" not in harness.json_types
    timeline_path = (
        harness.store.runs
        / str(harness.turn.session_id)
        / str(harness.turn.turn_id)
        / "timeline.json"
    )
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    assert timeline["stream"]["old_turn_face_leak_count"] == 0
    assert timeline["stream"]["stale_face_drop_count"] >= 1
    assert not (harness.store.latest / "latest_pipeline.json").exists()
    assert not (harness.store.latest / "latest_ue5_blendshape.json").exists()


@pytest.mark.asyncio
async def test_stream_stale_epoch_before_face_task_creation_suppresses_background_face_and_latest(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.mock.latency_ms.face = 50
    harness = stream_harness_factory(settings=settings, registry=build_registry(settings))
    current_epoch = 0
    harness.turn.generation_epoch = current_epoch
    harness.turn.generation_epoch_getter = lambda: current_epoch
    emit_binary_pair = harness.emit_binary_pair

    async def emit_then_stale(envelope, binary: bytes) -> None:
        nonlocal current_epoch
        await emit_binary_pair(envelope, binary)
        current_epoch += 1

    harness.emit_binary_pair = emit_then_stale

    await harness.run()

    assert harness.terminal_types == ["server.turn.cancelled"]
    assert "server.face.frames" not in harness.json_types
    assert "server.ue5.frames" not in harness.json_types
    assert "server.segment.ready" not in harness.json_types
    assert _server_sequences_are_contiguous(harness)
    assert not (harness.store.latest / "latest_pipeline.json").exists()
    assert not (harness.store.latest / "latest_ue5_blendshape.json").exists()


@pytest.mark.asyncio
async def test_stream_stale_epoch_while_background_face_running_suppresses_face_and_latest(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    current_epoch = 0

    def stale_turn() -> None:
        nonlocal current_epoch
        current_epoch += 1

    registry = build_registry(settings)
    registry = AdapterRegistry(
        asr=registry.asr,
        llm=registry.llm,
        tts=registry.tts,
        audio2face=_StalingAudio2FaceAdapter(stale_turn),
        ue5=registry.ue5,
    )
    harness = stream_harness_factory(settings=settings, registry=registry)
    harness.turn.generation_epoch = current_epoch
    harness.turn.generation_epoch_getter = lambda: current_epoch

    await harness.run()

    assert harness.terminal_types == ["server.turn.cancelled"]
    assert "server.face.frames" not in harness.json_types
    assert "server.ue5.frames" not in harness.json_types
    assert "server.segment.ready" not in harness.json_types
    assert _server_sequences_are_contiguous(harness)
    assert not (harness.store.latest / "latest_pipeline.json").exists()
    assert not (harness.store.latest / "latest_ue5_blendshape.json").exists()


@pytest.mark.asyncio
async def test_stream_provider_failure_emits_one_error(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.mock.fail_stage = "tts"
    harness = stream_harness_factory(settings=settings, registry=build_registry(settings))

    await harness.run()

    assert harness.terminal_types == ["server.pipeline.error"]
    assert harness.json_envelopes[-1].payload["error"]["code"] == "provider_failed"


@pytest.mark.asyncio
async def test_stream_provider_timeout_emits_error(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.tts.timeout_seconds = 0.01
    settings.mock.timeout_stage = "tts"
    harness = stream_harness_factory(settings=settings, registry=build_registry(settings))

    await harness.run()

    assert harness.terminal_types == ["server.pipeline.error"]
    assert harness.json_envelopes[-1].payload["error"]["code"] == "provider_timeout"


@pytest.mark.asyncio
async def test_stream_cancelled_turn_emits_cancelled(stream_harness) -> None:
    await stream_harness.turn.cancel()

    await stream_harness.run()

    assert stream_harness.terminal_types == ["server.turn.cancelled"]


def _server_sequences_are_contiguous(harness) -> bool:
    sequences = [envelope.sequence for envelope in harness.json_envelopes]
    return sequences == list(range(1, len(sequences) + 1))
