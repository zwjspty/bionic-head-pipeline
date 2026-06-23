from __future__ import annotations

import asyncio
import json

import pytest

from bionic_head.adapters.registry import AdapterRegistry, build_registry
from bionic_head.domain.models import AudioArtifact, DiagnosticResult, Emotion, FaceArtifact, TurnContext


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
