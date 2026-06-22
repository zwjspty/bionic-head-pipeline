from __future__ import annotations

import pytest

from bionic_head.adapters.registry import build_registry


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
