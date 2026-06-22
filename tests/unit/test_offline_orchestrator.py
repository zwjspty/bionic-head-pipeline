from __future__ import annotations

import asyncio
import json

import pytest

from bionic_head.adapters.registry import build_registry
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.orchestrators.offline import OfflineOrchestrator


@pytest.mark.asyncio
async def test_offline_pipeline_writes_all_artifacts(
    offline_orchestrator,
    artifact_store,
    speech_wav,
    turn_context,
) -> None:
    result = await offline_orchestrator.run(speech_wav, turn_context)

    assert result.asr.text == "你好"
    assert result.face.channel_count == 52
    assert (turn_context.artifact_dir / "input.wav").exists()
    assert (turn_context.artifact_dir / "asr.json").exists()
    assert (turn_context.artifact_dir / "llm.json").exists()
    assert result.audio.path.exists()
    assert result.face.path is not None
    assert result.face.path.exists()
    assert (turn_context.artifact_dir / "timeline.json").exists()
    assert (turn_context.artifact_dir / "ue5/result.json").exists()
    assert (artifact_store.latest / "latest_pipeline.json").exists()
    assert (artifact_store.latest / "latest_ue5_blendshape.json").exists()

    timeline = json.loads((turn_context.artifact_dir / "timeline.json").read_text(encoding="utf-8"))
    assert [stage["name"] for stage in timeline["stages"]] == [
        "asr",
        "llm",
        "tts",
        "audio2face",
        "ue5",
    ]
    assert {stage["status"] for stage in timeline["stages"]} == {"completed"}


@pytest.mark.asyncio
async def test_silence_stops_before_llm(offline_orchestrator, silence_wav, turn_context, mock_registry) -> None:
    with pytest.raises(PipelineException) as raised:
        await offline_orchestrator.run(silence_wav, turn_context)

    assert raised.value.code is ErrorCode.NO_SPEECH_DETECTED
    assert mock_registry.asr.call_count == 0
    assert mock_registry.llm.call_count == 0
    assert (turn_context.artifact_dir / "timeline.json").exists()


@pytest.mark.asyncio
async def test_provider_failure_writes_failed_timeline_and_skips_downstream(
    mock_settings,
    artifact_store,
    speech_wav,
    turn_context,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.mock.fail_stage = "tts"
    registry = build_registry(settings)
    orchestrator = _orchestrator(settings, registry, artifact_store, publish=True)

    with pytest.raises(PipelineException) as raised:
        await orchestrator.run(speech_wav, turn_context)

    assert raised.value.code is ErrorCode.PROVIDER_FAILED
    assert registry.tts.call_count == 0
    assert registry.audio2face.call_count == 0
    assert not (artifact_store.latest / "latest_pipeline.json").exists()
    timeline = json.loads((turn_context.artifact_dir / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["stages"][-1]["name"] == "tts"
    assert timeline["stages"][-1]["status"] == "failed"
    assert timeline["stages"][-1]["error_code"] == "provider_failed"


@pytest.mark.asyncio
async def test_provider_timeout_is_recorded(
    mock_settings,
    artifact_store,
    speech_wav,
    turn_context,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.tts.timeout_seconds = 0.01
    settings.mock.timeout_stage = "tts"
    registry = build_registry(settings)
    orchestrator = _orchestrator(settings, registry, artifact_store, publish=True)

    with pytest.raises(PipelineException) as raised:
        await orchestrator.run(speech_wav, turn_context)

    assert raised.value.code is ErrorCode.PROVIDER_TIMEOUT
    timeline = json.loads((turn_context.artifact_dir / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["stages"][-1]["name"] == "tts"
    assert timeline["stages"][-1]["error_code"] == "provider_timeout"


@pytest.mark.asyncio
async def test_stale_turn_does_not_publish_latest(
    mock_settings,
    mock_registry,
    artifact_store,
    speech_wav,
    turn_context,
) -> None:
    orchestrator = _orchestrator(mock_settings, mock_registry, artifact_store, publish=False)

    result = await orchestrator.run(speech_wav, turn_context)

    assert result.asr.text == "你好"
    assert not (artifact_store.latest / "latest_pipeline.json").exists()
    assert not (artifact_store.latest / "latest_ue5_blendshape.json").exists()
    assert (turn_context.artifact_dir / "ue5/result.json").exists()


@pytest.mark.asyncio
async def test_cancelled_turn_writes_timeline_but_does_not_publish_latest(
    offline_orchestrator,
    artifact_store,
    speech_wav,
    turn_context,
) -> None:
    turn_context.cancellation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await offline_orchestrator.run(speech_wav, turn_context)

    timeline = json.loads((turn_context.artifact_dir / "timeline.json").read_text(encoding="utf-8"))
    assert "cancelled" in timeline["marks"]
    assert not (artifact_store.latest / "latest_pipeline.json").exists()


def _orchestrator(settings, registry, artifact_store, *, publish: bool) -> OfflineOrchestrator:
    async def commit_if_current(_session_id, _turn_id, callback):
        if publish:
            callback()
        return publish

    return OfflineOrchestrator(
        settings=settings,
        registry=registry,
        store=artifact_store,
        commit_if_current=commit_if_current,
    )
