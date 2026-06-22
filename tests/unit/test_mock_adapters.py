from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from bionic_head.adapters.registry import build_registry
from bionic_head.config import load_settings
from bionic_head.core.audio import inspect_wav
from bionic_head.domain.errors import ErrorCode, PipelineException


@pytest.mark.asyncio
async def test_mock_chain_returns_deterministic_results(speech_wav, turn_context) -> None:
    registry = build_registry(load_settings(Path("config/mock.json")))

    asr = await registry.asr.transcribe(speech_wav, turn_context)
    llm = await registry.llm.chat(asr.text, [], turn_context)
    audio = await registry.tts.synthesize(llm.reply, llm.emotion, llm.intensity, turn_context)
    face = await registry.audio2face.drive(audio, llm.emotion, llm.intensity, turn_context)
    ue5 = await registry.ue5.format(face, turn_context)

    assert asr.text == "你好"
    assert llm.reply == "你好！很高兴见到你。"
    assert llm.emotion.value == "friendly"
    assert llm.intensity == pytest.approx(0.8)
    assert inspect_wav(audio.path).duration_seconds == pytest.approx(0.25)
    assert face.frame_count == 8
    assert len(face.frames[0]) == 52
    assert ue5.format == "morpheus_52_raw"
    assert ue5.frame_count == face.frame_count
    assert registry.asr.call_count == 1
    assert registry.llm.call_count == 1
    assert registry.tts.call_count == 1
    assert registry.audio2face.call_count == 1
    assert registry.ue5.call_count == 1


@pytest.mark.asyncio
async def test_mock_llm_stream_emits_character_tokens_and_final(mock_registry, turn_context) -> None:
    events = [
        event
        async for event in mock_registry.llm.chat_stream("你好", [], turn_context)
    ]

    assert "".join(event.text for event in events if event.kind == "token") == "你好！很高兴见到你。"
    assert events[-1].kind == "final"
    assert events[-1].result is not None
    assert events[-1].result.emotion.value == "friendly"
    assert mock_registry.llm.chat_call_count == 0
    assert mock_registry.llm.chat_stream_call_count == 1


@pytest.mark.asyncio
async def test_configured_failure_maps_to_provider_failed(mock_settings, speech_wav, turn_context) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.mock.fail_stage = "asr"
    registry = build_registry(settings)

    with pytest.raises(PipelineException) as captured:
        await registry.asr.transcribe(speech_wav, turn_context)

    assert captured.value.code is ErrorCode.PROVIDER_FAILED
    assert captured.value.stage == "asr"
    assert captured.value.provider == "mock"


@pytest.mark.asyncio
async def test_configured_timeout_maps_to_provider_timeout(mock_settings, speech_wav, turn_context) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.asr.timeout_seconds = 0.01
    settings.mock.timeout_stage = "asr"
    registry = build_registry(settings)

    with pytest.raises(PipelineException) as captured:
        await registry.asr.transcribe(speech_wav, turn_context)

    assert captured.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert captured.value.stage == "asr"
    assert captured.value.provider == "mock"


@pytest.mark.asyncio
async def test_cancellation_propagates_without_provider_failed(mock_registry, speech_wav, turn_context) -> None:
    turn_context.cancellation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await mock_registry.asr.transcribe(speech_wav, turn_context)

    assert mock_registry.asr.call_count == 0
