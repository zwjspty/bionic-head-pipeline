from pathlib import Path
import sys

import pytest

from bionic_head.adapters.registry import build_registry
from bionic_head.config import load_settings
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import Emotion


def test_registry_allows_mock_asr_and_real_ollama(mock_settings) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.llm.provider = "ollama"
    settings.adapters.ue5.provider = "morpheus-raw"

    registry = build_registry(settings)

    assert registry.asr.name == "mock"
    assert registry.llm.name == "ollama"
    assert registry.ue5.name == "morpheus-raw"


def test_registry_constructs_real_example_without_running_expensive_diagnostics() -> None:
    settings = load_settings(Path("config/real.example.json"))

    registry = build_registry(settings)

    assert registry.asr.name == "faster-whisper"
    assert registry.llm.name == "ollama"
    assert registry.tts.name == "piper"
    assert registry.audio2face.name == "morpheus"
    assert registry.ue5.name == "morpheus-raw"


@pytest.mark.asyncio
async def test_known_but_unconfigured_piper_raises_provider_unavailable(
    turn_context,
) -> None:
    settings = load_settings(Path("config/real.example.json"))
    registry = build_registry(settings)

    with pytest.raises(PipelineException) as raised:
        await registry.tts.synthesize("你好", Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert raised.value.provider == "piper"


def test_registry_builds_emotalk_sidecar_without_starting_subprocess(mock_settings) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.audio2face.provider = "emotalk_sidecar"
    settings.adapters.ue5.provider = "morpheus-raw"
    settings.providers.emotalk_sidecar.sidecar_command = [
        sys.executable,
        "-m",
        "bionic_head.emotalk_fake_sidecar",
    ]

    registry = build_registry(settings)

    assert registry.audio2face.name == "emotalk_sidecar"
    assert registry.audio2face.process_start_count == 0
