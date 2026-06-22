from __future__ import annotations

import pytest

from bionic_head.adapters.registry import AdapterRegistry, build_registry
from bionic_head.domain.errors import ErrorCode, PipelineException


def test_build_registry_returns_all_mock_adapters(mock_settings) -> None:
    registry = build_registry(mock_settings)

    assert isinstance(registry, AdapterRegistry)
    assert registry.asr.name == "mock"
    assert registry.llm.name == "mock"
    assert registry.tts.name == "mock"
    assert registry.audio2face.name == "mock"
    assert registry.ue5.name == "mock"


def test_unknown_provider_fails_at_startup(mock_settings) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.tts.provider = "piper"

    with pytest.raises(PipelineException) as captured:
        build_registry(settings)

    assert captured.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert captured.value.stage == "startup"
    assert captured.value.provider == "piper"


@pytest.mark.asyncio
async def test_diagnostics_are_fast_and_safe(mock_registry) -> None:
    diagnostics = await mock_registry.audio2face.diagnostics()

    assert diagnostics.adapter == "audio2face"
    assert diagnostics.provider == "mock"
    assert diagnostics.available is True
    assert diagnostics.latency_ms >= 0
    assert "mock" in diagnostics.message.lower()
