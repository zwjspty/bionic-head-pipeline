from types import SimpleNamespace

import pytest

from bionic_head.api.dependencies import AppContainer
from bionic_head.domain.models import DiagnosticResult


class _FakeLLM:
    name = "ollama"

    def __init__(self) -> None:
        self.call_count = 0

    async def prewarm(self) -> DiagnosticResult:
        self.call_count += 1
        return DiagnosticResult(
            adapter="llm",
            provider="ollama",
            available=True,
            latency_ms=1.0,
            message="prewarmed",
        )


@pytest.mark.asyncio
async def test_container_prewarms_enabled_ollama_provider(mock_settings) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.llm.provider = "ollama"
    settings.providers.ollama.prewarm = True
    llm = _FakeLLM()
    container = AppContainer(
        settings=settings,
        registry=SimpleNamespace(llm=llm),
        store=None,
        sessions=None,
    )

    results = await container.prewarm()

    assert llm.call_count == 1
    assert results[0].available is True
    assert results[0].provider == "ollama"


@pytest.mark.asyncio
async def test_container_skips_prewarm_when_disabled(mock_settings) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.llm.provider = "ollama"
    settings.providers.ollama.prewarm = False
    llm = _FakeLLM()
    container = AppContainer(
        settings=settings,
        registry=SimpleNamespace(llm=llm),
        store=None,
        sessions=None,
    )

    results = await container.prewarm()

    assert results == []
    assert llm.call_count == 0
