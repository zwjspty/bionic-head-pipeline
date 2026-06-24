from types import SimpleNamespace

import pytest

from bionic_head.api.dependencies import AppContainer
from bionic_head.domain.errors import ErrorCode, PipelineException
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


class _FakeAudio2Face:
    name = "emotalk_sidecar"

    def __init__(self) -> None:
        self.call_count = 0

    async def prewarm(self) -> DiagnosticResult:
        self.call_count += 1
        return DiagnosticResult(
            adapter="audio2face",
            provider="emotalk_sidecar",
            available=True,
            latency_ms=2.0,
            message="prewarmed",
        )


class _FailingAudio2Face:
    name = "emotalk_sidecar"

    async def prewarm(self) -> DiagnosticResult:
        raise PipelineException(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            stage="audio2face.prewarm",
            provider="emotalk_sidecar",
            retryable=True,
            message="prewarm failed",
        )


class _ClosableAudio2Face:
    name = "emotalk_sidecar"

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


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


@pytest.mark.asyncio
async def test_container_prewarms_enabled_emotalk_sidecar_provider(mock_settings) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.audio2face.provider = "emotalk_sidecar"
    settings.providers.emotalk_sidecar.prewarm_on_startup = True
    audio2face = _FakeAudio2Face()
    container = AppContainer(
        settings=settings,
        registry=SimpleNamespace(llm=SimpleNamespace(name="mock"), audio2face=audio2face),
        store=None,
        sessions=None,
    )

    results = await container.prewarm()

    assert audio2face.call_count == 1
    assert len(results) == 1
    assert results[0].adapter == "audio2face"
    assert results[0].provider == "emotalk_sidecar"


@pytest.mark.asyncio
async def test_container_skips_emotalk_sidecar_prewarm_when_disabled(mock_settings) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.audio2face.provider = "emotalk_sidecar"
    settings.providers.emotalk_sidecar.prewarm_on_startup = False
    audio2face = _FakeAudio2Face()
    container = AppContainer(
        settings=settings,
        registry=SimpleNamespace(llm=SimpleNamespace(name="mock"), audio2face=audio2face),
        store=None,
        sessions=None,
    )

    results = await container.prewarm()

    assert results == []
    assert audio2face.call_count == 0


@pytest.mark.asyncio
async def test_container_raises_when_required_emotalk_sidecar_prewarm_fails(mock_settings) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.audio2face.provider = "emotalk_sidecar"
    settings.providers.emotalk_sidecar.prewarm_on_startup = True
    settings.providers.emotalk_sidecar.prewarm_required = True
    container = AppContainer(
        settings=settings,
        registry=SimpleNamespace(llm=SimpleNamespace(name="mock"), audio2face=_FailingAudio2Face()),
        store=None,
        sessions=None,
    )

    with pytest.raises(PipelineException) as raised:
        await container.prewarm()

    assert raised.value.stage == "audio2face.prewarm"


@pytest.mark.asyncio
async def test_container_reports_optional_emotalk_sidecar_prewarm_failure(mock_settings) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.audio2face.provider = "emotalk_sidecar"
    settings.providers.emotalk_sidecar.prewarm_on_startup = True
    settings.providers.emotalk_sidecar.prewarm_required = False
    container = AppContainer(
        settings=settings,
        registry=SimpleNamespace(llm=SimpleNamespace(name="mock"), audio2face=_FailingAudio2Face()),
        store=None,
        sessions=None,
    )

    results = await container.prewarm()

    assert len(results) == 1
    assert results[0].available is False
    assert results[0].adapter == "audio2face"
    assert "prewarm failed" in results[0].message


@pytest.mark.asyncio
async def test_container_close_calls_available_provider_close_and_tolerates_missing_adapters(mock_settings) -> None:
    audio2face = _ClosableAudio2Face()
    container = AppContainer(
        settings=mock_settings,
        registry=SimpleNamespace(audio2face=audio2face),
        store=None,
        sessions=None,
    )

    await container.close()

    assert audio2face.closed is True
