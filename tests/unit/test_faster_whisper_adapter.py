import asyncio
from types import SimpleNamespace

import pytest

from bionic_head.adapters.faster_whisper import FasterWhisperASRAdapter
from bionic_head.adapters.registry import build_registry
from bionic_head.config import FasterWhisperSettings
from bionic_head.domain.errors import ErrorCode, PipelineException


def _settings() -> FasterWhisperSettings:
    return FasterWhisperSettings(
        model="base",
        device="cpu",
        compute_type="int8",
        language="zh",
    )


class FakeModel:
    def __init__(self, text: list[str] | None = None) -> None:
        self.text = text or [" 你好 ", " 世界 "]
        self.calls = 0

    def transcribe(self, path, language, vad_filter):
        self.calls += 1
        assert language == "zh"
        assert vad_filter is True
        segments = [SimpleNamespace(text=value) for value in self.text]
        return segments, SimpleNamespace(language="zh")


@pytest.mark.asyncio
async def test_transcribe_normalizes_segments(turn_context, speech_wav) -> None:
    adapter = FasterWhisperASRAdapter(
        settings=_settings(),
        model_factory=lambda **_: FakeModel(),
    )

    result = await adapter.transcribe(speech_wav, turn_context)

    assert result.text == "你好 世界"
    assert result.language == "zh"
    assert result.audio.sample_rate == 16000


@pytest.mark.asyncio
async def test_model_loads_lazily_and_is_reused(turn_context, speech_wav) -> None:
    model = FakeModel()
    factory_calls: list[dict[str, object]] = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return model

    adapter = FasterWhisperASRAdapter(settings=_settings(), model_factory=factory)

    assert factory_calls == []
    await adapter.transcribe(speech_wav, turn_context)
    await adapter.transcribe(speech_wav, turn_context)

    assert factory_calls == [
        {"model_size_or_path": "base", "device": "cpu", "compute_type": "int8"}
    ]
    assert model.calls == 2


@pytest.mark.asyncio
async def test_empty_transcript_maps_to_no_speech(turn_context, speech_wav) -> None:
    adapter = FasterWhisperASRAdapter(
        settings=_settings(),
        model_factory=lambda **_: FakeModel(text=[" ", ""]),
    )

    with pytest.raises(PipelineException) as raised:
        await adapter.transcribe(speech_wav, turn_context)

    assert raised.value.code is ErrorCode.NO_SPEECH_DETECTED
    assert raised.value.stage == "asr"


@pytest.mark.asyncio
async def test_worker_failure_maps_to_provider_failed_with_safe_message(
    turn_context,
    speech_wav,
) -> None:
    class FailingModel:
        def transcribe(self, path, language, vad_filter):
            raise RuntimeError("/private/model/cache failed")

    adapter = FasterWhisperASRAdapter(
        settings=_settings(),
        model_factory=lambda **_: FailingModel(),
    )

    with pytest.raises(PipelineException) as raised:
        await adapter.transcribe(speech_wav, turn_context)

    assert raised.value.code is ErrorCode.PROVIDER_FAILED
    assert raised.value.provider == "faster-whisper"
    assert "/private/model/cache" not in raised.value.safe_message


@pytest.mark.asyncio
async def test_pre_cancelled_turn_does_not_load_model(turn_context, speech_wav) -> None:
    factory_called = False

    def factory(**kwargs):
        nonlocal factory_called
        factory_called = True
        return FakeModel()

    turn_context.cancellation.cancel()
    adapter = FasterWhisperASRAdapter(settings=_settings(), model_factory=factory)

    with pytest.raises(asyncio.CancelledError):
        await adapter.transcribe(speech_wav, turn_context)

    assert factory_called is False


@pytest.mark.asyncio
async def test_late_cancel_discards_worker_result(turn_context, speech_wav) -> None:
    class CancellingModel(FakeModel):
        def transcribe(self, path, language, vad_filter):
            turn_context.cancellation.cancel()
            return super().transcribe(path, language, vad_filter)

    adapter = FasterWhisperASRAdapter(
        settings=_settings(),
        model_factory=lambda **_: CancellingModel(),
    )

    with pytest.raises(asyncio.CancelledError):
        await adapter.transcribe(speech_wav, turn_context)


@pytest.mark.asyncio
async def test_diagnostics_probe_dependency_without_loading_model(monkeypatch) -> None:
    factory_called = False

    def factory(**kwargs):
        nonlocal factory_called
        factory_called = True
        return FakeModel()

    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name == "faster_whisper" else None,
    )
    adapter = FasterWhisperASRAdapter(settings=_settings(), model_factory=factory)

    result = await adapter.diagnostics()

    assert result.adapter == "asr"
    assert result.provider == "faster-whisper"
    assert result.available is True
    assert "base" in result.message
    assert factory_called is False


@pytest.mark.asyncio
async def test_diagnostics_reports_missing_dependency(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    adapter = FasterWhisperASRAdapter(settings=_settings(), model_factory=lambda **_: FakeModel())

    result = await adapter.diagnostics()

    assert result.available is False
    assert "faster-whisper" in result.message


def test_registry_builds_faster_whisper_asr_with_other_mock_providers(mock_settings) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.asr.provider = "faster-whisper"

    registry = build_registry(settings)

    assert registry.asr.name == "faster-whisper"
    assert registry.llm.name == "mock"
