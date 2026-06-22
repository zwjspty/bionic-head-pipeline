import asyncio
import sys
from pathlib import Path

import pytest

from bionic_head.adapters.piper import PiperTTSAdapter
from bionic_head.adapters.registry import build_registry
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import Emotion


def _write_script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def fake_piper_script(tmp_path: Path) -> Path:
    return _write_script(
        tmp_path / "fake_piper.py",
        """
import array
import pathlib
import sys
import wave

output = pathlib.Path(sys.argv[-1])
output.parent.mkdir(parents=True, exist_ok=True)
text = sys.stdin.read()
(output.with_suffix(output.suffix + ".stdin.txt")).write_text(text, encoding="utf-8")
samples = array.array("h", [1200, -1200] * 800)
with wave.open(str(output), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(samples.tobytes())
""",
    )


@pytest.mark.asyncio
async def test_piper_writes_and_validates_wav(fake_piper_script, turn_context) -> None:
    adapter = PiperTTSAdapter(
        executable=sys.executable,
        args=[str(fake_piper_script), "{output_path}"],
        model_path="/models/voice.onnx",
        timeout_seconds=2,
        grace_seconds=0.1,
    )

    result = await adapter.synthesize("你好", Emotion.FRIENDLY, 0.8, turn_context)

    assert result.path.exists()
    assert result.sample_rate == 16000
    assert result.channels == 1
    assert result.sample_width_bytes == 2
    assert result.duration_seconds > 0
    assert result.path.with_suffix(result.path.suffix + ".stdin.txt").read_text(
        encoding="utf-8"
    ) == "你好"


@pytest.mark.asyncio
async def test_piper_can_pass_text_as_argument(tmp_path, turn_context) -> None:
    script = _write_script(
        tmp_path / "fake_piper_arg.py",
        """
import array
import pathlib
import sys
import wave

text = sys.argv[1]
output = pathlib.Path(sys.argv[-1])
(output.with_suffix(output.suffix + ".arg.txt")).write_text(text, encoding="utf-8")
samples = array.array("h", [800, -800] * 800)
with wave.open(str(output), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(22050)
    wav.writeframes(samples.tobytes())
""",
    )
    adapter = PiperTTSAdapter(
        executable=sys.executable,
        args=[str(script), "{text}", "{output_path}"],
        model_path="/models/voice.onnx",
        timeout_seconds=2,
        grace_seconds=0.1,
    )

    result = await adapter.synthesize("参数文本", Emotion.CALM, 0.3, turn_context)

    assert result.sample_rate == 22050
    assert result.path.with_suffix(result.path.suffix + ".arg.txt").read_text(
        encoding="utf-8"
    ) == "参数文本"


def test_rejects_unknown_template_variable() -> None:
    with pytest.raises(PipelineException) as raised:
        PiperTTSAdapter(
            executable=sys.executable,
            args=["{bad}", "{output_path}"],
            model_path="/models/voice.onnx",
            timeout_seconds=2,
            grace_seconds=0.1,
        )

    assert raised.value.code is ErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_nonzero_exit_maps_to_provider_failed(tmp_path, turn_context) -> None:
    script = _write_script(
        tmp_path / "failing_piper.py",
        "import sys; print('/private/piper/error', file=sys.stderr); sys.exit(5)",
    )
    adapter = PiperTTSAdapter(
        executable=sys.executable,
        args=[str(script), "{output_path}"],
        model_path="/models/voice.onnx",
        timeout_seconds=2,
        grace_seconds=0.1,
    )

    with pytest.raises(PipelineException) as raised:
        await adapter.synthesize("你好", Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.PROVIDER_FAILED
    assert raised.value.stage == "tts"
    assert raised.value.provider == "piper"
    assert "/private/piper/error" not in raised.value.safe_message


@pytest.mark.asyncio
async def test_timeout_maps_to_provider_timeout(tmp_path, turn_context) -> None:
    script = _write_script(tmp_path / "slow_piper.py", "import time; time.sleep(5)")
    adapter = PiperTTSAdapter(
        executable=sys.executable,
        args=[str(script), "{output_path}"],
        model_path="/models/voice.onnx",
        timeout_seconds=0.05,
        grace_seconds=0.05,
    )

    with pytest.raises(PipelineException) as raised:
        await adapter.synthesize("你好", Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert raised.value.provider == "piper"


@pytest.mark.asyncio
async def test_cancel_propagates_to_process_runner(tmp_path, turn_context) -> None:
    script = _write_script(tmp_path / "slow_piper.py", "import time; time.sleep(5)")
    adapter = PiperTTSAdapter(
        executable=sys.executable,
        args=[str(script), "{output_path}"],
        model_path="/models/voice.onnx",
        timeout_seconds=5,
        grace_seconds=0.05,
    )

    task = asyncio.create_task(
        adapter.synthesize("你好", Emotion.FRIENDLY, 0.8, turn_context)
    )
    await asyncio.sleep(0.05)
    turn_context.cancellation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_invalid_wav_maps_to_output_validation_failed(tmp_path, turn_context) -> None:
    script = _write_script(
        tmp_path / "bad_wav.py",
        """
import pathlib
import sys

pathlib.Path(sys.argv[-1]).write_bytes(b"not a wav")
""",
    )
    adapter = PiperTTSAdapter(
        executable=sys.executable,
        args=[str(script), "{output_path}"],
        model_path="/models/voice.onnx",
        timeout_seconds=2,
        grace_seconds=0.1,
    )

    with pytest.raises(PipelineException) as raised:
        await adapter.synthesize("你好", Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED
    assert raised.value.stage == "tts"


@pytest.mark.asyncio
async def test_diagnostics_checks_executable_model_and_template(tmp_path) -> None:
    model_path = tmp_path / "zh_CN-huayan-medium.onnx"
    model_path.write_bytes(b"model")
    adapter = PiperTTSAdapter(
        executable=sys.executable,
        args=["--model", "{model_path}", "--output_file", "{output_path}"],
        model_path=model_path,
        timeout_seconds=2,
        grace_seconds=0.1,
    )

    result = await adapter.diagnostics()

    assert result.adapter == "tts"
    assert result.provider == "piper"
    assert result.available is True


@pytest.mark.asyncio
async def test_diagnostics_reports_missing_model(tmp_path) -> None:
    adapter = PiperTTSAdapter(
        executable=sys.executable,
        args=["--model", "{model_path}", "--output_file", "{output_path}"],
        model_path=tmp_path / "missing.onnx",
        timeout_seconds=2,
        grace_seconds=0.1,
    )

    result = await adapter.diagnostics()

    assert result.available is False
    assert "model" in result.message.lower()


def test_registry_builds_piper_tts_with_other_mock_providers(mock_settings, tmp_path) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.tts.provider = "piper"
    settings.providers.piper.executable = sys.executable
    settings.providers.piper.args = ["--model", "{model_path}", "--output_file", "{output_path}"]
    settings.providers.piper.model_path = tmp_path / "voice.onnx"

    registry = build_registry(settings)

    assert registry.tts.name == "piper"
    assert registry.asr.name == "mock"
