import asyncio
import sys
from pathlib import Path

import pytest

from bionic_head.adapters.morpheus import MorpheusAudio2FaceAdapter
from bionic_head.adapters.morpheus_raw import MorpheusRawUE5Adapter
from bionic_head.adapters.registry import build_registry
from bionic_head.core.audio import audio_artifact_from_wav
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import Emotion


def _write_script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def mock_audio(speech_wav):
    return audio_artifact_from_wav(speech_wav)


@pytest.fixture
def fake_morpheus_script(tmp_path: Path) -> Path:
    return _write_script(
        tmp_path / "fake_morpheus.py",
        """
import json
import pathlib
import sys

import numpy as np

output_dir = pathlib.Path(sys.argv[-1])
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / "face.npy", np.ones((6, 52), dtype=np.float32) * 0.25)
(output_dir / "meta.json").write_text(json.dumps({"fps": 24}), encoding="utf-8")
""",
    )


def _adapter(script: Path, **overrides) -> MorpheusAudio2FaceAdapter:
    values = {
        "executable": sys.executable,
        "args": [str(script), "{input_path}", "{output_dir}"],
        "output_npy_glob": "*.npy",
        "output_json_glob": "*.json",
        "timeout_seconds": 2,
        "grace_seconds": 0.1,
    }
    values.update(overrides)
    return MorpheusAudio2FaceAdapter(**values)


@pytest.mark.asyncio
async def test_morpheus_loads_n_by_52_output(
    fake_morpheus_script,
    mock_audio,
    turn_context,
) -> None:
    adapter = _adapter(fake_morpheus_script)

    face = await adapter.drive(mock_audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert face.frame_count == 6
    assert face.channel_count == 52
    assert face.fps == 24
    assert face.frames[0][0] == pytest.approx(0.25)
    assert face.path is not None
    assert face.path.name == "face.npy"
    assert [path.name for path in face.auxiliary_paths] == ["meta.json"]


@pytest.mark.asyncio
async def test_raw_ue5_formatter_keeps_morpheus_contract(
    fake_morpheus_script,
    mock_audio,
    turn_context,
) -> None:
    face = await _adapter(fake_morpheus_script).drive(
        mock_audio,
        Emotion.FRIENDLY,
        0.8,
        turn_context,
    )
    payload = await MorpheusRawUE5Adapter().format(face, turn_context)

    assert payload.protocol == "bionic-head-ue5-v1"
    assert payload.format == "morpheus_52_raw"
    assert payload.channel_count == 52
    assert payload.frame_count == face.frame_count


@pytest.mark.asyncio
async def test_missing_output_maps_to_output_validation_failed(
    tmp_path,
    mock_audio,
    turn_context,
) -> None:
    script = _write_script(
        tmp_path / "no_output.py",
        "import pathlib, sys; pathlib.Path(sys.argv[-1]).mkdir(parents=True, exist_ok=True)",
    )

    with pytest.raises(PipelineException) as raised:
        await _adapter(script).drive(mock_audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED
    assert raised.value.stage == "audio2face"


@pytest.mark.asyncio
async def test_multiple_outputs_are_rejected(tmp_path, mock_audio, turn_context) -> None:
    script = _write_script(
        tmp_path / "multiple.py",
        """
import pathlib
import sys

import numpy as np

output_dir = pathlib.Path(sys.argv[-1])
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / "a.npy", np.zeros((2, 52), dtype=np.float32))
np.save(output_dir / "b.npy", np.zeros((2, 52), dtype=np.float32))
""",
    )

    with pytest.raises(PipelineException) as raised:
        await _adapter(script).drive(mock_audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("shape", "value"),
    [("(6, 51)", "0.0"), ("(6, 52)", "float('nan')")],
)
async def test_invalid_array_shape_or_nan_is_rejected(
    tmp_path,
    mock_audio,
    turn_context,
    shape,
    value,
) -> None:
    script = _write_script(
        tmp_path / "bad_array.py",
        f"""
import pathlib
import sys

import numpy as np

output_dir = pathlib.Path(sys.argv[-1])
output_dir.mkdir(parents=True, exist_ok=True)
array = np.zeros({shape}, dtype=np.float32)
array[0, 0] = {value}
np.save(output_dir / "face.npy", array)
""",
    )

    with pytest.raises(PipelineException) as raised:
        await _adapter(script).drive(mock_audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED


@pytest.mark.asyncio
async def test_timeout_maps_to_provider_timeout(tmp_path, mock_audio, turn_context) -> None:
    script = _write_script(tmp_path / "slow_morpheus.py", "import time; time.sleep(5)")

    with pytest.raises(PipelineException) as raised:
        await _adapter(script, timeout_seconds=0.05, grace_seconds=0.05).drive(
            mock_audio,
            Emotion.FRIENDLY,
            0.8,
            turn_context,
        )

    assert raised.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert raised.value.provider == "morpheus"


@pytest.mark.asyncio
async def test_cancel_propagates_to_process_runner(tmp_path, mock_audio, turn_context) -> None:
    script = _write_script(tmp_path / "slow_morpheus.py", "import time; time.sleep(5)")
    adapter = _adapter(script, timeout_seconds=5, grace_seconds=0.05)

    task = asyncio.create_task(adapter.drive(mock_audio, Emotion.FRIENDLY, 0.8, turn_context))
    await asyncio.sleep(0.05)
    turn_context.cancellation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_morpheus_calls_are_serialized(tmp_path, mock_audio, turn_context) -> None:
    script = _write_script(
        tmp_path / "serialized.py",
        """
import pathlib
import sys
import time

import numpy as np

output_dir = pathlib.Path(sys.argv[-1])
output_dir.mkdir(parents=True, exist_ok=True)
lock = output_dir.parent / "morpheus.lock"
if lock.exists():
    print("overlap", file=sys.stderr)
    sys.exit(9)
lock.write_text("locked", encoding="utf-8")
time.sleep(0.1)
np.save(output_dir / "face.npy", np.zeros((2, 52), dtype=np.float32))
lock.unlink()
""",
    )
    adapter = _adapter(script)

    face_a, face_b = await asyncio.gather(
        adapter.drive(mock_audio, Emotion.FRIENDLY, 0.8, turn_context),
        adapter.drive(mock_audio, Emotion.CALM, 0.4, turn_context),
    )

    assert face_a.frame_count == 2
    assert face_b.frame_count == 2


def test_rejects_unknown_template_variable() -> None:
    with pytest.raises(PipelineException) as raised:
        MorpheusAudio2FaceAdapter(
            executable=sys.executable,
            args=["{bad}", "{input_path}", "{output_dir}"],
            output_npy_glob="*.npy",
            output_json_glob="*.json",
            timeout_seconds=2,
            grace_seconds=0.1,
        )

    assert raised.value.code is ErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_diagnostics_reports_empty_command_argument() -> None:
    adapter = MorpheusAudio2FaceAdapter(
        executable=sys.executable,
        args=["", "{input_path}", "{output_dir}"],
        output_npy_glob="*.npy",
        output_json_glob="*.json",
        timeout_seconds=2,
        grace_seconds=0.1,
    )

    result = await adapter.diagnostics()

    assert result.adapter == "audio2face"
    assert result.provider == "morpheus"
    assert result.available is False
    assert "empty" in result.message.lower()


@pytest.mark.asyncio
async def test_diagnostics_accepts_executable_and_templates(fake_morpheus_script) -> None:
    result = await _adapter(fake_morpheus_script).diagnostics()

    assert result.available is True


def test_registry_builds_morpheus_and_raw_ue5_with_other_mock_providers(
    mock_settings,
    tmp_path,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.audio2face.provider = "morpheus"
    settings.adapters.ue5.provider = "morpheus-raw"
    settings.providers.morpheus.executable = sys.executable
    settings.providers.morpheus.args = ["script.py", "{input_path}", "{output_dir}"]
    settings.providers.morpheus.cwd = tmp_path

    registry = build_registry(settings)

    assert registry.audio2face.name == "morpheus"
    assert registry.ue5.name == "morpheus-raw"
    assert registry.asr.name == "mock"
