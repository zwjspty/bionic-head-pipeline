import sys
from pathlib import Path

import pytest

from bionic_head.adapters.emotalk import EmoTalkAudio2FaceAdapter
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


def _adapter(script: Path, **overrides) -> EmoTalkAudio2FaceAdapter:
    values = {
        "executable": sys.executable,
        "args": [str(script), "{input_path}", "{output_dir}"],
        "output_npy_glob": "*.npy",
        "output_json_glob": "*.json",
        "timeout_seconds": 2,
        "grace_seconds": 0.1,
    }
    values.update(overrides)
    return EmoTalkAudio2FaceAdapter(**values)


@pytest.mark.asyncio
async def test_emotalk_loads_n_by_52_output(tmp_path, mock_audio, turn_context) -> None:
    script = _write_script(
        tmp_path / "fake_emotalk.py",
        """
import json
import pathlib
import sys

import numpy as np

output_dir = pathlib.Path(sys.argv[-1])
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / "emotalk.npy", np.ones((3, 52), dtype=np.float32) * 0.5)
(output_dir / "meta.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")
""",
    )

    face = await _adapter(script).drive(mock_audio, Emotion.HAPPY, 1.0, turn_context)

    assert face.frame_count == 3
    assert face.channel_count == 52
    assert face.fps == 30
    assert face.path is not None
    assert face.path.name == "emotalk.npy"
    assert face.path.parent.name == "emotalk_0001"
    assert face.frames[0][0] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_emotalk_errors_use_emotalk_provider_name(
    tmp_path,
    mock_audio,
    turn_context,
) -> None:
    script = _write_script(tmp_path / "failing_emotalk.py", "import sys; sys.exit(7)")

    with pytest.raises(PipelineException) as raised:
        await _adapter(script).drive(mock_audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.PROVIDER_FAILED
    assert raised.value.provider == "emotalk"
    assert "EmoTalk" in raised.value.safe_message


@pytest.mark.asyncio
async def test_emotalk_diagnostics_report_provider_name(tmp_path) -> None:
    script = _write_script(tmp_path / "noop.py", "")

    result = await _adapter(script).diagnostics()

    assert result.adapter == "audio2face"
    assert result.provider == "emotalk"
    assert result.available is True
    assert "EmoTalk" in result.message


def test_registry_builds_emotalk_audio2face_with_raw_ue5(mock_settings, tmp_path) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.adapters.audio2face.provider = "emotalk"
    settings.adapters.ue5.provider = "morpheus-raw"
    settings.providers.emotalk.executable = sys.executable
    settings.providers.emotalk.args = ["script.py", "{input_path}", "{output_dir}"]
    settings.providers.emotalk.cwd = tmp_path

    registry = build_registry(settings)

    assert registry.audio2face.name == "emotalk"
    assert registry.ue5.name == "morpheus-raw"
