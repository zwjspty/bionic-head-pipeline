from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

from bionic_head.adapters.registry import build_registry
from bionic_head.config import AppSettings
from bionic_head.core.audio import audio_artifact_from_wav
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import AudioArtifact, Emotion


def _write_wav(
    path: Path,
    *,
    sample_rate: int,
    channels: int,
    samples: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return path


def _mono_wav(path: Path, *, sample_rate: int = 16000, seconds: float = 1.0) -> Path:
    sample_count = int(sample_rate * seconds)
    samples = np.linspace(-1200, 1200, sample_count, dtype=np.int16)
    return _write_wav(path, sample_rate=sample_rate, channels=1, samples=samples)


def _sidecar_script(path: Path, body: str) -> list[str]:
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


def _adapter(command: list[str]):
    from bionic_head.adapters.emotalk_sidecar import EmoTalkSidecarAudio2FaceAdapter

    return EmoTalkSidecarAudio2FaceAdapter(
        sidecar_command=command,
        sample_rate=16000,
        fps=30,
        timeout_seconds=1.0,
        channel_count=52,
    )


@pytest.mark.asyncio
async def test_provider_starts_fake_sidecar_and_returns_n_by_52(
    tmp_path: Path,
    turn_context,
) -> None:
    audio = audio_artifact_from_wav(_mono_wav(tmp_path / "speech.wav"))
    adapter = _adapter([sys.executable, "-m", "bionic_head.emotalk_fake_sidecar"])

    face = await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert face.frame_count == 30
    assert face.channel_count == 52
    assert face.fps == 30
    assert face.path is not None
    assert face.path.exists()
    assert np.load(face.path, allow_pickle=False).shape == (30, 52)

    await adapter.close()


@pytest.mark.asyncio
async def test_provider_preserves_session_turn_generation_epoch(
    tmp_path: Path,
    turn_context,
) -> None:
    audio = audio_artifact_from_wav(_mono_wav(tmp_path / "speech.wav"))
    adapter = _adapter([sys.executable, "-m", "bionic_head.emotalk_fake_sidecar"])
    epoch_context = replace(turn_context, generation_epoch=7)

    face = await adapter.drive(audio, Emotion.HAPPY, 0.5, epoch_context)

    meta_path = face.path.parent / "meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["request"]["session_id"] == str(epoch_context.session_id)
    assert payload["request"]["turn_id"] == str(epoch_context.turn_id)
    assert payload["request"]["generation_epoch"] == 7
    assert payload["response"]["session_id"] == str(epoch_context.session_id)
    assert payload["response"]["turn_id"] == str(epoch_context.turn_id)
    assert payload["response"]["generation_epoch"] == 7

    await adapter.close()


@pytest.mark.asyncio
async def test_provider_reuses_one_subprocess_for_three_calls(
    tmp_path: Path,
    turn_context,
) -> None:
    audio = audio_artifact_from_wav(_mono_wav(tmp_path / "speech.wav"))
    adapter = _adapter([sys.executable, "-m", "bionic_head.emotalk_fake_sidecar"])

    first = await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)
    first_pid = adapter.process_pid
    second = await adapter.drive(audio, Emotion.CALM, 0.4, turn_context)
    third = await adapter.drive(audio, Emotion.HAPPY, 0.9, turn_context)

    assert first.path.parent.name == "emotalk_sidecar_0001"
    assert second.path.parent.name == "emotalk_sidecar_0002"
    assert third.path.parent.name == "emotalk_sidecar_0003"
    assert adapter.process_pid == first_pid
    assert adapter.process_start_count == 1

    await adapter.close()


@pytest.mark.asyncio
async def test_provider_resamples_non_16k_wav_to_expected_frame_count(
    tmp_path: Path,
    turn_context,
) -> None:
    wav_path = _mono_wav(tmp_path / "speech-22k.wav", sample_rate=22050)
    audio = AudioArtifact(
        path=wav_path,
        sample_rate=22050,
        channels=1,
        sample_width_bytes=2,
        duration_seconds=1.0,
        byte_length=wav_path.stat().st_size,
    )
    adapter = _adapter([sys.executable, "-m", "bionic_head.emotalk_fake_sidecar"])

    face = await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert face.frame_count == 30

    await adapter.close()


@pytest.mark.asyncio
async def test_provider_maps_sidecar_error_response(
    tmp_path: Path,
    turn_context,
) -> None:
    audio = audio_artifact_from_wav(_mono_wav(tmp_path / "speech.wav"))
    adapter = _adapter(
        _sidecar_script(
            tmp_path / "error_sidecar.py",
            """
import sys

from bionic_head.sidecar_protocol import (
    SidecarResponse,
    decode_request,
    encode_message,
    encode_response,
    read_message,
)

header, body = read_message(sys.stdin.buffer)
request = decode_request(encode_message(header, body))
response = SidecarResponse.failure(
    "invalid_request",
    "boom",
    session_id=request.session_id,
    turn_id=request.turn_id,
    generation_epoch=request.generation_epoch,
)
sys.stdout.buffer.write(encode_response(response))
sys.stdout.buffer.flush()
""",
        )
    )

    with pytest.raises(PipelineException) as raised:
        await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.PROVIDER_FAILED
    assert raised.value.provider == "emotalk_sidecar"
    assert str(turn_context.session_id) in raised.value.safe_message
    assert str(turn_context.turn_id) in raised.value.safe_message
    assert "generation_epoch=0" in raised.value.safe_message
    assert "invalid_request" in raised.value.safe_message
    assert "pid=" in raised.value.safe_message

    await adapter.close()


@pytest.mark.asyncio
async def test_provider_maps_process_exit_to_provider_unavailable(
    tmp_path: Path,
    turn_context,
) -> None:
    audio = audio_artifact_from_wav(_mono_wav(tmp_path / "speech.wav"))
    adapter = _adapter(_sidecar_script(tmp_path / "exit_sidecar.py", "raise SystemExit(0)\n"))

    with pytest.raises(PipelineException) as raised:
        await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert raised.value.provider == "emotalk_sidecar"

    await adapter.close()


@pytest.mark.asyncio
async def test_provider_maps_missing_executable_to_provider_unavailable(
    tmp_path: Path,
    turn_context,
) -> None:
    audio = audio_artifact_from_wav(_mono_wav(tmp_path / "speech.wav"))
    adapter = _adapter(["/definitely/missing/emotalk-sidecar"])

    with pytest.raises(PipelineException) as raised:
        await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert raised.value.provider == "emotalk_sidecar"
    assert raised.value.stage == "audio2face"
    assert str(turn_context.session_id) in raised.value.safe_message
    assert str(turn_context.turn_id) in raised.value.safe_message
    assert f"generation_epoch={turn_context.generation_epoch}" in raised.value.safe_message
    assert "pid=None" in raised.value.safe_message


@pytest.mark.asyncio
async def test_provider_maps_failure_response_id_mismatch_to_output_validation_failed(
    tmp_path: Path,
    turn_context,
) -> None:
    audio = audio_artifact_from_wav(_mono_wav(tmp_path / "speech.wav"))
    adapter = _adapter(
        _sidecar_script(
            tmp_path / "mismatch_sidecar.py",
            """
import sys

from bionic_head.sidecar_protocol import (
    SidecarResponse,
    decode_request,
    encode_message,
    encode_response,
    read_message,
)

header, body = read_message(sys.stdin.buffer)
request = decode_request(encode_message(header, body))
response = SidecarResponse.failure(
    "invalid_request",
    "boom",
    session_id="wrong",
    turn_id="wrong",
    generation_epoch=999,
)
sys.stdout.buffer.write(encode_response(response))
sys.stdout.buffer.flush()
""",
        )
    )

    with pytest.raises(PipelineException) as raised:
        await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED
    assert raised.value.provider == "emotalk_sidecar"
    assert str(turn_context.session_id) in raised.value.safe_message
    assert str(turn_context.turn_id) in raised.value.safe_message
    assert "generation_epoch mismatch" in raised.value.safe_message
    assert "session_id mismatch" in raised.value.safe_message

    await adapter.close()


@pytest.mark.asyncio
async def test_provider_timeout_terminates_process(
    tmp_path: Path,
    turn_context,
) -> None:
    audio = audio_artifact_from_wav(_mono_wav(tmp_path / "speech.wav"))
    adapter = _adapter(
        _sidecar_script(
            tmp_path / "sleep_sidecar.py",
            """
import time

time.sleep(60)
""",
        )
    )
    adapter.timeout_seconds = 0.05

    with pytest.raises(PipelineException) as raised:
        await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert adapter.process is None or adapter.process.returncode is not None


@pytest.mark.asyncio
async def test_provider_rejects_truncated_stdout(
    tmp_path: Path,
    turn_context,
) -> None:
    audio = audio_artifact_from_wav(_mono_wav(tmp_path / "speech.wav"))
    adapter = _adapter(
        _sidecar_script(
            tmp_path / "truncated_sidecar.py",
            """
import struct
import sys

sys.stdout.buffer.write(struct.pack(">I", 32))
sys.stdout.buffer.write(b"{\\"protocol\\": \\"emotalk-sidecar-v1\\", \\"ok\\": true")
sys.stdout.buffer.flush()
""",
        )
    )

    with pytest.raises(PipelineException) as raised:
        await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert raised.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED

    await adapter.close()


@pytest.mark.asyncio
async def test_provider_close_stops_process(
    tmp_path: Path,
    turn_context,
) -> None:
    audio = audio_artifact_from_wav(_mono_wav(tmp_path / "speech.wav"))
    adapter = _adapter([sys.executable, "-m", "bionic_head.emotalk_fake_sidecar"])

    await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)
    process = adapter.process

    await adapter.close()

    assert process is not None
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_diagnostics_does_not_start_process() -> None:
    adapter = _adapter([sys.executable, "-m", "bionic_head.emotalk_fake_sidecar"])

    result = await adapter.diagnostics()

    assert result.available is True
    assert adapter.process is None
    assert adapter.process_start_count == 0


def test_registry_builds_sidecar_audio2face_without_starting_process(mock_settings) -> None:
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


def test_app_settings_accepts_emotalk_sidecar_provider_config() -> None:
    settings = AppSettings.model_validate(
        {
            "adapters": {
                "audio2face": {
                    "provider": "emotalk_sidecar",
                    "timeout_seconds": 10,
                }
            },
            "providers": {
                "emotalk_sidecar": {
                    "sidecar_command": ["python", "-m", "bionic_head.emotalk_fake_sidecar"],
                    "sample_rate": 16000,
                    "fps": 30,
                    "timeout_seconds": 10.0,
                    "channel_count": 52,
                }
            },
        }
    )

    assert settings.adapters.audio2face.provider == "emotalk_sidecar"
    assert settings.providers.emotalk_sidecar.sidecar_command == [
        "python",
        "-m",
        "bionic_head.emotalk_fake_sidecar",
    ]
