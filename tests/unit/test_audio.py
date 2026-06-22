from array import array
from pathlib import Path

import pytest

from bionic_head.core.audio import audio_artifact_from_wav, inspect_wav, pcm16le_to_wav, read_wav_pcm16
from bionic_head.domain.errors import ErrorCode, PipelineException


def test_pcm_round_trip_and_rms(tmp_path: Path) -> None:
    samples = array("h", [1000, -1000] * 1600)
    artifact = pcm16le_to_wav(samples.tobytes(), tmp_path / "input.wav")
    stats = inspect_wav(artifact.path)
    assert stats.sample_rate == 16000
    assert stats.channels == 1
    assert stats.sample_width_bytes == 2
    assert stats.rms > 0
    assert read_wav_pcm16(artifact.path) == samples.tobytes()


def test_odd_pcm_byte_length_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(PipelineException) as raised:
        pcm16le_to_wav(b"\x00", tmp_path / "bad.wav")
    assert raised.value.code is ErrorCode.INVALID_AUDIO_FORMAT


def test_malformed_wav_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not a wav")
    with pytest.raises(PipelineException) as raised:
        inspect_wav(path)
    assert raised.value.code is ErrorCode.INVALID_AUDIO_FORMAT


def test_audio_artifact_from_wav_reports_file_size(tmp_path: Path) -> None:
    samples = array("h", [500, -500] * 800)
    path = tmp_path / "speech.wav"
    pcm16le_to_wav(samples.tobytes(), path)
    artifact = audio_artifact_from_wav(path)
    assert artifact.byte_length == path.stat().st_size
    assert artifact.duration_seconds > 0
