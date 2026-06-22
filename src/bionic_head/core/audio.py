from __future__ import annotations

from array import array
from pathlib import Path
import wave

from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import AudioArtifact, AudioFormat, AudioStats


def _audio_error(message: str) -> PipelineException:
    return PipelineException(
        code=ErrorCode.INVALID_AUDIO_FORMAT,
        stage="audio",
        provider=None,
        retryable=False,
        message=message,
    )


def _read_wav(path: Path, expected: AudioFormat) -> tuple[AudioStats, bytes]:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()
            frames = wav.readframes(frame_count)
    except (wave.Error, OSError, EOFError) as exc:
        raise _audio_error("Invalid WAV audio") from exc

    if channels != expected.channels:
        raise _audio_error("WAV must be mono")
    if sample_width != expected.sample_width_bytes:
        raise _audio_error("WAV must be signed 16-bit PCM")
    if sample_rate != expected.sample_rate:
        raise _audio_error("WAV must be 16000 Hz")
    if frame_count <= 0 or not frames:
        raise _audio_error("WAV must contain audio frames")
    if len(frames) % expected.sample_width_bytes != 0:
        raise _audio_error("WAV frame data is not aligned to sample width")

    samples = array("h")
    samples.frombytes(frames)
    if not samples:
        raise _audio_error("WAV must contain samples")

    mean_square = sum(sample * sample for sample in samples) / len(samples)
    rms = (mean_square**0.5) / 32768.0
    peak = max(abs(sample) for sample in samples) / 32768.0
    duration = frame_count / float(sample_rate)
    stats = AudioStats(
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        frame_count=frame_count,
        duration_seconds=duration,
        rms=rms,
        peak=peak,
    )
    return stats, frames


def inspect_wav(path: Path, expected: AudioFormat | None = None) -> AudioStats:
    expected = expected or AudioFormat()
    stats, _ = _read_wav(path, expected)
    return stats


def audio_artifact_from_wav(path: Path) -> AudioArtifact:
    stats = inspect_wav(path)
    return AudioArtifact(
        path=path,
        sample_rate=stats.sample_rate,
        channels=stats.channels,
        sample_width_bytes=stats.sample_width_bytes,
        duration_seconds=stats.duration_seconds,
        byte_length=path.stat().st_size,
    )


def pcm16le_to_wav(pcm: bytes, path: Path, sample_rate: int = 16000) -> AudioArtifact:
    if not pcm:
        raise _audio_error("PCM audio must not be empty")
    if len(pcm) % 2 != 0:
        raise _audio_error("PCM audio must contain whole 16-bit samples")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return audio_artifact_from_wav(path)


def read_wav_pcm16(path: Path) -> bytes:
    _, frames = _read_wav(path, AudioFormat())
    return frames
