from collections.abc import Callable

import pytest

from scripts.local_demo_client import AudioPlaybackEngine, FacePlaybackEngine, MemoryAudioSink, PlaybackMetrics


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


def test_audio_engine_enqueues_wav_and_records_metrics(fake_clock: Callable[[], float]) -> None:
    metrics = PlaybackMetrics(clock=fake_clock)
    sink = MemoryAudioSink()
    audio = AudioPlaybackEngine(metrics, sink=sink)

    audio.enqueue_wav("chunk-1", b"RIFF....WAVE", generation_epoch=0)

    assert audio.queued_count == 1
    assert sink.played_chunks == [b"RIFF....WAVE"]
    assert metrics.to_dict()["client_audio_enqueued_count"] == 1
    assert metrics.to_dict()["client_audio_play_start_ms"] == 0.0


def test_stop_clears_audio_and_face_buffers(fake_clock: Callable[[], float]) -> None:
    metrics = PlaybackMetrics(clock=fake_clock)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)

    audio.enqueue_wav("chunk-1", b"RIFF....WAVE", generation_epoch=0)
    face.enqueue_frames("ue5-1", {"frames": [{"frame_index": 0}]}, generation_epoch=0)
    fake_clock.advance(0.050)

    metrics.mark_playback_stop_received()
    audio.stop()
    audio.clear()
    face.clear()

    summary = metrics.to_dict()
    assert audio.queued_count == 0
    assert face.buffered_chunk_count == 0
    assert summary["client_audio_stopped_ms"] == 50.0
    assert summary["client_face_buffer_cleared_ms"] == 50.0
