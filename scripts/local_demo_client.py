from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class AudioSink(Protocol):
    def play(self, wav_bytes: bytes) -> None:
        ...

    def stop(self) -> None:
        ...


class PlaybackMetrics:
    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._started_at = clock()
        self.client_audio_enqueued_count = 0
        self.client_audio_play_start_ms: float | None = None
        self.client_audio_stopped_ms: float | None = None
        self.client_face_buffer_cleared_ms: float | None = None
        self.client_playback_stop_received_ms: float | None = None

    def mark_audio_enqueued(self, generation_epoch: int | None = None) -> None:
        if self.client_audio_play_start_ms is None:
            self.client_audio_play_start_ms = self._elapsed_ms()
        self.client_audio_enqueued_count += 1

    def mark_audio_stopped(self) -> None:
        if self.client_audio_stopped_ms is None:
            self.client_audio_stopped_ms = self._elapsed_ms()

    def mark_playback_stop_received(self) -> None:
        if self.client_playback_stop_received_ms is None:
            self.client_playback_stop_received_ms = self._elapsed_ms()

    def mark_face_buffer_cleared(self) -> None:
        if self.client_face_buffer_cleared_ms is None:
            self.client_face_buffer_cleared_ms = self._elapsed_ms()

    def to_dict(self) -> dict[str, object]:
        return {
            "client_audio_enqueued_count": self.client_audio_enqueued_count,
            "client_audio_play_start_ms": self.client_audio_play_start_ms,
            "client_audio_stopped_ms": self.client_audio_stopped_ms,
            "client_playback_stop_received_ms": self.client_playback_stop_received_ms,
            "client_face_buffer_cleared_ms": self.client_face_buffer_cleared_ms,
        }

    def _elapsed_ms(self) -> float:
        return round((self._clock() - self._started_at) * 1000.0, 3)


class MemoryAudioSink:
    def __init__(self) -> None:
        self.played_chunks: list[bytes] = []
        self.stopped_count = 0

    def play(self, wav_bytes: bytes) -> None:
        self.played_chunks.append(wav_bytes)

    def stop(self) -> None:
        self.stopped_count += 1


class AudioPlaybackEngine:
    def __init__(self, metrics: PlaybackMetrics, sink: AudioSink | None = None) -> None:
        self._metrics = metrics
        self._sink = sink or MemoryAudioSink()
        self._queued_chunks: dict[str, bytes] = {}

    @property
    def queued_count(self) -> int:
        return len(self._queued_chunks)

    def enqueue_wav(self, chunk_id: str, wav_bytes: bytes, generation_epoch: int | None) -> None:
        self._queued_chunks[chunk_id] = wav_bytes
        self._metrics.mark_audio_enqueued(generation_epoch)
        self._sink.play(wav_bytes)

    def stop(self) -> None:
        self._sink.stop()
        self._metrics.mark_audio_stopped()

    def clear(self) -> None:
        self._queued_chunks.clear()


class FacePlaybackEngine:
    def __init__(self, metrics: PlaybackMetrics) -> None:
        self._metrics = metrics
        self._queued_chunks: dict[str, dict[str, object]] = {}

    @property
    def buffered_chunk_count(self) -> int:
        return len(self._queued_chunks)

    def enqueue_frames(
        self,
        chunk_id: str,
        payload: dict[str, object],
        generation_epoch: int | None,
    ) -> None:
        self._queued_chunks[chunk_id] = payload

    def clear(self) -> None:
        self._queued_chunks.clear()
        self._metrics.mark_face_buffer_cleared()
