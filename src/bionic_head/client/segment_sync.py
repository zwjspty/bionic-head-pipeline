from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from bionic_head.client.playback_clock import PlaybackClock


PlaybackSyncStrategy = Literal["immediate_audio", "wait_for_face"]


@dataclass(frozen=True)
class PlaybackAction:
    kind: Literal["play_audio", "display_face"]
    segment_id: str
    chunk_id: str
    generation_epoch: int | None
    wav_bytes: bytes | None = None
    face_payload: dict[str, object] | None = None


@dataclass
class _PendingSegment:
    segment_id: str
    generation_epoch: int | None
    tts_chunk_id: str | None = None
    wav_bytes: bytes | None = None
    tts_received_ms: float | None = None
    face_chunks: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    audio_released: bool = False
    face_released_count: int = 0


class SegmentSyncCoordinator:
    def __init__(
        self,
        *,
        strategy: PlaybackSyncStrategy,
        clock: PlaybackClock,
        wait_for_face_timeout_ms: int = 800,
    ) -> None:
        if strategy not in {"immediate_audio", "wait_for_face"}:
            raise ValueError("playback sync strategy must be immediate_audio or wait_for_face")
        if wait_for_face_timeout_ms < 0:
            raise ValueError("wait_for_face_timeout_ms must be non-negative")
        self.strategy = strategy
        self.clock = clock
        self.wait_for_face_timeout_ms = wait_for_face_timeout_ms
        self._latest_generation_epoch: int | None = None
        self._segments: dict[str, _PendingSegment] = {}
        self.stale_audio_drop_count = 0
        self.stale_face_drop_count = 0

    @property
    def pending_count(self) -> int:
        return len(self._segments)

    def update_latest_generation(self, generation_epoch: int | None) -> None:
        if generation_epoch is None:
            return
        if self._latest_generation_epoch is None or generation_epoch > self._latest_generation_epoch:
            self._latest_generation_epoch = generation_epoch

    def accept_tts(
        self,
        *,
        segment_id: str,
        chunk_id: str,
        wav_bytes: bytes,
        generation_epoch: int | None,
    ) -> list[PlaybackAction]:
        if self._is_stale(generation_epoch):
            self.stale_audio_drop_count += 1
            return []
        self.update_latest_generation(generation_epoch)
        self.clock.mark_tts_received(segment_id)
        segment = self._segment(segment_id, generation_epoch)
        segment.tts_chunk_id = chunk_id
        segment.wav_bytes = wav_bytes
        segment.tts_received_ms = self.clock.segment_metrics()[segment_id]["tts_received_ms"]  # type: ignore[assignment]
        return self._release_ready(segment)

    def accept_face(
        self,
        *,
        segment_id: str,
        chunk_id: str,
        payload: dict[str, object],
        generation_epoch: int | None,
    ) -> list[PlaybackAction]:
        if self._is_stale(generation_epoch):
            self.stale_face_drop_count += 1
            return []
        self.update_latest_generation(generation_epoch)
        self.clock.mark_ue5_received(segment_id)
        segment = self._segment(segment_id, generation_epoch)
        segment.face_chunks.append((chunk_id, payload))
        return self._release_ready(segment)

    def flush_timeouts(self, *, now_ms: float | None = None) -> list[PlaybackAction]:
        actions: list[PlaybackAction] = []
        for segment in list(self._segments.values()):
            if (
                self.strategy == "wait_for_face"
                and not segment.audio_released
                and segment.wav_bytes is not None
                and segment.tts_chunk_id is not None
                and segment.tts_received_ms is not None
            ):
                current_ms = now_ms if now_ms is not None else _first_numeric(
                    self.clock.metrics().get("client_tts_received_ms"),
                    0.0,
                )
                if current_ms - float(segment.tts_received_ms) >= self.wait_for_face_timeout_ms:
                    self.clock.mark_wait_for_face_timeout(segment.segment_id)
                    segment.audio_released = True
                    actions.append(
                        PlaybackAction(
                            kind="play_audio",
                            segment_id=segment.segment_id,
                            chunk_id=segment.tts_chunk_id,
                            generation_epoch=segment.generation_epoch,
                            wav_bytes=segment.wav_bytes,
                        )
                    )
        self._drop_completed_segments()
        return actions

    def clear(self, *, reason: str) -> None:
        self._segments.clear()

    def _segment(self, segment_id: str, generation_epoch: int | None) -> _PendingSegment:
        return self._segments.setdefault(
            segment_id,
            _PendingSegment(segment_id=segment_id, generation_epoch=generation_epoch),
        )

    def _release_ready(self, segment: _PendingSegment) -> list[PlaybackAction]:
        actions: list[PlaybackAction] = []
        if not segment.audio_released and segment.wav_bytes is not None and segment.tts_chunk_id is not None:
            if self.strategy == "immediate_audio" or segment.face_chunks:
                segment.audio_released = True
                actions.append(
                    PlaybackAction(
                        kind="play_audio",
                        segment_id=segment.segment_id,
                        chunk_id=segment.tts_chunk_id,
                        generation_epoch=segment.generation_epoch,
                        wav_bytes=segment.wav_bytes,
                    )
                )
        while segment.face_released_count < len(segment.face_chunks):
            if self.strategy == "wait_for_face" and not segment.audio_released:
                break
            chunk_id, payload = segment.face_chunks[segment.face_released_count]
            segment.face_released_count += 1
            actions.append(
                PlaybackAction(
                    kind="display_face",
                    segment_id=segment.segment_id,
                    chunk_id=chunk_id,
                    generation_epoch=segment.generation_epoch,
                    face_payload=payload,
                )
            )
        self._drop_completed_segments()
        return actions

    def _drop_completed_segments(self) -> None:
        for segment_id, segment in list(self._segments.items()):
            if (
                segment.audio_released
                and segment.wav_bytes is not None
                and segment.face_released_count >= len(segment.face_chunks)
            ):
                del self._segments[segment_id]

    def _is_stale(self, generation_epoch: int | None) -> bool:
        return (
            generation_epoch is not None
            and self._latest_generation_epoch is not None
            and generation_epoch < self._latest_generation_epoch
        )


def _first_numeric(value: object, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default
