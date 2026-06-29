from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter


@dataclass
class SegmentPlaybackTiming:
    segment_id: str
    tts_received_ms: float | None = None
    ue5_first_frame_received_ms: float | None = None
    audio_play_start_ms: float | None = None
    face_first_frame_displayed_ms: float | None = None
    audio_face_offset_ms: float | None = None
    audio_wait_for_face_ms: float | None = None
    face_late_by_ms: float | None = None
    audio_wait_for_face_timeout: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "tts_received_ms": self.tts_received_ms,
            "ue5_first_frame_received_ms": self.ue5_first_frame_received_ms,
            "audio_play_start_ms": self.audio_play_start_ms,
            "face_first_frame_displayed_ms": self.face_first_frame_displayed_ms,
            "audio_face_offset_ms": self.audio_face_offset_ms,
            "audio_wait_for_face_ms": self.audio_wait_for_face_ms,
            "face_late_by_ms": self.face_late_by_ms,
            "audio_wait_for_face_timeout": self.audio_wait_for_face_timeout,
        }


class PlaybackClock:
    def __init__(self, *, clock: Callable[[], float] = perf_counter) -> None:
        self._clock = clock
        self._started_at = clock()
        self._segments: dict[str, SegmentPlaybackTiming] = {}
        self._playback_stop_received_ms: float | None = None
        self._audio_stopped_ms: float | None = None
        self._face_cleared_ms: float | None = None

    def mark_tts_received(self, segment_id: str) -> float:
        segment = self._segment(segment_id)
        if segment.tts_received_ms is None:
            segment.tts_received_ms = self._elapsed_ms()
        return segment.tts_received_ms

    def mark_ue5_received(self, segment_id: str) -> float:
        segment = self._segment(segment_id)
        if segment.ue5_first_frame_received_ms is None:
            segment.ue5_first_frame_received_ms = self._elapsed_ms()
        return segment.ue5_first_frame_received_ms

    def mark_audio_play_start(self, segment_id: str) -> float:
        segment = self._segment(segment_id)
        if segment.audio_play_start_ms is None:
            segment.audio_play_start_ms = self._elapsed_ms()
            self._update_segment_offsets(segment)
        return segment.audio_play_start_ms

    def mark_face_display(self, segment_id: str) -> float:
        segment = self._segment(segment_id)
        if segment.face_first_frame_displayed_ms is None:
            segment.face_first_frame_displayed_ms = self._elapsed_ms()
            self._update_segment_offsets(segment)
        return segment.face_first_frame_displayed_ms

    def mark_wait_for_face_timeout(self, segment_id: str) -> None:
        self._segment(segment_id).audio_wait_for_face_timeout = True

    def mark_playback_stop_received(self) -> float:
        if self._playback_stop_received_ms is None:
            self._playback_stop_received_ms = self._elapsed_ms()
        return self._playback_stop_received_ms

    def mark_audio_stopped(self) -> float:
        if self._audio_stopped_ms is None:
            self._audio_stopped_ms = self._elapsed_ms()
        return self._audio_stopped_ms

    def mark_face_cleared(self) -> float:
        if self._face_cleared_ms is None:
            self._face_cleared_ms = self._elapsed_ms()
        return self._face_cleared_ms

    def metrics(self) -> dict[str, object]:
        first_segment = self._first_segment_with_playback()
        playback_stop_to_audio = _delta(self._playback_stop_received_ms, self._audio_stopped_ms)
        playback_stop_to_face = _delta(self._playback_stop_received_ms, self._face_cleared_ms)
        return {
            "client_tts_received_ms": first_segment.tts_received_ms if first_segment else None,
            "client_ue5_first_frame_received_ms": (
                first_segment.ue5_first_frame_received_ms if first_segment else None
            ),
            "client_audio_play_start_ms": first_segment.audio_play_start_ms if first_segment else None,
            "client_face_first_frame_displayed_ms": (
                first_segment.face_first_frame_displayed_ms if first_segment else None
            ),
            "client_audio_face_offset_ms": first_segment.audio_face_offset_ms if first_segment else None,
            "client_audio_wait_for_face_ms": first_segment.audio_wait_for_face_ms if first_segment else None,
            "client_face_late_by_ms": first_segment.face_late_by_ms if first_segment else None,
            "client_audio_wait_for_face_timeout": any(
                segment.audio_wait_for_face_timeout for segment in self._segments.values()
            ),
            "client_playback_stop_received_ms": self._playback_stop_received_ms,
            "server_playback_stop_received_ms": self._playback_stop_received_ms,
            "client_audio_stopped_ms": self._audio_stopped_ms,
            "client_face_buffer_cleared_ms": self._face_cleared_ms,
            "client_playback_stop_to_audio_stop_ms": playback_stop_to_audio,
            "client_playback_stop_to_face_clear_ms": playback_stop_to_face,
        }

    def segment_metrics(self) -> dict[str, dict[str, object]]:
        return {
            segment_id: segment.to_dict()
            for segment_id, segment in sorted(self._segments.items(), key=lambda item: item[0])
        }

    def _segment(self, segment_id: str) -> SegmentPlaybackTiming:
        return self._segments.setdefault(segment_id, SegmentPlaybackTiming(segment_id=segment_id))

    def _elapsed_ms(self) -> float:
        return round((self._clock() - self._started_at) * 1000.0, 3)

    def _update_segment_offsets(self, segment: SegmentPlaybackTiming) -> None:
        if segment.audio_play_start_ms is not None and segment.tts_received_ms is not None:
            segment.audio_wait_for_face_ms = round(
                max(0.0, segment.audio_play_start_ms - segment.tts_received_ms),
                3,
            )
        if segment.audio_play_start_ms is not None and segment.face_first_frame_displayed_ms is not None:
            offset = round(segment.face_first_frame_displayed_ms - segment.audio_play_start_ms, 3)
            segment.audio_face_offset_ms = offset
            segment.face_late_by_ms = max(0.0, offset)

    def _first_segment_with_playback(self) -> SegmentPlaybackTiming | None:
        for segment in self._segments.values():
            if (
                segment.tts_received_ms is not None
                or segment.audio_play_start_ms is not None
                or segment.face_first_frame_displayed_ms is not None
            ):
                return segment
        return None


def _delta(start_ms: float | None, end_ms: float | None) -> float | None:
    if start_ms is None or end_ms is None:
        return None
    return round(end_ms - start_ms, 3)
