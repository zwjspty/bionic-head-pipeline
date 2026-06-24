from __future__ import annotations

import copy
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class EyeContinuityKey:
    session_id: str
    turn_id: str
    generation_epoch: int


@dataclass(frozen=True)
class EyeContinuityMetrics:
    enabled: bool
    applied: bool
    reset: bool
    smooth_channel_count: int
    blink_channel_count: int
    overlap_frames: int
    actual_overlap_frames: int
    boundary_delta_before: float | None = None
    boundary_delta_after: float | None = None
    blink_enabled: bool = False
    blink_applied_count: int = 0
    blink_frame_count: int = 0
    blink_reset_count: int = 0
    global_frame_start: int = 0
    global_frame_end: int = 0

    def to_timing_payload(self) -> dict[str, bool | float]:
        payload: dict[str, bool | float] = {
            "eye_continuity_enabled": self.enabled,
            "eye_continuity_applied": self.applied,
            "eye_continuity_reset": self.reset,
            "eye_smooth_channel_count": float(self.smooth_channel_count),
            "blink_channel_count": float(self.blink_channel_count),
            "eye_continuity_overlap_frames": float(self.overlap_frames),
            "eye_continuity_actual_overlap_frames": float(self.actual_overlap_frames),
            "blink_enabled": self.blink_enabled,
            "blink_applied_count": float(self.blink_applied_count),
            "blink_frame_count": float(self.blink_frame_count),
            "blink_reset_count": float(self.blink_reset_count),
            "eye_global_frame_start": float(self.global_frame_start),
            "eye_global_frame_end": float(self.global_frame_end),
        }
        if self.boundary_delta_before is not None:
            payload["eye_boundary_delta_before"] = self.boundary_delta_before
        if self.boundary_delta_after is not None:
            payload["eye_boundary_delta_after"] = self.boundary_delta_after
        return payload


class EyeContinuityProcessor:
    def __init__(
        self,
        *,
        enabled: bool = True,
        eye_smooth_channel_indices: list[int] | None = None,
        blink_enabled: bool = False,
        blink_channel_indices: list[int] | None = None,
        overlap_frames: int = 6,
        blink_interval_min_sec: float = 2.5,
        blink_interval_max_sec: float = 6.0,
        blink_duration_frames: int = 5,
        blink_strength: float = 1.0,
        seed: int = 42,
        reset_blink_on_new_turn: bool = False,
    ) -> None:
        if overlap_frames < 0:
            raise ValueError("overlap_frames must be non-negative")
        if blink_interval_min_sec < 0:
            raise ValueError("blink_interval_min_sec must be non-negative")
        if blink_interval_max_sec < blink_interval_min_sec:
            raise ValueError("blink_interval_max_sec must be greater than or equal to blink_interval_min_sec")
        if blink_duration_frames < 1:
            raise ValueError("blink_duration_frames must be at least 1")
        if blink_strength < 0.0 or blink_strength > 1.0:
            raise ValueError("blink_strength must be between 0.0 and 1.0")

        self.enabled = enabled
        self.eye_smooth_channel_indices = list(eye_smooth_channel_indices or [])
        self.blink_enabled = blink_enabled
        self.blink_channel_indices = list(blink_channel_indices or [])
        self.overlap_frames = overlap_frames
        self.blink_interval_min_sec = blink_interval_min_sec
        self.blink_interval_max_sec = blink_interval_max_sec
        self.blink_duration_frames = blink_duration_frames
        self.blink_strength = blink_strength
        self.seed = seed
        self.reset_blink_on_new_turn = reset_blink_on_new_turn

        self._random = random.Random(seed)
        self._previous_key: EyeContinuityKey | None = None
        self._previous_segment_index: int | None = None
        self._previous_tail: list[list[float]] = []
        self._blink_session_id: str | None = None
        self._blink_turn_id: str | None = None
        self._global_frame_index = 0
        self._next_blink_frame = 0

    def reset(self) -> None:
        self._previous_key = None
        self._previous_segment_index = None
        self._previous_tail = []
        self._blink_session_id = None
        self._blink_turn_id = None
        self._global_frame_index = 0
        self._next_blink_frame = 0
        self._random = random.Random(self.seed)

    def process(
        self,
        frames: list[list[float]],
        *,
        session_id: str,
        turn_id: str,
        generation_epoch: int,
        segment_index: int,
        fps: int,
    ) -> tuple[list[list[float]], EyeContinuityMetrics]:
        copied = copy.deepcopy(frames)
        channel_count = _validate_frames(copied)
        if fps <= 0:
            raise ValueError("fps must be positive")

        smooth_indices = _validate_channel_indices(self.eye_smooth_channel_indices, channel_count)
        blink_indices = _validate_channel_indices(self.blink_channel_indices, channel_count)

        blink_reset_count = self._ensure_blink_state(session_id=session_id, turn_id=turn_id, fps=fps)
        global_frame_start = self._global_frame_index
        global_frame_end = global_frame_start + len(copied)

        if not self.enabled:
            self._global_frame_index = global_frame_end
            return copied, EyeContinuityMetrics(
                enabled=False,
                applied=False,
                reset=False,
                smooth_channel_count=len(smooth_indices),
                blink_channel_count=len(blink_indices),
                overlap_frames=self.overlap_frames,
                actual_overlap_frames=0,
                blink_enabled=False,
                blink_reset_count=blink_reset_count,
                global_frame_start=global_frame_start,
                global_frame_end=global_frame_end,
            )

        key = EyeContinuityKey(
            session_id=session_id,
            turn_id=turn_id,
            generation_epoch=generation_epoch,
        )
        reset = self._should_reset_smoothing(key, segment_index)
        smoothing_applied = False
        actual_overlap = 0
        boundary_delta_before: float | None = None
        boundary_delta_after: float | None = None

        if smooth_indices and not reset:
            actual_overlap = min(self.overlap_frames, len(self._previous_tail), len(copied))
            if actual_overlap > 0:
                previous_overlap = self._previous_tail[-actual_overlap:]
                boundary_delta_before = _mean_abs_delta_for_indices(
                    previous_overlap[-1],
                    copied[0],
                    smooth_indices,
                )
                for index in range(actual_overlap):
                    alpha = (index + 1) / float(actual_overlap)
                    for channel_index in smooth_indices:
                        previous = float(previous_overlap[index][channel_index])
                        current = float(copied[index][channel_index])
                        copied[index][channel_index] = previous * (1.0 - alpha) + current * alpha
                boundary_delta_after = _mean_abs_delta_for_indices(
                    previous_overlap[-1],
                    copied[0],
                    smooth_indices,
                )
                smoothing_applied = True

        blink_applied_count, blink_frame_count = self._apply_blink(
            copied,
            blink_indices=blink_indices,
            global_frame_start=global_frame_start,
            fps=fps,
        )
        self._global_frame_index = global_frame_end
        self._store_tail(key, segment_index, copied)

        return copied, EyeContinuityMetrics(
            enabled=True,
            applied=smoothing_applied or blink_frame_count > 0,
            reset=reset,
            smooth_channel_count=len(smooth_indices),
            blink_channel_count=len(blink_indices),
            overlap_frames=self.overlap_frames,
            actual_overlap_frames=actual_overlap,
            boundary_delta_before=boundary_delta_before,
            boundary_delta_after=boundary_delta_after,
            blink_enabled=self.blink_enabled and bool(blink_indices),
            blink_applied_count=blink_applied_count,
            blink_frame_count=blink_frame_count,
            blink_reset_count=blink_reset_count,
            global_frame_start=global_frame_start,
            global_frame_end=global_frame_end,
        )

    def _should_reset_smoothing(self, key: EyeContinuityKey, segment_index: int) -> bool:
        return (
            self._previous_key != key
            or self._previous_segment_index is None
            or segment_index != self._previous_segment_index + 1
        )

    def _store_tail(self, key: EyeContinuityKey, segment_index: int, frames: list[list[float]]) -> None:
        tail_length = min(self.overlap_frames, len(frames))
        self._previous_key = key
        self._previous_segment_index = segment_index
        self._previous_tail = copy.deepcopy(frames[-tail_length:]) if tail_length > 0 else []

    def _ensure_blink_state(self, *, session_id: str, turn_id: str, fps: int) -> int:
        should_reset = (
            self._blink_session_id != session_id
            or (self.reset_blink_on_new_turn and self._blink_turn_id is not None and self._blink_turn_id != turn_id)
        )
        reset_count = 1 if should_reset and self._blink_session_id is not None else 0
        if should_reset:
            self._blink_session_id = session_id
            self._blink_turn_id = turn_id
            self._global_frame_index = 0
            self._random = random.Random(self.seed)
            self._next_blink_frame = self._draw_interval_frames(fps)
        else:
            self._blink_turn_id = turn_id
        return reset_count

    def _apply_blink(
        self,
        frames: list[list[float]],
        *,
        blink_indices: list[int],
        global_frame_start: int,
        fps: int,
    ) -> tuple[int, int]:
        if not self.blink_enabled or not blink_indices or not frames:
            return 0, 0

        applied_events: set[int] = set()
        blink_frame_count = 0
        for local_index, frame in enumerate(frames):
            absolute_index = global_frame_start + local_index
            while absolute_index >= self._next_blink_frame + self.blink_duration_frames:
                self._next_blink_frame += self.blink_duration_frames + self._draw_interval_frames(fps)

            blink_offset = absolute_index - self._next_blink_frame
            if blink_offset < 0 or blink_offset >= self.blink_duration_frames:
                continue

            curve_value = _blink_curve_value(blink_offset, self.blink_duration_frames) * self.blink_strength
            if curve_value <= 0.0:
                continue

            blink_frame_count += 1
            applied_events.add(self._next_blink_frame)
            for channel_index in blink_indices:
                frame[channel_index] = max(float(frame[channel_index]), curve_value)

        return len(applied_events), blink_frame_count

    def _draw_interval_frames(self, fps: int) -> int:
        interval_seconds = self._random.uniform(self.blink_interval_min_sec, self.blink_interval_max_sec)
        return max(0, int(round(interval_seconds * fps)))


def _validate_frames(frames: list[list[float]]) -> int:
    if not frames:
        return 0
    channel_count = len(frames[0])
    for frame in frames:
        if len(frame) != channel_count:
            raise ValueError("frames must be rectangular")
    return channel_count


def _validate_channel_indices(channel_indices: list[int], channel_count: int) -> list[int]:
    unique_indices = list(dict.fromkeys(channel_indices))
    if channel_count == 0:
        if unique_indices:
            raise ValueError("channel index cannot be validated for empty frames")
        return unique_indices
    for channel_index in unique_indices:
        if channel_index < 0 or channel_index >= channel_count:
            raise ValueError(f"channel index {channel_index} out of range for {channel_count} channels")
    return unique_indices


def _mean_abs_delta_for_indices(left: list[float], right: list[float], channel_indices: list[int]) -> float:
    if not channel_indices:
        return 0.0
    deltas = [abs(float(left[index]) - float(right[index])) for index in channel_indices]
    return sum(deltas) / len(deltas)


def _blink_curve_value(offset: int, duration_frames: int) -> float:
    if duration_frames <= 1:
        return 1.0
    midpoint = (duration_frames - 1) / 2.0
    if midpoint == 0.0:
        return 1.0
    return max(0.0, 1.0 - abs(offset - midpoint) / midpoint)
