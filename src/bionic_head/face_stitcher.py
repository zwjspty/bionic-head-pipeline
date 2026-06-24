from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import copy


@dataclass(frozen=True)
class FaceStitchKey:
    session_id: str
    turn_id: str
    generation_epoch: int


@dataclass(frozen=True)
class FaceStitchMetrics:
    enabled: bool
    applied: bool
    reset: bool
    overlap_frames: int
    actual_overlap_frames: int
    boundary_delta_before: float | None = None
    boundary_delta_after: float | None = None

    def to_timing_payload(self) -> dict[str, bool | float]:
        payload: dict[str, bool | float] = {
            "face_stitch_enabled": self.enabled,
            "face_stitch_applied": self.applied,
            "face_stitch_reset": self.reset,
            "face_stitch_overlap_frames": float(self.overlap_frames),
            "face_stitch_actual_overlap_frames": float(self.actual_overlap_frames),
        }
        if self.boundary_delta_before is not None:
            payload["face_boundary_delta_before"] = self.boundary_delta_before
        if self.boundary_delta_after is not None:
            payload["face_boundary_delta_after"] = self.boundary_delta_after
        return payload


class FaceSegmentStitcher:
    def __init__(self, *, enabled: bool = True, overlap_frames: int = 8) -> None:
        if overlap_frames < 0:
            raise ValueError("overlap_frames must be non-negative")
        self.enabled = enabled
        self.overlap_frames = overlap_frames
        self._previous_key: FaceStitchKey | None = None
        self._previous_segment_index: int | None = None
        self._previous_tail: list[list[float]] = []

    def reset(self) -> None:
        self._previous_key = None
        self._previous_segment_index = None
        self._previous_tail = []

    def stitch(
        self,
        frames: list[list[float]],
        *,
        session_id: str,
        turn_id: str,
        generation_epoch: int,
        segment_index: int,
    ) -> tuple[list[list[float]], FaceStitchMetrics]:
        copied = copy.deepcopy(frames)
        if not self.enabled:
            return copied, FaceStitchMetrics(
                enabled=False,
                applied=False,
                reset=False,
                overlap_frames=self.overlap_frames,
                actual_overlap_frames=0,
            )

        key = FaceStitchKey(
            session_id=session_id,
            turn_id=turn_id,
            generation_epoch=generation_epoch,
        )
        reset = self._should_reset(key, segment_index)
        actual_overlap = 0 if reset else min(self.overlap_frames, len(self._previous_tail), len(copied))
        if actual_overlap <= 0:
            self._store_tail(key, segment_index, copied)
            return copied, FaceStitchMetrics(
                enabled=True,
                applied=False,
                reset=reset,
                overlap_frames=self.overlap_frames,
                actual_overlap_frames=0,
            )

        previous_overlap = self._previous_tail[-actual_overlap:]
        boundary_delta_before = _mean_abs_delta(previous_overlap[-1], copied[0])
        for index in range(actual_overlap):
            alpha = (index + 1) / float(actual_overlap)
            copied[index] = [
                previous * (1.0 - alpha) + current * alpha
                for previous, current in zip(previous_overlap[index], copied[index])
            ]
        boundary_delta_after = _mean_abs_delta(previous_overlap[-1], copied[0])
        self._store_tail(key, segment_index, copied)
        return copied, FaceStitchMetrics(
            enabled=True,
            applied=True,
            reset=False,
            overlap_frames=self.overlap_frames,
            actual_overlap_frames=actual_overlap,
            boundary_delta_before=boundary_delta_before,
            boundary_delta_after=boundary_delta_after,
        )

    def _should_reset(self, key: FaceStitchKey, segment_index: int) -> bool:
        return (
            self._previous_key != key
            or self._previous_segment_index is None
            or segment_index != self._previous_segment_index + 1
        )

    def _store_tail(self, key: FaceStitchKey, segment_index: int, frames: list[list[float]]) -> None:
        tail_length = min(self.overlap_frames, len(frames))
        self._previous_key = key
        self._previous_segment_index = segment_index
        self._previous_tail = copy.deepcopy(frames[-tail_length:]) if tail_length > 0 else []


def _mean_abs_delta(left: Iterable[float], right: Iterable[float]) -> float:
    deltas = [abs(float(previous) - float(current)) for previous, current in zip(left, right)]
    return sum(deltas) / len(deltas) if deltas else 0.0
