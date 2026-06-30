from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field


class UE5PlaybackContractError(ValueError):
    """Raised when a UE5 playback event violates the local playback contract."""


@dataclass(frozen=True)
class UE5PlaybackAction:
    action: str
    accepted: bool
    reason: str
    generation_epoch: int
    chunk_id: str | None = None
    frame_count: int = 0


@dataclass
class UE5PlaybackReceiverState:
    """Small UE5-side replay model for buffer/stale/drop contract checks."""

    active_generation_epoch: int = 0
    seen_chunk_ids: set[str] = field(default_factory=set)
    _next_frame_index_by_segment: dict[tuple[int, str], int] = field(default_factory=dict)
    metrics: dict[str, int] = field(
        default_factory=lambda: {
            "received_frame_chunks": 0,
            "buffered_frame_count": 0,
            "stale_drop_count": 0,
            "duplicate_drop_count": 0,
            "missing_or_gap_count": 0,
            "playback_stop_count": 0,
            "buffer_clear_count": 0,
        }
    )

    def accept_ue5_frame_chunk(self, payload: Mapping[str, object]) -> UE5PlaybackAction:
        chunk = validate_ue5_frame_chunk(payload)
        generation_epoch = int(chunk["generation_epoch"])
        chunk_id = _optional_string(chunk.get("chunk_id"))
        segment_id = _optional_string(chunk.get("segment_id")) or "_default"
        frame_count = int(chunk["frame_count"])

        if generation_epoch < self.active_generation_epoch:
            self.metrics["stale_drop_count"] += 1
            return UE5PlaybackAction(
                action="drop",
                accepted=False,
                reason="stale_generation",
                generation_epoch=generation_epoch,
                chunk_id=chunk_id,
                frame_count=frame_count,
            )

        if generation_epoch > self.active_generation_epoch:
            self.active_generation_epoch = generation_epoch
            self._clear_buffer()

        if chunk_id and chunk_id in self.seen_chunk_ids:
            self.metrics["duplicate_drop_count"] += 1
            return UE5PlaybackAction(
                action="drop",
                accepted=False,
                reason="duplicate_chunk",
                generation_epoch=generation_epoch,
                chunk_id=chunk_id,
                frame_count=frame_count,
            )

        key = (generation_epoch, segment_id)
        expected_start = self._next_frame_index_by_segment.get(key)
        start_frame_index = int(chunk["start_frame_index"])
        reason = "accepted"
        if expected_start is not None and start_frame_index != expected_start:
            self.metrics["missing_or_gap_count"] += 1
            reason = "gap_detected"

        if chunk_id:
            self.seen_chunk_ids.add(chunk_id)
        self._next_frame_index_by_segment[key] = start_frame_index + frame_count
        self.metrics["received_frame_chunks"] += 1
        self.metrics["buffered_frame_count"] += frame_count
        return UE5PlaybackAction(
            action="accept",
            accepted=True,
            reason=reason,
            generation_epoch=generation_epoch,
            chunk_id=chunk_id,
            frame_count=frame_count,
        )

    def accept_playback_stop(self, payload: Mapping[str, object]) -> UE5PlaybackAction:
        stop = validate_playback_stop(payload)
        generation_epoch = int(stop["generation_epoch"])
        if generation_epoch > self.active_generation_epoch:
            self.active_generation_epoch = generation_epoch
        self.metrics["playback_stop_count"] += 1
        self._clear_buffer()
        return UE5PlaybackAction(
            action="clear",
            accepted=True,
            reason=_optional_string(stop.get("reason")) or "playback_stop",
            generation_epoch=self.active_generation_epoch,
        )

    def _clear_buffer(self) -> None:
        self.seen_chunk_ids.clear()
        self._next_frame_index_by_segment.clear()
        self.metrics["buffered_frame_count"] = 0
        self.metrics["buffer_clear_count"] += 1


def replay_ue5_events(events: list[Mapping[str, object]]) -> dict[str, int]:
    """Replay stream-like UE5 events and return receiver-side contract metrics."""

    state = UE5PlaybackReceiverState()
    for event in events:
        event_type = _optional_string(event.get("type"))
        if event_type == "server.playback.stop":
            payload = event.get("payload", event)
            if not isinstance(payload, Mapping):
                raise UE5PlaybackContractError("server.playback.stop payload must be an object")
            state.accept_playback_stop(payload)
        elif event_type == "server.ue5.frames" or "frames" in event:
            payload = event.get("payload", event)
            if not isinstance(payload, Mapping):
                raise UE5PlaybackContractError("server.ue5.frames payload must be an object")
            state.accept_ue5_frame_chunk(payload)
        else:
            continue

    return {"active_generation_epoch": state.active_generation_epoch, **state.metrics}


def validate_ue5_frame_chunk(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate a server.ue5.frames payload for the playback contract.

    The validator intentionally checks the contract shape, ordering metadata,
    and numeric safety without knowing anything about Unreal Engine internals.
    It returns a shallow normalized copy so callers can safely use numeric
    fields after validation.
    """

    data = dict(payload)

    _require_equal(data, "format", "morpheus_52_raw")
    channel_count = _require_int(data, "channel_count", min_value=0)
    if channel_count != 52:
        raise UE5PlaybackContractError("channel_count must be 52")

    fps = _require_number(data, "fps")
    if fps <= 0:
        raise UE5PlaybackContractError("fps must be positive")

    generation_epoch = _require_int(data, "generation_epoch", min_value=0)
    segment_index = data.get("segment_index")
    if segment_index is not None:
        _ensure_int_value(segment_index, "segment_index", min_value=0)

    start_frame_index = _require_int(data, "start_frame_index", min_value=0)
    frame_count = _require_int(data, "frame_count", min_value=0)

    if "pts_start_ms" in data and data["pts_start_ms"] is not None:
        pts_start_ms = _require_number(data, "pts_start_ms")
        if pts_start_ms < 0:
            raise UE5PlaybackContractError("pts_start_ms must be >= 0")

    frames = data.get("frames")
    if not isinstance(frames, list):
        raise UE5PlaybackContractError("frames must be a list")
    if len(frames) != frame_count:
        raise UE5PlaybackContractError("frame_count must match len(frames)")

    normalized_frames: list[dict[str, object]] = []
    for offset, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise UE5PlaybackContractError("frames entries must be objects")
        frame_data = dict(frame)
        frame_index = _require_int(frame_data, "frame_index", min_value=0)
        expected_frame_index = start_frame_index + offset
        if frame_index != expected_frame_index:
            raise UE5PlaybackContractError(
                "frame_index must be contiguous from start_frame_index"
            )

        weights = frame_data.get("weights")
        if not isinstance(weights, list):
            raise UE5PlaybackContractError("weights must be a list")
        if len(weights) != 52:
            raise UE5PlaybackContractError("weights must contain exactly 52 values")

        normalized_weights = [_require_finite_number(weight, "weights") for weight in weights]
        frame_data["weights"] = normalized_weights
        normalized_frames.append(frame_data)

    data["channel_count"] = channel_count
    data["fps"] = fps
    data["generation_epoch"] = generation_epoch
    data["start_frame_index"] = start_frame_index
    data["frame_count"] = frame_count
    data["frames"] = normalized_frames
    return data


def validate_playback_stop(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate a server.playback.stop payload."""

    data = dict(payload)
    data["generation_epoch"] = _require_int(data, "generation_epoch", min_value=0)
    return data


def _require_equal(data: Mapping[str, object], field: str, expected: object) -> None:
    if data.get(field) != expected:
        raise UE5PlaybackContractError(f"{field} must be {expected!r}")


def _require_int(data: Mapping[str, object], field: str, *, min_value: int | None = None) -> int:
    value = data.get(field)
    return _ensure_int_value(value, field, min_value=min_value)


def _ensure_int_value(
    value: object, field: str, *, min_value: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UE5PlaybackContractError(f"{field} must be an integer")
    if min_value is not None and value < min_value:
        raise UE5PlaybackContractError(f"{field} must be >= {min_value}")
    return value


def _require_number(data: Mapping[str, object], field: str) -> float:
    return _require_finite_number(data.get(field), field)


def _require_finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UE5PlaybackContractError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise UE5PlaybackContractError(f"{field} must be a finite number")
    return normalized


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise UE5PlaybackContractError("optional string fields must be strings")
    return value
