from __future__ import annotations

import math

import pytest

from bionic_head.ue5_playback_contract import (
    UE5PlaybackContractError,
    validate_playback_stop,
    validate_ue5_frame_chunk,
)


def valid_chunk(**overrides):
    payload = {
        "protocol": "bionic-head-ue5-v1",
        "format": "morpheus_52_raw",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "generation_epoch": 0,
        "chunk_id": "chunk-0001-0000",
        "segment_id": "chunk-0001",
        "segment_index": 0,
        "fps": 30,
        "channel_count": 52,
        "channels": [f"morpheus_{index:02d}" for index in range(52)],
        "start_frame_index": 0,
        "frame_count": 2,
        "pts_start_ms": 0.0,
        "is_last": True,
        "frames": [
            {"frame_index": 0, "time_seconds": 0.0, "weights": [0.0] * 52},
            {"frame_index": 1, "time_seconds": 1.0 / 30.0, "weights": [0.1] * 52},
        ],
    }
    payload.update(overrides)
    return payload


def test_validates_valid_ue5_frame_chunk() -> None:
    normalized = validate_ue5_frame_chunk(valid_chunk())

    assert normalized["format"] == "morpheus_52_raw"
    assert normalized["generation_epoch"] == 0
    assert normalized["start_frame_index"] == 0
    assert normalized["frame_count"] == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "arkit", "format"),
        ("channel_count", 51, "channel_count"),
        ("fps", 0, "fps"),
        ("generation_epoch", -1, "generation_epoch"),
        ("segment_index", -1, "segment_index"),
        ("start_frame_index", -1, "start_frame_index"),
        ("pts_start_ms", -1.0, "pts_start_ms"),
    ],
)
def test_rejects_invalid_scalar_fields(field: str, value: object, message: str) -> None:
    with pytest.raises(UE5PlaybackContractError, match=message):
        validate_ue5_frame_chunk(valid_chunk(**{field: value}))


def test_rejects_missing_start_frame_index() -> None:
    payload = valid_chunk()
    del payload["start_frame_index"]

    with pytest.raises(UE5PlaybackContractError, match="start_frame_index"):
        validate_ue5_frame_chunk(payload)


def test_rejects_frame_count_mismatch() -> None:
    with pytest.raises(UE5PlaybackContractError, match="frame_count"):
        validate_ue5_frame_chunk(valid_chunk(frame_count=3))


def test_rejects_non_contiguous_frame_index() -> None:
    payload = valid_chunk()
    payload["frames"][1]["frame_index"] = 3

    with pytest.raises(UE5PlaybackContractError, match="frame_index"):
        validate_ue5_frame_chunk(payload)


def test_rejects_wrong_frame_weight_length() -> None:
    payload = valid_chunk()
    payload["frames"][0]["weights"] = [0.0] * 51

    with pytest.raises(UE5PlaybackContractError, match="weights"):
        validate_ue5_frame_chunk(payload)


def test_rejects_non_finite_frame_weight() -> None:
    payload = valid_chunk()
    payload["frames"][0]["weights"][0] = math.nan

    with pytest.raises(UE5PlaybackContractError, match="finite"):
        validate_ue5_frame_chunk(payload)


def test_validates_playback_stop_payload() -> None:
    normalized = validate_playback_stop({"generation_epoch": 2, "reason": "client_cancel"})

    assert normalized["generation_epoch"] == 2
    assert normalized["reason"] == "client_cancel"


def test_rejects_invalid_playback_stop_generation_epoch() -> None:
    with pytest.raises(UE5PlaybackContractError, match="generation_epoch"):
        validate_playback_stop({"generation_epoch": -1})
