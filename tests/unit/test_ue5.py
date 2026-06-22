from __future__ import annotations

import math

import pytest

from bionic_head.core.ue5 import build_ue5_payload, chunk_ue5_frames
from bionic_head.domain.errors import ErrorCode, PipelineException


def test_formats_and_chunks_52_channel_frames() -> None:
    payload = build_ue5_payload([[0.0] * 52 for _ in range(31)], fps=30)
    chunks = list(chunk_ue5_frames(payload, chunk_size=30, chunk_id="s0"))

    assert payload.channels[0] == "morpheus_00"
    assert payload.channels[-1] == "morpheus_51"
    assert [chunk["frame_count"] for chunk in chunks] == [30, 1]
    assert chunks[0]["is_last"] is False
    assert chunks[-1]["is_last"] is True
    assert chunks[-1]["frames"][0]["frame_index"] == 30
    assert chunks[-1]["frames"][0]["time_seconds"] == pytest.approx(1.0)


def test_rejects_non_52_channel_frames() -> None:
    with pytest.raises(PipelineException) as captured:
        build_ue5_payload([[0.0] * 51], fps=30)

    assert captured.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED
    assert captured.value.stage == "ue5"


def test_rejects_non_finite_weights() -> None:
    frame = [0.0] * 52
    frame[0] = math.nan

    with pytest.raises(PipelineException) as captured:
        build_ue5_payload([frame], fps=30)

    assert captured.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED
