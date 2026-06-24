from __future__ import annotations

import pytest

from bionic_head.face_stitcher import FaceSegmentStitcher


def _frames(values: list[float]) -> list[list[float]]:
    return [[value] * 52 for value in values]


def test_first_segment_is_unchanged_and_not_applied() -> None:
    stitcher = FaceSegmentStitcher(enabled=True, overlap_frames=3)
    frames = _frames([0.1, 0.2, 0.3])

    stitched, metrics = stitcher.stitch(
        frames,
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
    )

    assert stitched == frames
    assert stitched is not frames
    assert metrics.enabled is True
    assert metrics.applied is False
    assert metrics.reset is True
    assert metrics.actual_overlap_frames == 0
    assert metrics.boundary_delta_before is None
    assert metrics.boundary_delta_after is None


def test_second_consecutive_segment_crossfades_head_and_reduces_boundary_delta() -> None:
    stitcher = FaceSegmentStitcher(enabled=True, overlap_frames=3)
    stitcher.stitch(
        _frames([0.0, 0.0, 0.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
    )

    stitched, metrics = stitcher.stitch(
        _frames([1.0, 1.0, 1.0, 1.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=2,
    )

    assert metrics.applied is True
    assert metrics.reset is False
    assert metrics.actual_overlap_frames == 3
    assert metrics.boundary_delta_before == pytest.approx(1.0)
    assert metrics.boundary_delta_after == pytest.approx(1.0 / 3.0)
    assert stitched[0] == pytest.approx([1.0 / 3.0] * 52)
    assert stitched[1] == pytest.approx([2.0 / 3.0] * 52)
    assert stitched[2] == pytest.approx([1.0] * 52)
    assert stitched[3] == pytest.approx([1.0] * 52)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"session_id": "s2", "turn_id": "t1", "generation_epoch": 0, "segment_index": 2},
        {"session_id": "s1", "turn_id": "t2", "generation_epoch": 0, "segment_index": 2},
        {"session_id": "s1", "turn_id": "t1", "generation_epoch": 1, "segment_index": 2},
        {"session_id": "s1", "turn_id": "t1", "generation_epoch": 0, "segment_index": 4},
    ],
)
def test_context_change_or_non_consecutive_segment_resets(kwargs: dict[str, object]) -> None:
    stitcher = FaceSegmentStitcher(enabled=True, overlap_frames=2)
    stitcher.stitch(
        _frames([0.0, 0.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
    )

    stitched, metrics = stitcher.stitch(_frames([1.0, 1.0]), **kwargs)

    assert stitched == _frames([1.0, 1.0])
    assert metrics.reset is True
    assert metrics.applied is False
    assert metrics.actual_overlap_frames == 0


def test_overlap_is_clamped_for_short_segments() -> None:
    stitcher = FaceSegmentStitcher(enabled=True, overlap_frames=8)
    stitcher.stitch(
        _frames([0.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
    )

    stitched, metrics = stitcher.stitch(
        _frames([1.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=2,
    )

    assert metrics.actual_overlap_frames == 1
    assert stitched == _frames([1.0])


def test_disabled_stitcher_returns_unchanged_frames_without_stateful_crossfade() -> None:
    stitcher = FaceSegmentStitcher(enabled=False, overlap_frames=8)

    first, first_metrics = stitcher.stitch(
        _frames([0.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
    )
    second, second_metrics = stitcher.stitch(
        _frames([1.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=2,
    )

    assert first == _frames([0.0])
    assert second == _frames([1.0])
    assert first_metrics.enabled is False
    assert second_metrics.enabled is False
    assert first_metrics.applied is False
    assert second_metrics.applied is False
