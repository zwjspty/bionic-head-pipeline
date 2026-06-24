from __future__ import annotations

import pytest

from bionic_head.eye_continuity import EyeContinuityProcessor


def _frames(values: list[float], *, channels: int = 52) -> list[list[float]]:
    return [[value] * channels for value in values]


def test_no_configured_eye_or_blink_channels_is_noop() -> None:
    processor = EyeContinuityProcessor(enabled=True)
    frames = _frames([0.1, 0.2, 0.3])

    processed, metrics = processor.process(
        frames,
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
        fps=30,
    )

    assert processed == frames
    assert processed is not frames
    assert metrics.enabled is True
    assert metrics.applied is False
    assert metrics.reset is True
    assert metrics.smooth_channel_count == 0
    assert metrics.blink_channel_count == 0
    assert metrics.actual_overlap_frames == 0
    assert metrics.boundary_delta_before is None
    assert metrics.boundary_delta_after is None
    assert metrics.blink_enabled is False
    assert metrics.blink_applied_count == 0
    assert metrics.blink_frame_count == 0
    assert metrics.global_frame_start == 0
    assert metrics.global_frame_end == 3


def test_eye_smoothing_modifies_only_configured_channels_and_reduces_delta() -> None:
    processor = EyeContinuityProcessor(
        enabled=True,
        eye_smooth_channel_indices=[2, 5],
        overlap_frames=2,
    )
    processor.process(
        _frames([0.0, 0.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
        fps=30,
    )

    processed, metrics = processor.process(
        _frames([1.0, 1.0, 1.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=2,
        fps=30,
    )

    assert metrics.applied is True
    assert metrics.reset is False
    assert metrics.actual_overlap_frames == 2
    assert metrics.boundary_delta_before == pytest.approx(1.0)
    assert metrics.boundary_delta_after == pytest.approx(0.5)
    assert metrics.boundary_delta_after <= metrics.boundary_delta_before
    assert processed[0][2] == pytest.approx(0.5)
    assert processed[0][5] == pytest.approx(0.5)
    assert processed[1][2] == pytest.approx(1.0)
    assert processed[1][5] == pytest.approx(1.0)
    assert processed[0][0] == pytest.approx(1.0)
    assert processed[0][4] == pytest.approx(1.0)


def test_global_frame_index_continues_across_same_session_segments() -> None:
    processor = EyeContinuityProcessor(enabled=True)

    first, first_metrics = processor.process(
        _frames([0.0, 0.0, 0.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
        fps=30,
    )
    second, second_metrics = processor.process(
        _frames([0.0, 0.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=2,
        fps=30,
    )

    assert len(first) == 3
    assert len(second) == 2
    assert first_metrics.global_frame_start == 0
    assert first_metrics.global_frame_end == 3
    assert second_metrics.global_frame_start == 3
    assert second_metrics.global_frame_end == 5


def test_session_change_resets_global_frame_index() -> None:
    processor = EyeContinuityProcessor(enabled=True)
    processor.process(
        _frames([0.0, 0.0, 0.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
        fps=30,
    )

    _, metrics = processor.process(
        _frames([0.0, 0.0]),
        session_id="s2",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
        fps=30,
    )

    assert metrics.reset is True
    assert metrics.blink_reset_count == 1
    assert metrics.global_frame_start == 0
    assert metrics.global_frame_end == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"session_id": "s1", "turn_id": "t2", "generation_epoch": 0, "segment_index": 2},
        {"session_id": "s1", "turn_id": "t1", "generation_epoch": 1, "segment_index": 2},
        {"session_id": "s1", "turn_id": "t1", "generation_epoch": 0, "segment_index": 4},
    ],
)
def test_turn_generation_or_non_consecutive_segment_resets_eye_smoothing(
    kwargs: dict[str, object],
) -> None:
    processor = EyeContinuityProcessor(
        enabled=True,
        eye_smooth_channel_indices=[0],
        overlap_frames=2,
    )
    processor.process(
        _frames([0.0, 0.0]),
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
        fps=30,
    )

    processed, metrics = processor.process(_frames([1.0, 1.0]), fps=30, **kwargs)

    assert processed == _frames([1.0, 1.0])
    assert metrics.reset is True
    assert metrics.applied is False
    assert metrics.actual_overlap_frames == 0
    assert metrics.boundary_delta_before is None
    assert metrics.boundary_delta_after is None


def test_blink_scheduler_applies_deterministic_curve_without_changing_frame_count() -> None:
    processor = EyeContinuityProcessor(
        enabled=True,
        blink_enabled=True,
        blink_channel_indices=[4],
        blink_interval_min_sec=0.1,
        blink_interval_max_sec=0.1,
        blink_duration_frames=5,
        blink_strength=1.0,
        seed=7,
    )
    frames = [[0.0] * 52 for _ in range(6)]

    processed, metrics = processor.process(
        frames,
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
        fps=10,
    )

    assert len(processed) == len(frames)
    assert metrics.applied is True
    assert metrics.blink_enabled is True
    assert metrics.blink_applied_count == 1
    assert metrics.blink_frame_count == 3
    assert [frame[4] for frame in processed] == pytest.approx([0.0, 0.0, 0.5, 1.0, 0.5, 0.0])
    assert all(frame[3] == 0.0 for frame in processed)


def test_disabled_processor_returns_unchanged_frames() -> None:
    processor = EyeContinuityProcessor(
        enabled=False,
        eye_smooth_channel_indices=[0],
        blink_enabled=True,
        blink_channel_indices=[1],
    )
    frames = _frames([0.1, 0.2])

    processed, metrics = processor.process(
        frames,
        session_id="s1",
        turn_id="t1",
        generation_epoch=0,
        segment_index=1,
        fps=30,
    )

    assert processed == frames
    assert processed is not frames
    assert metrics.enabled is False
    assert metrics.applied is False
    assert metrics.blink_applied_count == 0
    assert metrics.blink_frame_count == 0


def test_invalid_channel_index_raises_value_error() -> None:
    processor = EyeContinuityProcessor(enabled=True, eye_smooth_channel_indices=[52])

    with pytest.raises(ValueError, match="channel index"):
        processor.process(
            _frames([0.0]),
            session_id="s1",
            turn_id="t1",
            generation_epoch=0,
            segment_index=1,
            fps=30,
        )
