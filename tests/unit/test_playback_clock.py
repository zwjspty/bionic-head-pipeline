import pytest

from bionic_head.client.playback_clock import PlaybackClock


def test_clock_records_audio_face_offset() -> None:
    values = iter([0.0, 0.0, 0.6, 1.0, 1.2])
    clock = PlaybackClock(clock=lambda: next(values))

    clock.mark_tts_received("chunk-0001")
    clock.mark_audio_play_start("chunk-0001")
    clock.mark_ue5_received("chunk-0001")
    clock.mark_face_display("chunk-0001")

    metrics = clock.metrics()
    segment = clock.segment_metrics()["chunk-0001"]
    assert metrics["client_audio_play_start_ms"] == pytest.approx(600.0)
    assert metrics["client_face_first_frame_displayed_ms"] == pytest.approx(1200.0)
    assert metrics["client_audio_face_offset_ms"] == pytest.approx(600.0)
    assert metrics["client_face_late_by_ms"] == pytest.approx(600.0)
    assert segment["audio_play_start_ms"] == pytest.approx(600.0)
    assert segment["face_first_frame_displayed_ms"] == pytest.approx(1200.0)


def test_clock_records_wait_for_face_and_stop_offsets() -> None:
    values = iter([0.0, 0.0, 0.4, 0.9, 0.91, 1.0, 1.05, 1.06])
    clock = PlaybackClock(clock=lambda: next(values))

    clock.mark_tts_received("chunk-0001")
    clock.mark_ue5_received("chunk-0001")
    clock.mark_audio_play_start("chunk-0001")
    clock.mark_face_display("chunk-0001")
    clock.mark_playback_stop_received()
    clock.mark_audio_stopped()
    clock.mark_face_cleared()

    metrics = clock.metrics()
    assert metrics["client_audio_wait_for_face_ms"] == pytest.approx(900.0)
    assert metrics["client_audio_face_offset_ms"] == pytest.approx(10.0)
    assert metrics["client_playback_stop_to_audio_stop_ms"] == pytest.approx(50.0)
    assert metrics["client_playback_stop_to_face_clear_ms"] == pytest.approx(60.0)


def test_clock_records_wait_for_face_timeout() -> None:
    values = iter([0.0, 0.0, 0.1, 0.95])
    clock = PlaybackClock(clock=lambda: next(values))

    clock.mark_tts_received("chunk-0001")
    clock.mark_wait_for_face_timeout("chunk-0001")
    clock.mark_audio_play_start("chunk-0001")

    metrics = clock.metrics()
    segment = clock.segment_metrics()["chunk-0001"]
    assert metrics["client_audio_wait_for_face_timeout"] is True
    assert segment["audio_wait_for_face_timeout"] is True
