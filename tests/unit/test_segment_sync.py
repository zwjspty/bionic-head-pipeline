from bionic_head.client.playback_clock import PlaybackClock
from bionic_head.client.segment_sync import SegmentSyncCoordinator


def _clock(*values: float) -> PlaybackClock:
    iterator = iter(values)
    return PlaybackClock(clock=lambda: next(iterator))


def test_immediate_audio_releases_audio_when_tts_arrives() -> None:
    sync = SegmentSyncCoordinator(strategy="immediate_audio", clock=_clock(0.0, 0.1))

    actions = sync.accept_tts(
        segment_id="chunk-0001",
        chunk_id="chunk-0001",
        wav_bytes=b"wav",
        generation_epoch=0,
    )

    assert [action.kind for action in actions] == ["play_audio"]
    assert actions[0].segment_id == "chunk-0001"
    assert actions[0].wav_bytes == b"wav"


def test_wait_for_face_holds_tts_until_matching_face_arrives() -> None:
    sync = SegmentSyncCoordinator(strategy="wait_for_face", clock=_clock(0.0, 0.1, 0.5))

    assert (
        sync.accept_tts(
            segment_id="chunk-0001",
            chunk_id="chunk-0001",
            wav_bytes=b"wav",
            generation_epoch=0,
        )
        == []
    )
    actions = sync.accept_face(
        segment_id="chunk-0001",
        chunk_id="chunk-0001-0000",
        payload={"frame_count": 2},
        generation_epoch=0,
    )

    assert [action.kind for action in actions] == ["play_audio", "display_face"]
    assert actions[0].wav_bytes == b"wav"
    assert actions[1].face_payload == {"frame_count": 2}


def test_wait_for_face_handles_face_before_tts() -> None:
    sync = SegmentSyncCoordinator(strategy="wait_for_face", clock=_clock(0.0, 0.2, 0.4))

    assert (
        sync.accept_face(
            segment_id="chunk-0001",
            chunk_id="chunk-0001-0000",
            payload={"frame_count": 2},
            generation_epoch=0,
        )
        == []
    )
    actions = sync.accept_tts(
        segment_id="chunk-0001",
        chunk_id="chunk-0001",
        wav_bytes=b"wav",
        generation_epoch=0,
    )

    assert [action.kind for action in actions] == ["play_audio", "display_face"]


def test_wait_for_face_timeout_releases_audio_without_face() -> None:
    sync = SegmentSyncCoordinator(
        strategy="wait_for_face",
        clock=_clock(0.0, 0.1),
        wait_for_face_timeout_ms=800,
    )
    sync.accept_tts(
        segment_id="chunk-0001",
        chunk_id="chunk-0001",
        wav_bytes=b"wav",
        generation_epoch=0,
    )

    actions = sync.flush_timeouts(now_ms=901.0)

    assert [action.kind for action in actions] == ["play_audio"]
    assert sync.clock.metrics()["client_audio_wait_for_face_timeout"] is True


def test_stale_generation_events_are_dropped() -> None:
    sync = SegmentSyncCoordinator(strategy="immediate_audio", clock=_clock(0.0, 0.1))
    sync.update_latest_generation(2)

    assert (
        sync.accept_tts(
            segment_id="chunk-old",
            chunk_id="chunk-old",
            wav_bytes=b"old",
            generation_epoch=1,
        )
        == []
    )
    assert sync.stale_audio_drop_count == 1


def test_clear_removes_pending_segments() -> None:
    sync = SegmentSyncCoordinator(strategy="wait_for_face", clock=_clock(0.0, 0.1))
    sync.accept_tts(
        segment_id="chunk-0001",
        chunk_id="chunk-0001",
        wav_bytes=b"wav",
        generation_epoch=0,
    )

    sync.clear(reason="playback_stop")

    assert sync.pending_count == 0
    assert sync.flush_timeouts(now_ms=1000.0) == []
