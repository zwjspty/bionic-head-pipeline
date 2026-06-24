from __future__ import annotations

from scripts.interrupt_stress import summarize_events


def test_summarize_events_counts_playback_stop_and_old_generation_face_leaks() -> None:
    events = [
        {"type": "server.session.ready", "sequence": 1, "generation_epoch": 0},
        {"type": "server.tts.audio", "sequence": 2, "generation_epoch": 0},
        {"type": "server.playback.stop", "sequence": 3, "generation_epoch": 1},
        {"type": "server.ue5.frames", "sequence": 4, "generation_epoch": 0},
        {"type": "server.face.frames", "sequence": 5, "generation_epoch": 1},
    ]

    summary = summarize_events(events)

    assert summary["events"] == 5
    assert summary["playback_stop_count"] == 1
    assert summary["latest_generation_epoch"] == 1
    assert summary["old_turn_face_leak_count"] == 1
    assert summary["strictly_increasing_sequences"] is True
