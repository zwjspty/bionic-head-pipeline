from __future__ import annotations

from conftest import post_audio, strictly_increasing_sequences, terminal_types


def test_p0_acceptance(app, speech_wav, websocket_turn):
    offline = post_audio(app, speech_wav)
    assert offline.status_code == 200
    assert offline.json()["face"]["channel_count"] == 52

    events = websocket_turn(app)
    assert terminal_types(events) == ["server.pipeline.done"]
    assert strictly_increasing_sequences(events)
