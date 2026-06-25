import json
import sys
import wave
from collections.abc import Callable
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import scripts.local_demo_client as local_demo_client
from scripts.local_demo_client import (
    AudioPlaybackEngine,
    FacePlaybackEngine,
    LocalDemoReceiver,
    MemoryAudioSink,
    PlaybackMetrics,
    ProtocolError,
    SoundDeviceAudioSink,
    build_parser,
    run_local_demo,
)


SESSION_ID = UUID("00000000-0000-0000-0000-000000000011")
TURN_ID = UUID("00000000-0000-0000-0000-000000000012")


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


class FakeWebSocket:
    def __init__(self, responses: list[str | bytes]) -> None:
        self._responses = list(responses)
        self.sent: list[str | bytes] = []

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        if not self._responses:
            raise AssertionError("fake websocket exhausted")
        return self._responses.pop(0)


class FakeConnect:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> FakeWebSocket:
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


@pytest.fixture
def server_envelope():
    next_sequence = 1

    def _build(
        event_type: str,
        *,
        payload: dict[str, object],
        generation_epoch: int = 0,
    ) -> dict[str, object]:
        nonlocal next_sequence
        envelope = {
            "protocol": "bionic-head-stream-v1",
            "type": event_type,
            "event_id": str(uuid4()),
            "session_id": str(SESSION_ID),
            "turn_id": str(TURN_ID),
            "sequence": next_sequence,
            "generation_epoch": generation_epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "session_id": str(SESSION_ID),
                "turn_id": str(TURN_ID),
                "generation_epoch": generation_epoch,
                **payload,
            },
        }
        next_sequence += 1
        return envelope

    return _build


def test_build_parser_accepts_no_audio_mode() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--url",
            "ws://127.0.0.1:8005/pipeline/stream",
            "--wav",
            "/tmp/input.wav",
            "--output-dir",
            "/tmp/out",
            "--no-play-audio",
        ]
    )

    assert args.play_audio is False


def test_build_parser_accepts_cancel_after_ms() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--url",
            "ws://127.0.0.1:8005/pipeline/stream",
            "--wav",
            "/tmp/input.wav",
            "--output-dir",
            "/tmp/out",
            "--cancel-after-ms",
            "500",
        ]
    )

    assert args.cancel_after_ms == 500


def test_build_parser_defaults_to_audio_playback() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--url",
            "ws://127.0.0.1:8005/pipeline/stream",
            "--wav",
            "/tmp/input.wav",
            "--output-dir",
            "/tmp/out",
        ]
    )

    assert args.play_audio is True


def _wav_bytes_from_int16_samples(samples: list[int], sample_rate: int = 16000) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
    return buffer.getvalue()


def test_sounddevice_audio_sink_decodes_wav_and_plays(monkeypatch: pytest.MonkeyPatch) -> None:
    played: list[tuple[object, int]] = []
    fake_sounddevice = SimpleNamespace(
        play=lambda samples, samplerate: played.append((samples, samplerate)),
        stop=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    sink = SoundDeviceAudioSink()
    sink.play(_wav_bytes_from_int16_samples([0, 1000, -1000]))

    assert len(played) == 1
    samples, samplerate = played[0]
    assert samplerate == 16000
    assert samples.tolist() == [0, 1000, -1000]


def test_sounddevice_audio_sink_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sounddevice":
            raise ImportError("missing sounddevice")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(SystemExit, match="sounddevice is required for audio playback"):
        SoundDeviceAudioSink()


@pytest.mark.asyncio
async def test_run_local_demo_streams_audio_and_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    server_envelope,
) -> None:
    ready = server_envelope("server.session.ready", payload={})
    tts = server_envelope(
        "server.tts.audio",
        payload={
            "chunk_id": "chunk-1",
            "segment_id": "segment-1",
            "format": "wav",
            "byte_length": 12,
            "generation_epoch": 0,
        },
    )
    done = server_envelope("server.pipeline.done", payload={})
    websocket = FakeWebSocket(
        [json.dumps(ready), json.dumps(tts), b"RIFF....WAVE", json.dumps(done)]
    )

    monkeypatch.setattr(
        local_demo_client,
        "read_pcm16_from_wav",
        lambda _: b"\x01\x02" * 320,
    )
    monkeypatch.setattr(local_demo_client, "pcm_chunks", lambda pcm, *, chunk_ms: [pcm])
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        SimpleNamespace(connect=lambda url: FakeConnect(websocket)),
    )

    terminal = await run_local_demo(
        "ws://127.0.0.1:8005/pipeline/stream",
        tmp_path / "input.wav",
        tmp_path / "out",
        20,
        play_audio=False,
    )

    assert terminal == "server.pipeline.done"
    sent_events = [json.loads(message) for message in websocket.sent if isinstance(message, str)]
    assert [event["type"] for event in sent_events] == [
        "client.session.start",
        "client.audio.start",
        "client.audio.chunk",
        "client.audio.end",
    ]
    assert websocket.sent[3] == b"\x01\x02" * 320
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["tts_chunks"] == 1
    assert summary["terminal_event"] == "server.pipeline.done"


@pytest.mark.asyncio
async def test_run_local_demo_sends_turn_cancel_after_delay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    server_envelope,
) -> None:
    ready = server_envelope("server.session.ready", payload={})
    cancelled = server_envelope("server.turn.cancelled", payload={}, generation_epoch=1)
    websocket = FakeWebSocket([json.dumps(ready), json.dumps(cancelled)])

    monkeypatch.setattr(
        local_demo_client,
        "read_pcm16_from_wav",
        lambda _: b"\x01\x02" * 320,
    )
    monkeypatch.setattr(local_demo_client, "pcm_chunks", lambda pcm, *, chunk_ms: [pcm])
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        SimpleNamespace(connect=lambda url: FakeConnect(websocket)),
    )

    terminal = await run_local_demo(
        "ws://127.0.0.1:8005/pipeline/stream",
        tmp_path / "input.wav",
        tmp_path / "out",
        20,
        play_audio=False,
        cancel_after_ms=0,
    )

    assert terminal == "server.turn.cancelled"
    sent_events = [json.loads(message) for message in websocket.sent if isinstance(message, str)]
    assert [event["type"] for event in sent_events] == [
        "client.session.start",
        "client.audio.start",
        "client.audio.chunk",
        "client.audio.end",
        "client.turn.cancel",
    ]
    assert sent_events[-1]["sequence"] == 5


@pytest.mark.asyncio
async def test_run_local_demo_rejects_non_ready_first_event_without_streaming_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    server_envelope,
) -> None:
    first_error = server_envelope("server.pipeline.error", payload={"message": "boom"})
    websocket = FakeWebSocket([json.dumps(first_error)])

    monkeypatch.setattr(
        local_demo_client,
        "read_pcm16_from_wav",
        lambda _: b"\x01\x02" * 320,
    )
    monkeypatch.setattr(local_demo_client, "pcm_chunks", lambda pcm, *, chunk_ms: [pcm])
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        SimpleNamespace(connect=lambda url: FakeConnect(websocket)),
    )

    with pytest.raises(ProtocolError, match="expected first server event to be server.session.ready"):
        await run_local_demo(
            "ws://127.0.0.1:8005/pipeline/stream",
            tmp_path / "input.wav",
            tmp_path / "out",
            20,
            play_audio=False,
        )

    sent_events = [json.loads(message) for message in websocket.sent if isinstance(message, str)]
    assert [event["type"] for event in sent_events] == ["client.session.start"]


def test_audio_engine_enqueues_wav_and_records_metrics(fake_clock: Callable[[], float]) -> None:
    metrics = PlaybackMetrics(clock=fake_clock)
    sink = MemoryAudioSink()
    audio = AudioPlaybackEngine(metrics, sink=sink)

    audio.enqueue_wav("chunk-1", b"RIFF....WAVE", generation_epoch=0)

    assert audio.queued_count == 1
    assert sink.played_chunks == [b"RIFF....WAVE"]
    assert metrics.to_dict()["client_audio_enqueued_count"] == 1
    assert metrics.to_dict()["client_audio_play_start_ms"] == 0.0


def test_stop_clears_audio_and_face_buffers(fake_clock: Callable[[], float]) -> None:
    metrics = PlaybackMetrics(clock=fake_clock)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)

    audio.enqueue_wav("chunk-1", b"RIFF....WAVE", generation_epoch=0)
    face.enqueue_frames("ue5-1", {"frames": [{"frame_index": 0}]}, generation_epoch=0)
    fake_clock.advance(0.050)

    metrics.mark_playback_stop_received()
    audio.stop()
    audio.clear()
    face.clear()

    summary = metrics.to_dict()
    assert audio.queued_count == 0
    assert face.buffered_chunk_count == 0
    assert summary["client_audio_stopped_ms"] == 50.0
    assert summary["client_face_buffer_cleared_ms"] == 50.0


def test_receiver_accepts_tts_metadata_then_binary(tmp_path, server_envelope) -> None:
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    receiver.accept_json(
        server_envelope(
            "server.tts.audio",
            payload={
                "chunk_id": "chunk-1",
                "segment_id": "segment-1",
                "format": "wav",
                "byte_length": 12,
                "generation_epoch": 0,
            },
            generation_epoch=0,
        )
    )
    receiver.accept_binary(b"RIFF....WAVE")

    assert (tmp_path / "tts" / "chunk-1.wav").read_bytes() == b"RIFF....WAVE"
    assert audio.queued_count == 1


def test_receiver_rejects_binary_length_mismatch(tmp_path, server_envelope) -> None:
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    receiver.accept_json(
        server_envelope(
            "server.tts.audio",
            payload={
                "chunk_id": "chunk-1",
                "segment_id": "segment-1",
                "format": "wav",
                "byte_length": 12,
            },
        )
    )

    with pytest.raises(ProtocolError):
        receiver.accept_binary(b"short")

    assert receiver.pending_tts is None
    assert audio.queued_count == 0


def test_receiver_playback_stop_clears_buffers(tmp_path, server_envelope) -> None:
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    audio.enqueue_wav("chunk-1", b"RIFF....WAVE", generation_epoch=0)
    face.enqueue_frames("ue5-1", {"frames": [{"frame_index": 0}]}, generation_epoch=0)
    receiver.accept_json(server_envelope("server.playback.stop", payload={}, generation_epoch=1))

    assert audio.queued_count == 0
    assert face.buffered_chunk_count == 0
    assert metrics.to_dict()["client_playback_stop_received_ms"] == 0.0


def test_receiver_playback_stop_clears_pending_tts_binary(tmp_path, server_envelope) -> None:
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    receiver.accept_json(
        server_envelope(
            "server.tts.audio",
            payload={
                "chunk_id": "chunk-1",
                "segment_id": "segment-1",
                "format": "wav",
                "byte_length": 12,
            },
        )
    )
    receiver.accept_json(server_envelope("server.playback.stop", payload={}, generation_epoch=1))

    with pytest.raises(ProtocolError, match="binary frame arrived without server.tts.audio metadata"):
        receiver.accept_binary(b"RIFF....WAVE")

    assert receiver.pending_tts is None
    assert not (tmp_path / "tts" / "chunk-1.wav").exists()
    assert audio.queued_count == 0


def test_receiver_turn_cancelled_clears_pending_tts_binary(tmp_path, server_envelope) -> None:
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    receiver.accept_json(
        server_envelope(
            "server.tts.audio",
            payload={
                "chunk_id": "chunk-1",
                "segment_id": "segment-1",
                "format": "wav",
                "byte_length": 12,
            },
        )
    )
    receiver.accept_json(server_envelope("server.turn.cancelled", payload={}, generation_epoch=1))

    with pytest.raises(ProtocolError, match="binary frame arrived without server.tts.audio metadata"):
        receiver.accept_binary(b"RIFF....WAVE")

    assert receiver.pending_tts is None
    assert not (tmp_path / "tts" / "chunk-1.wav").exists()
    assert audio.queued_count == 0


def test_receiver_drops_stale_generation_audio_and_face(tmp_path, server_envelope) -> None:
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    receiver.accept_json(server_envelope("server.playback.stop", payload={}, generation_epoch=2))
    receiver.accept_json(
        server_envelope(
            "server.tts.audio",
            payload={
                "chunk_id": "stale-audio",
                "segment_id": "segment-1",
                "format": "wav",
                "byte_length": 12,
            },
            generation_epoch=1,
        )
    )
    receiver.accept_json(
        server_envelope(
            "server.ue5.frames",
            payload={
                "chunk_id": "stale-face",
                "segment_id": "segment-1",
                "frames": [{"frame_index": 0}],
            },
            generation_epoch=1,
        )
    )

    assert receiver.pending_tts is None
    assert audio.queued_count == 0
    assert face.buffered_chunk_count == 0
    assert receiver.summary["stale_drop_count"] == 2
    assert receiver.summary["stale_face_drop_count"] == 1
    assert receiver.summary["old_turn_face_leak_count"] == 1


def test_receiver_validates_ue5_frame_sequence(tmp_path, server_envelope) -> None:
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    receiver.accept_json(
        server_envelope(
            "server.ue5.frames",
            payload={
                "chunk_id": "segment-1-0000",
                "segment_id": "segment-1",
                "start_frame_index": 0,
                "frame_count": 1,
                "frames": [{"frame_index": 0}],
            },
        )
    )


@pytest.mark.parametrize(
    "stop_event",
    ["server.playback.stop", "server.turn.cancelled"],
)
def test_receiver_resets_ue5_frame_sequence_after_stop_or_cancel(
    tmp_path, server_envelope, stop_event
) -> None:
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    receiver.accept_json(
        server_envelope(
            "server.ue5.frames",
            payload={
                "chunk_id": "segment-1-0000",
                "segment_id": "segment-1",
                "start_frame_index": 0,
                "frame_count": 1,
                "frames": [{"frame_index": 0}],
            },
            generation_epoch=0,
        )
    )
    receiver.accept_json(server_envelope(stop_event, payload={}, generation_epoch=1))

    receiver.accept_json(
        server_envelope(
            "server.ue5.frames",
            payload={
                "chunk_id": "segment-1-0001",
                "segment_id": "segment-1",
                "start_frame_index": 0,
                "frame_count": 1,
                "frames": [{"frame_index": 0}],
            },
            generation_epoch=1,
        )
    )

    assert receiver.summary["ue5_chunks"] == 2


def test_receiver_requires_ue5_start_frame_index_and_frame_count(tmp_path, server_envelope) -> None:
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    with pytest.raises(
        ProtocolError, match="server.ue5.frames requires start_frame_index and frame_count"
    ):
        receiver.accept_json(
            server_envelope(
                "server.ue5.frames",
                payload={
                    "chunk_id": "segment-1-0000",
                    "segment_id": "segment-1",
                    "frames": [{"frame_index": 0}],
                },
            )
        )


def test_receiver_finish_writes_summary_and_terminal_event(tmp_path, server_envelope) -> None:
    metrics = PlaybackMetrics(clock=lambda: 0.0)
    audio = AudioPlaybackEngine(metrics, sink=MemoryAudioSink())
    face = FacePlaybackEngine(metrics)
    receiver = LocalDemoReceiver(tmp_path, audio, face)

    receiver.accept_json(
        server_envelope(
            "server.tts.audio",
            payload={
                "chunk_id": "chunk-1",
                "segment_id": "segment-1",
                "format": "wav",
                "byte_length": 12,
            },
        )
    )
    receiver.accept_binary(b"RIFF....WAVE")
    receiver.accept_json(
        server_envelope(
            "server.ue5.frames",
            payload={
                "chunk_id": "ue5-1",
                "segment_id": "segment-1",
                "start_frame_index": 0,
                "frame_count": 1,
                "frames": [{"frame_index": 0}],
            },
        )
    )
    receiver.accept_json(server_envelope("server.pipeline.done", payload={}))

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["tts_chunks"] == 1
    assert summary["ue5_chunks"] == 1
    assert summary["terminal_event"] == "server.pipeline.done"
    assert summary["client_audio_enqueued_count"] == 1
