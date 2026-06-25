from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import UUID
from typing import Protocol


TERMINAL_TYPES = {"server.pipeline.done", "server.pipeline.error", "server.turn.cancelled"}


class ProtocolError(RuntimeError):
    pass


class AudioSink(Protocol):
    def play(self, wav_bytes: bytes) -> None:
        ...

    def stop(self) -> None:
        ...


class PlaybackMetrics:
    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._started_at = clock()
        self.client_audio_enqueued_count = 0
        self.client_audio_play_start_ms: float | None = None
        self.client_audio_stopped_ms: float | None = None
        self.client_face_buffer_cleared_ms: float | None = None
        self.client_playback_stop_received_ms: float | None = None

    def mark_audio_enqueued(self, generation_epoch: int | None = None) -> None:
        if self.client_audio_play_start_ms is None:
            self.client_audio_play_start_ms = self._elapsed_ms()
        self.client_audio_enqueued_count += 1

    def mark_audio_stopped(self) -> None:
        if self.client_audio_stopped_ms is None:
            self.client_audio_stopped_ms = self._elapsed_ms()

    def mark_playback_stop_received(self) -> None:
        if self.client_playback_stop_received_ms is None:
            self.client_playback_stop_received_ms = self._elapsed_ms()

    def mark_face_buffer_cleared(self) -> None:
        if self.client_face_buffer_cleared_ms is None:
            self.client_face_buffer_cleared_ms = self._elapsed_ms()

    def to_dict(self) -> dict[str, object]:
        return {
            "client_audio_enqueued_count": self.client_audio_enqueued_count,
            "client_audio_play_start_ms": self.client_audio_play_start_ms,
            "client_audio_stopped_ms": self.client_audio_stopped_ms,
            "client_playback_stop_received_ms": self.client_playback_stop_received_ms,
            "client_face_buffer_cleared_ms": self.client_face_buffer_cleared_ms,
        }

    def _elapsed_ms(self) -> float:
        return round((self._clock() - self._started_at) * 1000.0, 3)


class MemoryAudioSink:
    def __init__(self) -> None:
        self.played_chunks: list[bytes] = []
        self.stopped_count = 0

    def play(self, wav_bytes: bytes) -> None:
        self.played_chunks.append(wav_bytes)

    def stop(self) -> None:
        self.stopped_count += 1


class AudioPlaybackEngine:
    def __init__(self, metrics: PlaybackMetrics, sink: AudioSink | None = None) -> None:
        self._metrics = metrics
        self._sink = sink or MemoryAudioSink()
        self._queued_chunks: dict[str, bytes] = {}

    @property
    def metrics(self) -> PlaybackMetrics:
        return self._metrics

    @property
    def queued_count(self) -> int:
        return len(self._queued_chunks)

    def enqueue_wav(self, chunk_id: str, wav_bytes: bytes, generation_epoch: int | None) -> None:
        self._queued_chunks[chunk_id] = wav_bytes
        self._metrics.mark_audio_enqueued(generation_epoch)
        self._sink.play(wav_bytes)

    def stop(self) -> None:
        self._sink.stop()
        self._metrics.mark_audio_stopped()

    def clear(self) -> None:
        self._queued_chunks.clear()


class FacePlaybackEngine:
    def __init__(self, metrics: PlaybackMetrics) -> None:
        self._metrics = metrics
        self._queued_chunks: dict[str, dict[str, object]] = {}

    @property
    def buffered_chunk_count(self) -> int:
        return len(self._queued_chunks)

    def enqueue_frames(
        self,
        chunk_id: str,
        payload: dict[str, object],
        generation_epoch: int | None,
    ) -> None:
        self._queued_chunks[chunk_id] = payload

    def clear(self) -> None:
        self._queued_chunks.clear()
        self._metrics.mark_face_buffer_cleared()


@dataclass
class PendingTTS:
    chunk_id: str
    segment_id: str
    byte_length: int
    format: str
    generation_epoch: int | None


class LocalDemoReceiver:
    def __init__(
        self,
        output_dir: Path,
        audio: AudioPlaybackEngine,
        face: FacePlaybackEngine,
        *,
        session_id: UUID | None = None,
        turn_id: UUID | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.audio = audio
        self.face = face
        self.session_id = session_id
        self.turn_id = turn_id
        self._clock = clock
        self._started_at = clock()
        self._metrics = audio.metrics
        self.next_sequence = 1
        self.pending_tts: PendingTTS | None = None
        self.next_ue5_frame_index_by_segment: dict[str, int] = {}
        self.terminal_event: str | None = None
        self.summary: dict[str, object] = {
            "events": 0,
            "tts_chunks": 0,
            "ue5_chunks": 0,
            "terminal_event": None,
            "terminal_event_ms": None,
            "event_counts": {},
            "event_first_ms": {},
            "latest_generation_epoch": None,
            "playback_stop_count": 0,
            "stale_drop_count": 0,
            "stale_face_drop_count": 0,
            "old_turn_face_leak_count": 0,
        }
        (self.output_dir / "tts").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "ue5").mkdir(parents=True, exist_ok=True)

    def accept_json(self, envelope: dict[str, object]) -> None:
        self._validate_envelope(envelope)
        received_ms = self._elapsed_ms()
        event_type = str(envelope["type"])
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise ProtocolError("server event payload must be an object")

        self.summary["events"] = int(self.summary["events"]) + 1
        self._record_event_count(event_type)
        self._record_first_event(event_type, received_ms)
        event_epoch = self._event_generation_epoch(envelope, payload)
        if self._is_stale_generation(event_epoch):
            self.summary["stale_drop_count"] = int(self.summary["stale_drop_count"]) + 1
            if event_type in {"server.face.frames", "server.ue5.frames"}:
                self.summary["stale_face_drop_count"] = (
                    int(self.summary["stale_face_drop_count"]) + 1
                )
                self.summary["old_turn_face_leak_count"] = (
                    int(self.summary["old_turn_face_leak_count"]) + 1
                )
            return

        self._record_generation_epoch(event_epoch)
        if event_type == "server.tts.audio":
            self._accept_tts_metadata(payload)
            return
        if event_type == "server.ue5.frames":
            self._accept_ue5_frames(payload, event_epoch)
            return
        if event_type == "server.playback.stop":
            self._metrics.mark_playback_stop_received()
            self._clear_pending_playback()
            self.summary["playback_stop_count"] = int(self.summary["playback_stop_count"]) + 1
            return
        if event_type == "server.turn.cancelled":
            self._metrics.mark_playback_stop_received()
            self._clear_pending_playback()
            self._mark_terminal(event_type, received_ms)
            return
        if event_type in TERMINAL_TYPES:
            self._mark_terminal(event_type, received_ms)

    def accept_binary(self, payload: bytes) -> None:
        pending = self.pending_tts
        if pending is None:
            raise ProtocolError("binary frame arrived without server.tts.audio metadata")
        if len(payload) != pending.byte_length:
            self.pending_tts = None
            raise ProtocolError("binary frame length does not match server.tts.audio metadata")
        if pending.format != "wav":
            self.pending_tts = None
            raise ProtocolError("only WAV TTS chunks are supported")

        (self.output_dir / "tts" / f"{pending.chunk_id}.wav").write_bytes(payload)
        self.audio.enqueue_wav(pending.chunk_id, payload, generation_epoch=pending.generation_epoch)
        self.summary["tts_chunks"] = int(self.summary["tts_chunks"]) + 1
        self.pending_tts = None

    def finish(self) -> None:
        summary = {**self.summary, **self._metrics.to_dict()}
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _validate_envelope(self, envelope: dict[str, object]) -> None:
        if envelope.get("protocol") != "bionic-head-stream-v1":
            raise ProtocolError("unexpected protocol")
        sequence = envelope.get("sequence")
        if sequence != self.next_sequence:
            raise ProtocolError(f"expected server sequence {self.next_sequence}, got {sequence}")
        self.next_sequence += 1

        session_id = UUID(str(envelope["session_id"]))
        turn_id = envelope.get("turn_id")
        parsed_turn_id = UUID(str(turn_id)) if turn_id is not None else None
        if self.session_id is None:
            self.session_id = session_id
        if session_id != self.session_id:
            raise ProtocolError("server event session_id does not match")
        if parsed_turn_id is not None:
            if self.turn_id is None:
                self.turn_id = parsed_turn_id
            elif parsed_turn_id != self.turn_id:
                raise ProtocolError("server event turn_id does not match")

    def _accept_tts_metadata(self, payload: dict[str, object]) -> None:
        if self.pending_tts is not None:
            raise ProtocolError("previous server.tts.audio is still waiting for binary")
        byte_length = payload.get("byte_length")
        if not isinstance(byte_length, int) or byte_length < 1:
            raise ProtocolError("server.tts.audio byte_length must be positive")
        self.pending_tts = PendingTTS(
            chunk_id=str(payload.get("chunk_id", "chunk")),
            segment_id=str(payload.get("segment_id", payload.get("chunk_id", "chunk"))),
            byte_length=byte_length,
            format=str(payload.get("format", "")),
            generation_epoch=self._payload_generation_epoch(payload),
        )

    def _accept_ue5_frames(self, payload: dict[str, object], generation_epoch: int | None) -> None:
        chunk_id = str(payload.get("chunk_id", f"ue5-{self.summary['ue5_chunks']}"))
        start = payload.get("start_frame_index")
        frame_count = payload.get("frame_count")
        if not isinstance(start, int) or not isinstance(frame_count, int):
            raise ProtocolError("server.ue5.frames requires start_frame_index and frame_count")
        segment_id = str(payload.get("segment_id") or self._segment_id_for_ue5_chunk(chunk_id))
        expected_start = self.next_ue5_frame_index_by_segment.get(segment_id, 0)
        if start != expected_start:
            raise ProtocolError("server.ue5.frames has a gap or overlap")
        (self.output_dir / "ue5" / f"{chunk_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.face.enqueue_frames(chunk_id, payload, generation_epoch=generation_epoch)
        self.next_ue5_frame_index_by_segment[segment_id] = start + frame_count
        self.summary["ue5_chunks"] = int(self.summary["ue5_chunks"]) + 1

    def _clear_pending_playback(self) -> None:
        self.pending_tts = None
        self.audio.stop()
        self.audio.clear()
        self.face.clear()
        self.next_ue5_frame_index_by_segment.clear()

    def _segment_id_for_ue5_chunk(self, chunk_id: str) -> str:
        prefix, separator, suffix = chunk_id.rpartition("-")
        if separator and suffix.isdigit():
            return prefix
        return chunk_id

    def _mark_terminal(self, event_type: str, received_ms: float) -> None:
        self.terminal_event = event_type
        self.summary["terminal_event"] = event_type
        self.summary["terminal_event_ms"] = received_ms
        self.finish()

    def _record_event_count(self, event_type: str) -> None:
        counts = self.summary["event_counts"]
        if isinstance(counts, dict):
            counts[event_type] = int(counts.get(event_type, 0)) + 1

    def _record_first_event(self, event_type: str, received_ms: float) -> None:
        first_events = self.summary["event_first_ms"]
        if isinstance(first_events, dict) and event_type not in first_events:
            first_events[event_type] = received_ms

    def _event_generation_epoch(
        self,
        envelope: dict[str, object],
        payload: dict[str, object],
    ) -> int | None:
        value = envelope.get("generation_epoch")
        if not isinstance(value, int):
            value = payload.get("generation_epoch")
        return value if isinstance(value, int) else None

    def _payload_generation_epoch(self, payload: dict[str, object]) -> int | None:
        value = payload.get("generation_epoch")
        return value if isinstance(value, int) else None

    def _is_stale_generation(self, event_epoch: int | None) -> bool:
        latest = self.summary["latest_generation_epoch"]
        return isinstance(event_epoch, int) and isinstance(latest, int) and event_epoch < latest

    def _record_generation_epoch(self, event_epoch: int | None) -> None:
        if event_epoch is None:
            return
        latest = self.summary["latest_generation_epoch"]
        if not isinstance(latest, int) or event_epoch > latest:
            self.summary["latest_generation_epoch"] = event_epoch

    def _elapsed_ms(self) -> float:
        return round((self._clock() - self._started_at) * 1000.0, 3)
