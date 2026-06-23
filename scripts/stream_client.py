from __future__ import annotations

import argparse
import asyncio
import json
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import UUID, uuid4


TERMINAL_TYPES = {"server.pipeline.done", "server.pipeline.error", "server.turn.cancelled"}


class ProtocolError(RuntimeError):
    pass


@dataclass
class PendingTTS:
    chunk_id: str
    byte_length: int
    format: str


class ClientReceiver:
    def __init__(
        self,
        output_dir: Path,
        *,
        session_id: UUID | None = None,
        turn_id: UUID | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.session_id = session_id
        self.turn_id = turn_id
        self._clock = clock
        self._started = clock()
        self.next_sequence = 1
        self.pending_tts: PendingTTS | None = None
        self.pending_segments: dict[str, object] = {}
        self.pending_ue5_chunks: dict[str, object] = {}
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
            "first_tts_binary_ms": None,
            "latest_generation_epoch": None,
            "playback_stop_count": 0,
            "stale_drop_count": 0,
        }
        (self.output_dir / "tts").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "ue5").mkdir(parents=True, exist_ok=True)

    def accept_json(self, envelope: dict[str, object]) -> None:
        self._validate_envelope(envelope)
        received_ms = self._elapsed_ms()
        self._append_event(envelope, received_ms=received_ms)
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
            return
        self._record_generation_epoch(event_epoch)
        if event_type == "server.tts.audio":
            self._accept_tts_metadata(payload)
        elif event_type == "server.ue5.frames":
            self._accept_ue5_frames(payload)
        elif event_type == "server.segment.ready":
            chunk_id = str(payload.get("chunk_id", f"segment-{len(self.pending_segments)}"))
            self.pending_segments[chunk_id] = payload
        elif event_type == "server.playback.stop":
            self._clear_pending_playback()
            self.summary["playback_stop_count"] = int(self.summary["playback_stop_count"]) + 1
        elif event_type == "server.turn.cancelled":
            self._clear_pending_playback()
            self._mark_terminal(event_type, received_ms)
        elif event_type in TERMINAL_TYPES:
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
        path = self.output_dir / "tts" / f"{pending.chunk_id}.wav"
        path.write_bytes(payload)
        self.summary["tts_chunks"] = int(self.summary["tts_chunks"]) + 1
        if self.summary["first_tts_binary_ms"] is None:
            self.summary["first_tts_binary_ms"] = self._elapsed_ms()
        self.pending_tts = None

    def finish(self) -> None:
        (self.output_dir / "summary.json").write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2),
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
        chunk_id = str(payload.get("chunk_id", "chunk"))
        byte_length = payload.get("byte_length")
        if not isinstance(byte_length, int) or byte_length < 1:
            raise ProtocolError("server.tts.audio byte_length must be positive")
        self.pending_tts = PendingTTS(
            chunk_id=chunk_id,
            byte_length=byte_length,
            format=str(payload.get("format", "")),
        )

    def _accept_ue5_frames(self, payload: dict[str, object]) -> None:
        chunk_id = str(payload.get("chunk_id", f"ue5-{self.summary['ue5_chunks']}"))
        start = payload.get("start_frame_index")
        frame_count = payload.get("frame_count")
        if not isinstance(start, int) or not isinstance(frame_count, int):
            raise ProtocolError("server.ue5.frames requires start_frame_index and frame_count")
        segment_id = self._segment_id_for_ue5_chunk(chunk_id)
        expected_start = self.next_ue5_frame_index_by_segment.get(segment_id, 0)
        if start != expected_start:
            raise ProtocolError("server.ue5.frames has a gap or overlap")
        path = self.output_dir / "ue5" / f"{chunk_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.pending_ue5_chunks[chunk_id] = payload
        self.next_ue5_frame_index_by_segment[segment_id] = start + frame_count
        self.summary["ue5_chunks"] = int(self.summary["ue5_chunks"]) + 1

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
        if not isinstance(first_events, dict):
            return
        if event_type not in first_events:
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

    def _is_stale_generation(self, event_epoch: int | None) -> bool:
        latest = self.summary["latest_generation_epoch"]
        return (
            isinstance(event_epoch, int)
            and isinstance(latest, int)
            and event_epoch < latest
        )

    def _record_generation_epoch(self, event_epoch: int | None) -> None:
        if event_epoch is None:
            return
        latest = self.summary["latest_generation_epoch"]
        if not isinstance(latest, int) or event_epoch > latest:
            self.summary["latest_generation_epoch"] = event_epoch

    def _clear_pending_playback(self) -> None:
        self.pending_tts = None
        self.pending_segments.clear()
        self.pending_ue5_chunks.clear()
        self.next_ue5_frame_index_by_segment.clear()

    def _elapsed_ms(self) -> float:
        return round((self._clock() - self._started) * 1000.0, 3)

    def _append_event(self, envelope: dict[str, object], *, received_ms: float) -> None:
        record = {**envelope, "_client_received_ms": received_ms}
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_pcm16_from_wav(path: Path) -> bytes:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()
            frames = wav.readframes(frame_count)
    except (wave.Error, OSError, EOFError) as exc:
        raise SystemExit(f"Invalid WAV input: {path}") from exc
    if channels != 1 or sample_width != 2 or sample_rate != 16000:
        raise SystemExit("Input WAV must be mono PCM16 at 16000 Hz")
    return frames


def pcm_chunks(pcm: bytes, *, chunk_ms: int) -> list[bytes]:
    if not 20 <= chunk_ms <= 100:
        raise SystemExit("--chunk-ms must be between 20 and 100")
    bytes_per_ms = 16000 * 2 // 1000
    chunk_size = bytes_per_ms * chunk_ms
    minimum_size = bytes_per_ms * 20
    chunks: list[bytes] = []
    for index in range(0, len(pcm), chunk_size):
        chunk = pcm[index : index + chunk_size]
        if len(chunk) < minimum_size:
            chunk = chunk + b"\x00" * (minimum_size - len(chunk))
        chunks.append(chunk)
    return chunks


def client_event(
    event_type: str,
    *,
    session_id: UUID,
    turn_id: UUID | None,
    sequence: int,
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "protocol": "bionic-head-stream-v1",
        "type": event_type,
        "event_id": str(uuid4()),
        "session_id": str(session_id),
        "turn_id": str(turn_id) if turn_id is not None else None,
        "sequence": sequence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }


async def run_client(url: str, wav_path: Path, output_dir: Path, chunk_ms: int) -> str:
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit("websockets is required; install the client extra") from exc

    pcm = read_pcm16_from_wav(wav_path)
    session_id = uuid4()
    turn_id = uuid4()
    receiver = ClientReceiver(output_dir, session_id=session_id, turn_id=turn_id)
    sequence = 1

    async with websockets.connect(url) as websocket:
        await websocket.send(
            json.dumps(
                client_event(
                    "client.session.start",
                    session_id=session_id,
                    turn_id=None,
                    sequence=sequence,
                    payload={"client_name": "stream_client"},
                )
            )
        )
        sequence += 1
        first = await websocket.recv()
        if isinstance(first, bytes):
            raise ProtocolError("expected server.session.ready JSON")
        receiver.accept_json(json.loads(first))

        await websocket.send(
            json.dumps(
                client_event(
                    "client.audio.start",
                    session_id=session_id,
                    turn_id=turn_id,
                    sequence=sequence,
                    payload={"sample_rate": 16000, "channels": 1, "sample_width_bytes": 2},
                )
            )
        )
        sequence += 1
        for chunk in pcm_chunks(pcm, chunk_ms=chunk_ms):
            await websocket.send(
                json.dumps(
                    client_event(
                        "client.audio.chunk",
                        session_id=session_id,
                        turn_id=turn_id,
                        sequence=sequence,
                        payload={
                            "byte_length": len(chunk),
                            "duration_ms": int(len(chunk) / 2 / 16000 * 1000),
                        },
                    )
                )
            )
            sequence += 1
            await websocket.send(chunk)
        await websocket.send(
            json.dumps(
                client_event(
                    "client.audio.end",
                    session_id=session_id,
                    turn_id=turn_id,
                    sequence=sequence,
                    payload={"reason": "client_end"},
                )
            )
        )

        while receiver.terminal_event is None:
            message = await websocket.recv()
            if isinstance(message, bytes):
                receiver.accept_binary(message)
            else:
                receiver.accept_json(json.loads(message))

    receiver.finish()
    return str(receiver.terminal_event)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate bionic-head streaming protocol")
    parser.add_argument("--url", required=True, help="WebSocket URL, e.g. ws://127.0.0.1:8000/pipeline/stream")
    parser.add_argument("--wav", required=True, type=Path, help="Input mono PCM16 16 kHz WAV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for received events/audio/frames")
    parser.add_argument("--chunk-ms", type=int, default=40, help="Client PCM chunk duration, 20-100 ms")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    terminal = asyncio.run(run_client(args.url, args.wav, args.output_dir, args.chunk_ms))
    print(f"terminal_event={terminal}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
