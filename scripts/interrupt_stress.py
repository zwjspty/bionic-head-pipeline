from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FACE_EVENT_TYPES = {"server.face.frames", "server.ue5.frames"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WebSocket interrupt stress against a running server")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8000/pipeline/stream")
    parser.add_argument("--wav", required=True, type=Path, help="Mono 16kHz PCM16 WAV used for every turn")
    parser.add_argument("--interrupts", type=int, default=50)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def summarize_events(events: list[dict[str, object]]) -> dict[str, object]:
    latest_generation_epoch: int | None = None
    old_turn_face_leak_count = 0
    playback_stop_count = 0
    event_counts: dict[str, int] = {}
    sequences: list[int] = []

    for event in events:
        event_type = str(event.get("type"))
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        sequence = event.get("sequence")
        if isinstance(sequence, int):
            sequences.append(sequence)
        generation_epoch = _event_generation_epoch(event)
        if generation_epoch is not None:
            if (
                latest_generation_epoch is not None
                and generation_epoch < latest_generation_epoch
                and event_type in FACE_EVENT_TYPES
            ):
                old_turn_face_leak_count += 1
            latest_generation_epoch = (
                generation_epoch
                if latest_generation_epoch is None
                else max(latest_generation_epoch, generation_epoch)
            )
        if event_type == "server.playback.stop":
            playback_stop_count += 1

    return {
        "events": len(events),
        "event_counts": event_counts,
        "playback_stop_count": playback_stop_count,
        "latest_generation_epoch": latest_generation_epoch,
        "old_turn_face_leak_count": old_turn_face_leak_count,
        "strictly_increasing_sequences": sequences == sorted(sequences)
        and len(sequences) == len(set(sequences)),
    }


async def run_interrupt_stress(
    *,
    ws_url: str,
    wav_path: Path,
    interrupts: int,
    chunk_ms: int,
) -> dict[str, object]:
    if interrupts < 1:
        raise ValueError("interrupts must be positive")
    chunks = pcm_chunks(read_pcm16_from_wav(wav_path), chunk_ms=chunk_ms)
    if not chunks:
        raise ValueError("wav produced no PCM chunks")

    import websockets

    session_id = uuid4()
    sequence = count(1)
    events: list[dict[str, object]] = []
    started = perf_counter()
    async with websockets.connect(ws_url, max_size=None) as ws:
        await _send_json(ws, client_event("client.session.start", session_id, None, next(sequence), {}))
        await _receive_until(ws, events, lambda event: event.get("type") == "server.session.ready")

        current_turn = uuid4()
        await _send_audio_turn(ws, session_id, current_turn, sequence, chunks, chunk_ms=chunk_ms)
        await _receive_until(
            ws,
            events,
            lambda event: event.get("type") == "server.tts.audio"
            and event.get("turn_id") == str(current_turn),
        )

        for _ in range(interrupts):
            next_turn = uuid4()
            await _send_json(
                ws,
                client_event("client.audio.start", session_id, next_turn, next(sequence), {}),
            )
            await _send_interrupt_speech(ws, session_id, next_turn, sequence, chunks, chunk_ms)
            await _receive_until(
                ws,
                events,
                lambda event: event.get("type") == "server.state"
                and event.get("turn_id") == str(next_turn),
            )
            current_turn = next_turn
            await _send_json(
                ws,
                client_event(
                    "client.audio.end",
                    session_id,
                    current_turn,
                    next(sequence),
                    {"reason": "client_end"},
                ),
            )
            await _receive_until(
                ws,
                events,
                lambda event: event.get("type") == "server.tts.audio"
                and event.get("turn_id") == str(current_turn),
            )

    summary = summarize_events(events)
    summary["interrupts_requested"] = interrupts
    summary["success"] = (
        summary["playback_stop_count"] == interrupts
        and summary["old_turn_face_leak_count"] == 0
        and summary["strictly_increasing_sequences"] is True
    )
    summary["wall_ms"] = round((perf_counter() - started) * 1000.0, 3)
    return summary


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


async def _send_audio_turn(
    ws,
    session_id: UUID,
    turn_id: UUID,
    sequence,
    chunks: list[bytes],
    *,
    chunk_ms: int,
) -> None:
    await _send_json(ws, client_event("client.audio.start", session_id, turn_id, next(sequence), {}))
    for chunk in chunks:
        await _send_audio_chunk(ws, session_id, turn_id, sequence, chunk, duration_ms=chunk_ms)
    await _send_json(
        ws,
        client_event(
            "client.audio.end",
            session_id,
            turn_id,
            next(sequence),
            {"reason": "client_end"},
        ),
    )


async def _send_interrupt_speech(
    ws,
    session_id: UUID,
    turn_id: UUID,
    sequence,
    chunks: list[bytes],
    chunk_ms: int,
) -> None:
    sent_ms = 0
    for chunk in chunks:
        await _send_audio_chunk(ws, session_id, turn_id, sequence, chunk, duration_ms=chunk_ms)
        sent_ms += chunk_ms
        if sent_ms >= 100:
            return


async def _send_audio_chunk(
    ws,
    session_id: UUID,
    turn_id: UUID,
    sequence,
    chunk: bytes,
    *,
    duration_ms: int,
) -> None:
    await _send_json(
        ws,
        client_event(
            "client.audio.chunk",
            session_id,
            turn_id,
            next(sequence),
            {"byte_length": len(chunk), "duration_ms": duration_ms},
        ),
    )
    await ws.send(chunk)


async def _send_json(ws, payload: dict[str, object]) -> None:
    await ws.send(json.dumps(payload, ensure_ascii=False))


async def _receive_until(ws, events: list[dict[str, object]], predicate) -> dict[str, object]:
    while True:
        message = await ws.recv()
        if not isinstance(message, str):
            raise RuntimeError("server sent binary without preceding server.tts.audio metadata")
        event = json.loads(message)
        if not isinstance(event, dict):
            raise RuntimeError("server sent non-object JSON event")
        events.append(event)
        if event.get("type") == "server.tts.audio":
            binary = await ws.recv()
            if not isinstance(binary, bytes):
                raise RuntimeError("server.tts.audio was not followed by binary audio")
        if predicate(event):
            return event


def _event_generation_epoch(event: dict[str, object]) -> int | None:
    value = event.get("generation_epoch")
    if not isinstance(value, int):
        payload = event.get("payload")
        if isinstance(payload, dict):
            value = payload.get("generation_epoch")
    return value if isinstance(value, int) else None


def main() -> None:
    args = build_parser().parse_args()
    summary = asyncio.run(
        run_interrupt_stress(
            ws_url=args.ws_url,
            wav_path=args.wav,
            interrupts=args.interrupts,
            chunk_ms=args.chunk_ms,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
