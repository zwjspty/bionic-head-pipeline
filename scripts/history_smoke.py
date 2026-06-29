from __future__ import annotations

import argparse
import asyncio
import json
import sys
from array import array
from pathlib import Path
from uuid import UUID, uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bionic_head.client.history_smoke import (
    HistorySmokeReport,
    HistorySmokeTurn,
    build_history_smoke_report,
    write_history_smoke_report,
)
from scripts.stream_client import client_event, pcm_chunks, read_pcm16_from_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a two-turn stream history smoke.")
    parser.add_argument("--url", required=True, help="WebSocket URL, e.g. ws://127.0.0.1:8005/pipeline/stream")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["mock", "real"], default="real")
    parser.add_argument("--turn1-wav", type=Path)
    parser.add_argument("--turn2-wav", type=Path)
    parser.add_argument("--expect", default="小张")
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    return parser


async def run_history_smoke(
    *,
    url: str,
    output_dir: Path,
    mode: str,
    turn1_wav: Path | None,
    turn2_wav: Path | None,
    expected_text: str,
    chunk_ms: int,
    timeout_sec: float,
) -> HistorySmokeReport:
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit("websockets is required; install the client extra") from exc

    if mode == "real" and (turn1_wav is None or turn2_wav is None):
        raise SystemExit("--turn1-wav and --turn2-wav are required for real history smoke")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tts").mkdir(exist_ok=True)
    (output_dir / "ue5").mkdir(exist_ok=True)
    events_path = output_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()

    session_id = uuid4()
    turn_ids = [uuid4(), uuid4()]
    sequence = 1
    turns: list[HistorySmokeTurn] = []
    summary: dict[str, object] = {
        "mode": mode,
        "session_id": str(session_id),
        "events": 0,
        "tts_chunks": 0,
        "ue5_chunks": 0,
    }

    async def send_json(websocket, event_type: str, turn_id: UUID | None, payload: dict[str, object]) -> None:
        nonlocal sequence
        current_sequence = sequence
        sequence += 1
        await websocket.send(
            json.dumps(
                client_event(
                    event_type,
                    session_id=session_id,
                    turn_id=turn_id,
                    sequence=current_sequence,
                    payload=payload,
                )
            )
        )

    async def send_pcm_turn(websocket, turn_id: UUID, wav_path: Path | None) -> None:
        await send_json(
            websocket,
            "client.audio.start",
            turn_id,
            {"sample_rate": 16000, "channels": 1, "sample_width_bytes": 2},
        )
        for chunk in pcm_chunks(_pcm_for_turn(wav_path), chunk_ms=chunk_ms):
            await send_json(
                websocket,
                "client.audio.chunk",
                turn_id,
                {"byte_length": len(chunk), "duration_ms": int(len(chunk) / 2 / 16000 * 1000)},
            )
            await websocket.send(chunk)
        await send_json(websocket, "client.audio.end", turn_id, {"reason": "client_end"})

    async def wait_for_turn_done(websocket, *, turn_index: int, turn_id: UUID) -> HistorySmokeTurn:
        asr_text: str | None = None
        reply_parts: list[str] = []
        history_enabled: bool | None = None
        history_before: int | None = None
        history_after: int | None = None
        terminal_event: str | None = None
        pending_tts: dict[str, object] | None = None

        while terminal_event is None:
            message = await asyncio.wait_for(websocket.recv(), timeout=timeout_sec)
            if isinstance(message, bytes):
                if pending_tts is not None:
                    chunk_id = str(pending_tts.get("chunk_id", f"turn{turn_index}-tts"))
                    (output_dir / "tts" / f"{chunk_id}.wav").write_bytes(message)
                    summary["tts_chunks"] = int(summary["tts_chunks"]) + 1
                    pending_tts = None
                continue

            envelope = json.loads(message)
            _append_event(events_path, envelope)
            summary["events"] = int(summary["events"]) + 1
            event_type = str(envelope.get("type"))
            payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}

            if event_type == "server.tts.audio" and str(envelope.get("turn_id")) == str(turn_id):
                pending_tts = payload
                continue
            if event_type == "server.ue5.frames" and str(envelope.get("turn_id")) == str(turn_id):
                chunk_id = str(payload.get("chunk_id", f"turn{turn_index}-ue5-{summary['ue5_chunks']}"))
                (output_dir / "ue5" / f"{chunk_id}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                summary["ue5_chunks"] = int(summary["ue5_chunks"]) + 1

            if str(envelope.get("turn_id")) != str(turn_id):
                continue
            if event_type == "server.asr.final":
                text = payload.get("text")
                asr_text = str(text) if text is not None else None
            elif event_type == "server.llm.token":
                text = payload.get("text")
                if text is not None:
                    reply_parts.append(str(text))
            elif event_type in {"server.pipeline.done", "server.pipeline.error", "server.turn.cancelled"}:
                terminal_event = event_type
                history_enabled = _bool_or_none(payload.get("history_enabled"))
                history_before = _int_or_none(payload.get("history_turn_count_before"))
                history_after = _int_or_none(payload.get("history_turn_count_after"))

        return HistorySmokeTurn(
            turn_index=turn_index,
            turn_id=str(turn_id),
            terminal_event=terminal_event,
            asr_text=asr_text,
            llm_reply="".join(reply_parts) or None,
            history_enabled=history_enabled,
            history_turn_count_before=history_before,
            history_turn_count_after=history_after,
        )

    async with websockets.connect(url) as websocket:
        await send_json(websocket, "client.session.start", None, {"client_name": "history_smoke"})
        first = await asyncio.wait_for(websocket.recv(), timeout=timeout_sec)
        if isinstance(first, bytes):
            raise RuntimeError("expected server.session.ready JSON")
        _append_event(events_path, json.loads(first))
        summary["events"] = int(summary["events"]) + 1

        await send_pcm_turn(websocket, turn_ids[0], turn1_wav)
        turns.append(await wait_for_turn_done(websocket, turn_index=1, turn_id=turn_ids[0]))
        await send_pcm_turn(websocket, turn_ids[1], turn2_wav)
        turns.append(await wait_for_turn_done(websocket, turn_index=2, turn_id=turn_ids[1]))

    report = build_history_smoke_report(
        mode=mode,
        session_id=str(session_id),
        expected_text=expected_text,
        turns=turns,
    )
    summary["success"] = report.success
    summary["failure_reasons"] = report.failure_reasons
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_history_smoke_report(output_dir / "history_smoke_report.json", report)
    return report


def _pcm_for_turn(wav_path: Path | None) -> bytes:
    if wav_path is not None:
        return read_pcm16_from_wav(wav_path)
    samples = array("h", [2000, -2000] * 800)
    return samples.tobytes()


def _append_event(path: Path, envelope: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope, ensure_ascii=False) + "\n")


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "real" and (args.turn1_wav is None or args.turn2_wav is None):
        raise SystemExit("real mode requires --turn1-wav and --turn2-wav")

    report = asyncio.run(
        run_history_smoke(
            url=args.url,
            output_dir=args.output_dir,
            mode=args.mode,
            turn1_wav=args.turn1_wav,
            turn2_wav=args.turn2_wav,
            expected_text=args.expect,
            chunk_ms=args.chunk_ms,
            timeout_sec=args.timeout_sec,
        )
    )
    write_history_smoke_report(args.output_dir / "history_smoke_report.json", report)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if not report.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
