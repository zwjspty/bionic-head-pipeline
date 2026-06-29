from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import UUID, uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.local_demo_client import (  # noqa: E402
    AudioPlaybackEngine,
    AudioSink,
    FacePlaybackEngine,
    LocalDemoReceiver,
    MemoryAudioSink,
    PlaybackMetrics,
    ProtocolError,
    SoundDeviceAudioSink,
)
from scripts.stream_client import client_event  # noqa: E402
from bionic_head.client.scripted import build_interaction_report  # noqa: E402


class MicrophoneInput(Protocol):
    async def start(self) -> None:
        ...

    async def read_chunk(self) -> bytes:
        ...

    async def stop(self) -> None:
        ...

    async def close(self) -> None:
        ...


class CommandSource(Protocol):
    async def read_command(self) -> str:
        ...


class StdinCommandSource:
    def __init__(self, prompt: str = "[Enter=start/stop, c=cancel, q=quit] ") -> None:
        self._prompt = prompt

    async def read_command(self) -> str:
        return await asyncio.to_thread(input, self._prompt)


class FakeMicBackend:
    def __init__(
        self,
        *,
        sample_rate: int,
        chunk_ms: int,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self._chunks = list(chunks) if chunks is not None else None
        self._active = False
        self._sample_offset = 0

    async def start(self) -> None:
        self._active = True

    async def read_chunk(self) -> bytes:
        if not self._active:
            return b""
        if self._chunks is not None:
            if not self._chunks:
                return b""
            return self._chunks.pop(0)
        await asyncio.sleep(self.chunk_ms / 1000.0)
        if not self._active:
            return b""
        sample_count = chunk_samples_for_ms(self.sample_rate, self.chunk_ms)
        samples = bytearray()
        for index in range(sample_count):
            sample_index = self._sample_offset + index
            value = int(2500 * math.sin(2 * math.pi * 220 * sample_index / self.sample_rate))
            samples.extend(value.to_bytes(2, byteorder="little", signed=True))
        self._sample_offset += sample_count
        return bytes(samples)

    async def stop(self) -> None:
        self._active = False

    async def close(self) -> None:
        await self.stop()


class SoundDeviceMicrophoneInput:
    def __init__(self, *, sample_rate: int, chunk_ms: int, channels: int = 1) -> None:
        try:
            import sounddevice
        except ImportError as exc:
            raise SystemExit("sounddevice is required for microphone input; install the client-audio extra") from exc

        self._sounddevice = sounddevice
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.channels = channels
        self._stream = None
        self._queue: asyncio.Queue[bytes] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        if self._stream is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()

        def callback(indata, frames, time, status) -> None:  # noqa: ANN001
            if self._queue is None or self._loop is None:
                return
            data = indata.copy().tobytes() if hasattr(indata, "copy") else bytes(indata)
            self._loop.call_soon_threadsafe(self._queue.put_nowait, data)

        self._stream = self._sounddevice.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=chunk_samples_for_ms(self.sample_rate, self.chunk_ms),
            callback=callback,
        )
        self._stream.start()

    async def read_chunk(self) -> bytes:
        if self._queue is None:
            raise RuntimeError("microphone has not been started")
        return await self._queue.get()

    async def stop(self) -> None:
        if self._stream is None:
            return
        stream = self._stream
        self._stream = None
        stream.stop()
        stream.close()
        self._queue = None
        self._loop = None

    async def close(self) -> None:
        await self.stop()


@dataclass
class MicMetrics:
    recording_started_count: int = 0
    recording_stopped_count: int = 0
    chunks_sent: int = 0
    bytes_sent: int = 0
    manual_cancel_count: int = 0

    def to_summary(self) -> dict[str, int]:
        return {
            "client_mic_recording_started_count": self.recording_started_count,
            "client_mic_recording_stopped_count": self.recording_stopped_count,
            "client_mic_chunks_sent": self.chunks_sent,
            "client_mic_bytes_sent": self.bytes_sent,
            "client_manual_cancel_count": self.manual_cancel_count,
        }


def chunk_samples_for_ms(sample_rate: int, chunk_ms: int) -> int:
    return int(sample_rate * chunk_ms / 1000)


def create_microphone_backend(backend: str, *, sample_rate: int, chunk_ms: int) -> MicrophoneInput:
    if backend == "fake":
        return FakeMicBackend(sample_rate=sample_rate, chunk_ms=chunk_ms)
    if backend == "sounddevice":
        return SoundDeviceMicrophoneInput(sample_rate=sample_rate, chunk_ms=chunk_ms)
    raise SystemExit(f"unsupported microphone backend: {backend}")


def create_audio_sink(backend: str) -> AudioSink:
    if backend == "null":
        return MemoryAudioSink()
    if backend == "sounddevice":
        return SoundDeviceAudioSink()
    raise SystemExit(f"unsupported audio backend: {backend}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the interactive microphone demo client.")
    parser.add_argument("--url", required=True, help="WebSocket URL, e.g. ws://127.0.0.1:8005/pipeline/stream")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for received audio/frames/summary")
    parser.add_argument("--mode", choices=["interactive", "scripted"], default="interactive")
    parser.add_argument("--scripted-turns", type=int, default=2)
    parser.add_argument("--scripted-cancel-after-ms", type=int, default=300)
    parser.add_argument("--chunk-ms", type=int, default=40, help="Microphone PCM chunk duration in milliseconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate; default is 16 kHz")
    parser.add_argument(
        "--playback-sync",
        choices=["immediate_audio", "wait_for_face"],
        default="immediate_audio",
        help="Client-side audio/face synchronization strategy.",
    )
    parser.add_argument(
        "--wait-for-face-timeout-ms",
        type=int,
        default=800,
        help="Fallback delay before playing audio when wait_for_face has not received matching UE5 frames.",
    )
    parser.add_argument(
        "--mic-backend",
        choices=["sounddevice", "fake"],
        default=None,
        help="Microphone backend. Use fake for protocol smoke tests without a real microphone.",
    )
    parser.add_argument(
        "--audio-backend",
        choices=["sounddevice", "null"],
        help="Audio playback backend. Defaults to sounddevice unless --no-play-audio is used.",
    )
    parser.add_argument(
        "--play-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Play received TTS audio locally; use --no-play-audio for metrics-only smoke tests",
    )
    return parser


class InteractiveDemoSession:
    def __init__(
        self,
        *,
        websocket,
        receiver: LocalDemoReceiver,
        microphone: MicrophoneInput,
        command_source: CommandSource,
        session_id: UUID,
        turn_id: UUID,
        sample_rate: int,
        sequence: int = 2,
    ) -> None:
        self.websocket = websocket
        self.receiver = receiver
        self.microphone = microphone
        self.command_source = command_source
        self.session_id = session_id
        self.turn_id = turn_id
        self.sample_rate = sample_rate
        self.sequence = sequence
        self.mic_metrics = MicMetrics()
        self._send_lock = asyncio.Lock()
        self._recording = False
        self._recording_task: asyncio.Task[None] | None = None
        self._receiver_done = asyncio.Event()
        self._receiver_error: BaseException | None = None

    async def run(self) -> str:
        receiver_task = asyncio.create_task(self._receive_loop())
        try:
            while self.receiver.terminal_event is None:
                command = (await self.command_source.read_command()).strip().lower()
                if command == "":
                    if self._recording:
                        await self.stop_recording(reason="user_toggle_stop")
                    else:
                        await self.start_recording()
                elif command == "c":
                    await self.send_cancel()
                elif command == "q":
                    if self._recording:
                        await self.stop_recording(reason="client_quit")
                    break
                else:
                    print("Commands: Enter=start/stop recording, c=cancel playback, q=quit")
                await asyncio.sleep(0)

            if self.receiver.terminal_event is None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._receiver_done.wait(), timeout=0.25)
            if self._receiver_error is not None:
                raise self._receiver_error
        finally:
            if self._recording:
                await self.stop_recording(reason="cleanup")
            await self.microphone.close()
            if not receiver_task.done():
                receiver_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver_task
            self._append_mic_metrics()
            self.receiver.finish()
        return str(self.receiver.terminal_event)

    async def start_recording(self) -> None:
        if self._recording:
            return
        await self.microphone.start()
        self._recording = True
        self.mic_metrics.recording_started_count += 1
        await self._send_event(
            "client.audio.start",
            payload={
                "sample_rate": self.sample_rate,
                "channels": 1,
                "sample_width_bytes": 2,
            },
        )
        self._recording_task = asyncio.create_task(self._recording_loop())

    async def stop_recording(self, *, reason: str) -> None:
        if not self._recording:
            return
        self._recording = False
        await self.microphone.stop()
        if self._recording_task is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._recording_task, timeout=1.0)
            if not self._recording_task.done():
                self._recording_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._recording_task
        self._recording_task = None
        self.mic_metrics.recording_stopped_count += 1
        await self._send_event("client.audio.end", payload={"reason": reason})

    async def send_cancel(self) -> None:
        self.mic_metrics.manual_cancel_count += 1
        self.receiver.audio.metrics.mark_client_interrupt_sent()
        await self._send_event("client.turn.cancel", payload={"reason": "manual_keyboard_interrupt"})

    async def _recording_loop(self) -> None:
        while self._recording:
            chunk = await self.microphone.read_chunk()
            if not chunk:
                break
            async with self._send_lock:
                current_sequence = self.sequence
                self.sequence += 1
                await self.websocket.send(
                    json.dumps(
                        client_event(
                            "client.audio.chunk",
                            session_id=self.session_id,
                            turn_id=self.turn_id,
                            sequence=current_sequence,
                            payload={
                                "byte_length": len(chunk),
                                "duration_ms": int(len(chunk) / 2 / self.sample_rate * 1000),
                            },
                        )
                    )
                )
                await self.websocket.send(chunk)
            self.mic_metrics.chunks_sent += 1
            self.mic_metrics.bytes_sent += len(chunk)

    async def _receive_loop(self) -> None:
        try:
            while self.receiver.terminal_event is None:
                message = await self.websocket.recv()
                if isinstance(message, bytes):
                    self.receiver.accept_binary(message)
                else:
                    self.receiver.accept_json(json.loads(message))
                self.receiver.flush_sync_timeouts()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001
            self._receiver_error = exc
        finally:
            self._receiver_done.set()

    async def _send_event(self, event_type: str, *, payload: dict[str, object]) -> None:
        async with self._send_lock:
            current_sequence = self.sequence
            self.sequence += 1
            await self.websocket.send(
                json.dumps(
                    client_event(
                        event_type,
                        session_id=self.session_id,
                        turn_id=self.turn_id,
                        sequence=current_sequence,
                        payload=payload,
                    )
                )
            )

    def _append_mic_metrics(self) -> None:
        self.receiver.summary.update(self.mic_metrics.to_summary())


async def run_interactive_demo(
    *,
    url: str,
    output_dir: Path,
    command_source: CommandSource,
    microphone: MicrophoneInput,
    play_audio: bool | None = None,
    audio_backend: str | None = None,
    chunk_ms: int,
    sample_rate: int,
    clock: Callable[[], float] = perf_counter,
    playback_sync: str = "immediate_audio",
    wait_for_face_timeout_ms: int = 800,
) -> str:
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit("websockets is required; install the client extra") from exc

    session_id = uuid4()
    turn_id = uuid4()
    resolved_audio_backend = audio_backend or ("sounddevice" if play_audio else "null")
    audio_sink = create_audio_sink(resolved_audio_backend)
    metrics = PlaybackMetrics(clock=clock)
    receiver = LocalDemoReceiver(
        output_dir,
        AudioPlaybackEngine(metrics, sink=audio_sink),
        FacePlaybackEngine(metrics),
        session_id=session_id,
        turn_id=turn_id,
        clock=clock,
        playback_sync=playback_sync,
        wait_for_face_timeout_ms=wait_for_face_timeout_ms,
    )
    sequence = 1

    async with websockets.connect(url) as websocket:
        await websocket.send(
            json.dumps(
                client_event(
                    "client.session.start",
                    session_id=session_id,
                    turn_id=None,
                    sequence=sequence,
                    payload={"client_name": "interactive_demo_client"},
                )
            )
        )
        sequence += 1

        first = await websocket.recv()
        if isinstance(first, bytes):
            raise ProtocolError("expected server.session.ready JSON")
        first_event = json.loads(first)
        if first_event.get("type") != "server.session.ready":
            raise ProtocolError("expected first server event to be server.session.ready")
        receiver.accept_json(first_event)

        session = InteractiveDemoSession(
            websocket=websocket,
            receiver=receiver,
            microphone=microphone,
            command_source=command_source,
            session_id=session_id,
            turn_id=turn_id,
            sample_rate=sample_rate,
            sequence=sequence,
        )
        return await session.run()


async def run_scripted_demo(
    *,
    url: str,
    output_dir: Path,
    scripted_turns: int,
    scripted_cancel_after_ms: int,
    chunk_ms: int,
    sample_rate: int,
    audio_backend: str,
    clock: Callable[[], float] = perf_counter,
    wait_timeout_sec: float = 5.0,
    playback_sync: str = "immediate_audio",
    wait_for_face_timeout_ms: int = 800,
) -> str:
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit("websockets is required; install the client extra") from exc
    if scripted_turns < 1:
        raise SystemExit("--scripted-turns must be at least 1")

    session_id = uuid4()
    turn_ids = [uuid4() for _ in range(scripted_turns)]
    first_playback_started = asyncio.Event()
    metrics = PlaybackMetrics(clock=clock)
    audio = AudioPlaybackEngine(
        metrics,
        sink=create_audio_sink(audio_backend),
        on_first_play=first_playback_started.set,
    )
    receiver = LocalDemoReceiver(
        output_dir,
        audio,
        FacePlaybackEngine(metrics),
        session_id=session_id,
        turn_id=turn_ids[0],
        clock=clock,
        terminal_types={"server.pipeline.done", "server.pipeline.error"},
        allow_turn_switch=True,
        playback_sync=playback_sync,
        wait_for_face_timeout_ms=wait_for_face_timeout_ms,
    )
    mic_metrics = MicMetrics()
    sequence = 1
    receiver_error: BaseException | None = None

    async def wait_until(predicate: Callable[[], bool], *, description: str) -> None:
        started = perf_counter()
        while not predicate():
            if perf_counter() - started > wait_timeout_sec:
                raise TimeoutError(f"timed out waiting for {description}")
            await asyncio.sleep(0)

    async def send_event(websocket, event_type: str, turn_id: UUID | None, payload: dict[str, object]) -> None:
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

    async def receive_loop(websocket) -> None:
        nonlocal receiver_error
        try:
            while receiver.terminal_event is None:
                message = await websocket.recv()
                if isinstance(message, bytes):
                    receiver.accept_binary(message)
                else:
                    receiver.accept_json(json.loads(message))
                receiver.flush_sync_timeouts()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001
            receiver_error = exc

    async def send_fake_turn(websocket, turn_id: UUID, *, reason: str) -> None:
        await send_event(
            websocket,
            "client.audio.start",
            turn_id,
            {"sample_rate": sample_rate, "channels": 1, "sample_width_bytes": 2},
        )
        mic = FakeMicBackend(
            sample_rate=sample_rate,
            chunk_ms=chunk_ms,
        )
        await mic.start()
        mic_metrics.recording_started_count += 1
        chunk_count = max(1, int(round(1000 / chunk_ms)))
        for _ in range(chunk_count):
            chunk = await mic.read_chunk()
            if not chunk:
                break
            await send_event(
                websocket,
                "client.audio.chunk",
                turn_id,
                {
                    "byte_length": len(chunk),
                    "duration_ms": int(len(chunk) / 2 / sample_rate * 1000),
                },
            )
            await websocket.send(chunk)
            mic_metrics.chunks_sent += 1
            mic_metrics.bytes_sent += len(chunk)
        await mic.stop()
        await mic.close()
        mic_metrics.recording_stopped_count += 1
        await send_event(websocket, "client.audio.end", turn_id, {"reason": reason})

    async with websockets.connect(url) as websocket:
        await send_event(
            websocket,
            "client.session.start",
            None,
            {"client_name": "interactive_demo_client"},
        )
        first = await websocket.recv()
        if isinstance(first, bytes):
            raise ProtocolError("expected server.session.ready JSON")
        first_event = json.loads(first)
        if first_event.get("type") != "server.session.ready":
            raise ProtocolError("expected first server event to be server.session.ready")
        receiver.accept_json(first_event)

        receive_task = asyncio.create_task(receive_loop(websocket))
        try:
            for index, turn_id in enumerate(turn_ids):
                await send_fake_turn(websocket, turn_id, reason="client_end")
                if index == 0 and scripted_turns > 1:
                    await asyncio.wait_for(first_playback_started.wait(), timeout=wait_timeout_sec)
                    if scripted_cancel_after_ms > 0:
                        await asyncio.sleep(scripted_cancel_after_ms / 1000.0)
                    metrics.mark_client_interrupt_sent()
                    mic_metrics.manual_cancel_count += 1
                    await send_event(
                        websocket,
                        "client.turn.cancel",
                        turn_id,
                        {"reason": "scripted_playback_interrupt"},
                    )
                    await wait_until(
                        lambda: _event_count(receiver, "server.turn.cancelled") >= 1,
                        description="server.turn.cancelled",
                    )
                elif index == scripted_turns - 1:
                    await wait_until(
                        lambda: receiver.terminal_event is not None,
                        description="terminal event",
                    )
            if receiver_error is not None:
                raise receiver_error
        finally:
            if not receive_task.done():
                receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receive_task

    receiver.summary.update(mic_metrics.to_summary())
    receiver.finish()
    report_summary = {
        **receiver.summary,
        **metrics.to_dict(),
        **receiver.sync_clock.metrics(),
    }
    report = build_interaction_report(
        report_summary,
        mode="scripted",
        turn_count=scripted_turns,
        completed_turn_count=1 if receiver.terminal_event == "server.pipeline.done" else 0,
        cancelled_turn_count=1 if _event_count(receiver, "server.turn.cancelled") >= 1 else 0,
    )
    (output_dir / "interaction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(receiver.terminal_event)


def _event_count(receiver: LocalDemoReceiver, event_type: str) -> int:
    counts = receiver.summary.get("event_counts")
    if not isinstance(counts, dict):
        return 0
    return int(counts.get(event_type, 0))


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "scripted":
        mic_backend = args.mic_backend or "fake"
        audio_backend = args.audio_backend or "null"
        if mic_backend != "fake":
            raise SystemExit("scripted mode requires --mic-backend fake")
        terminal_event = asyncio.run(
            run_scripted_demo(
                url=args.url,
                output_dir=args.output_dir,
                scripted_turns=args.scripted_turns,
                scripted_cancel_after_ms=args.scripted_cancel_after_ms,
                chunk_ms=args.chunk_ms,
                sample_rate=args.sample_rate,
                audio_backend=audio_backend,
                playback_sync=args.playback_sync,
                wait_for_face_timeout_ms=args.wait_for_face_timeout_ms,
            )
        )
    else:
        mic_backend = args.mic_backend or "sounddevice"
        audio_backend = args.audio_backend or ("sounddevice" if args.play_audio else "null")
        terminal_event = asyncio.run(
            run_interactive_demo(
                url=args.url,
                output_dir=args.output_dir,
                command_source=StdinCommandSource(),
                microphone=create_microphone_backend(
                    mic_backend,
                    sample_rate=args.sample_rate,
                    chunk_ms=args.chunk_ms,
                ),
                audio_backend=audio_backend,
                chunk_ms=args.chunk_ms,
                sample_rate=args.sample_rate,
                playback_sync=args.playback_sync,
                wait_for_face_timeout_ms=args.wait_for_face_timeout_ms,
            )
        )
    print(
        json.dumps(
            {
                "terminal_event": terminal_event,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
