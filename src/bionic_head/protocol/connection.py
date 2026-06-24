from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from time import monotonic
from uuid import UUID
import asyncio
import json

from fastapi import WebSocket
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from bionic_head.api.dependencies import AppContainer
from bionic_head.core.audio import pcm16le_to_wav
from bionic_head.core.state import TurnHandle, TurnState, TurnStateMachine
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.protocol.events import (
    ClientAudioChunkPayload,
    ClientAudioEndPayload,
    ClientAudioStartPayload,
    ClientCancelPayload,
    ClientPingPayload,
    ClientSequenceValidator,
    ClientSessionStartPayload,
    EventEnvelope,
    EventFactory,
    EventType,
)


@dataclass
class PendingBinary:
    turn_id: UUID
    byte_length: int
    duration_ms: int


@dataclass
class PendingInterrupt:
    turn_id: UUID
    pcm_buffer: bytearray
    speech_ms: int
    turn_started_monotonic: float
    last_non_silent_monotonic: float | None


@dataclass(order=True)
class _OutboundMessage:
    priority: int
    sequence: int
    envelope: EventEnvelope = field(compare=False)
    binary: bytes | None = field(default=None, compare=False)
    completion: asyncio.Future[None] | None = field(default=None, compare=False)


_HIGH_PRIORITY_EVENTS = {
    EventType.SERVER_PLAYBACK_STOP,
    EventType.SERVER_TURN_CANCELLED,
    EventType.SERVER_PIPELINE_ERROR,
}

_WAIT_FOR_SEND_EVENTS = {
    *_HIGH_PRIORITY_EVENTS,
    EventType.SERVER_SESSION_READY,
    EventType.SERVER_STATE,
    EventType.SERVER_PONG,
    EventType.SERVER_PIPELINE_DONE,
}


class StreamConnection:
    def __init__(self, websocket: WebSocket, container: AppContainer) -> None:
        self.websocket = websocket
        self.container = container
        self.session_id: UUID | None = None
        self.event_factory: EventFactory | None = None
        self.sequence_validator = ClientSequenceValidator()
        self.state_machine = TurnStateMachine()
        self.current_turn: TurnHandle | None = None
        self.pending_binary_metadata: PendingBinary | None = None
        self.interrupt_candidate: PendingInterrupt | None = None
        self.pcm_buffer = bytearray()
        self.last_non_silent_monotonic: float | None = None
        self.turn_started_monotonic: float | None = None
        self.watchdog_task: asyncio.Task[object] | None = None
        self._send_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._provider_cancel_tasks: set[asyncio.Task[list[object]]] = set()
        self._outbound_queue: asyncio.PriorityQueue[_OutboundMessage] = asyncio.PriorityQueue()
        self._outbound_sender_task: asyncio.Task[None] | None = None
        self._outbound_sequence = 0
        self._outbound_sent_sequence = 0
        self._outbound_stale_drop_count = 0
        self.generation_epoch = 0

    async def run(self) -> None:
        admission = None
        try:
            while True:
                message = await self.websocket.receive()
                message_type = message.get("type")
                if message_type == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"] is not None:
                    await self._handle_binary(message["bytes"])
                    continue
                if "text" not in message or message["text"] is None:
                    await self._protocol_error("Unsupported WebSocket message")
                    continue
                if self.pending_binary_metadata is not None:
                    await self._protocol_error("Expected binary audio after chunk metadata")
                    continue

                envelope = await self._parse_envelope(message["text"])
                if envelope is None:
                    continue
                if envelope.type is EventType.CLIENT_SESSION_START:
                    if admission is not None:
                        await self._protocol_error("Session already started")
                        continue
                    ClientSessionStartPayload.model_validate(envelope.payload)
                    self.session_id = envelope.session_id
                    self.event_factory = EventFactory(
                        session_id=self.session_id,
                        generation_epoch_getter=lambda: self.generation_epoch,
                    )
                    try:
                        admission = self.container.sessions.admit(self.session_id)
                        await admission.__aenter__()
                    except PipelineException as exc:
                        await self._send_pipeline_error(exc, turn_id=None)
                        admission = None
                        continue
                    if (
                        self.container.settings.adapters.audio2face.provider == "emotalk_sidecar"
                        and self.container.settings.providers.emotalk_sidecar.prewarm_on_session_start
                    ):
                        try:
                            await self.container.prewarm_audio2face()
                        except PipelineException as exc:
                            await self._send_pipeline_error(exc, turn_id=None)
                            await admission.__aexit__(None, None, None)
                            admission = None
                            continue
                    await self._send_json_direct(
                        self._factory().server(EventType.SERVER_SESSION_READY, None, {})
                    )
                    continue

                try:
                    self._validate_client_event(envelope)
                    await self._handle_event(envelope)
                except PipelineException as exc:
                    await self._send_pipeline_error(exc, envelope.turn_id)
        except WebSocketDisconnect:
            pass
        finally:
            await self._cancel_current_turn(emit=False)
            if admission is not None:
                await admission.__aexit__(None, None, None)

    async def _parse_envelope(self, raw_text: str) -> EventEnvelope | None:
        try:
            envelope = EventEnvelope.model_validate(json.loads(raw_text))
            self.sequence_validator.validate(envelope)
            return envelope
        except (json.JSONDecodeError, ValidationError) as exc:
            await self._protocol_error("Invalid event envelope")
            return None
        except PipelineException as exc:
            await self._send_pipeline_error(exc, turn_id=None)
            return None

    def _validate_client_event(self, envelope: EventEnvelope) -> None:
        if self.session_id is None or self.event_factory is None:
            raise self._violation("client.session.start is required before audio")
        if envelope.session_id != self.session_id:
            raise self._violation("Client event session_id does not match the active session")
        if envelope.type not in {EventType.CLIENT_AUDIO_START, EventType.CLIENT_PING}:
            if self._is_interrupt_candidate_event(envelope):
                return
            if self.current_turn is None or envelope.turn_id != self.current_turn.turn_id:
                raise self._violation("Client event turn_id does not match the active turn")

    async def _handle_event(self, envelope: EventEnvelope) -> None:
        if envelope.type is EventType.CLIENT_PING:
            ClientPingPayload.model_validate(envelope.payload)
            await self._send_json_direct(
                self._factory().server(EventType.SERVER_PONG, envelope.turn_id, {})
            )
        elif envelope.type is EventType.CLIENT_AUDIO_START:
            ClientAudioStartPayload.model_validate(envelope.payload)
            await self._start_turn(envelope)
        elif envelope.type is EventType.CLIENT_AUDIO_CHUNK:
            await self._handle_audio_chunk_metadata(envelope)
        elif envelope.type is EventType.CLIENT_AUDIO_END:
            ClientAudioEndPayload.model_validate(envelope.payload)
            if self._is_interrupt_candidate_event(envelope):
                self._discard_interrupt_candidate()
                return
            await self._finalize_current_turn()
        elif envelope.type is EventType.CLIENT_TURN_CANCEL:
            ClientCancelPayload.model_validate(envelope.payload)
            await self._cancel_current_turn(emit=True)
        else:
            raise self._violation(f"Unsupported client event type: {envelope.type.value}")

    async def _start_turn(self, envelope: EventEnvelope) -> None:
        async with self._state_lock:
            if self.current_turn is not None and self.state_machine.state in {
                TurnState.SPEAKING,
                TurnState.THINKING,
            }:
                self._start_interrupt_candidate(envelope.turn_id)
                return

            if self.current_turn is not None and self.state_machine.state in {
                TurnState.LISTENING,
            }:
                await self._cancel_current_turn(emit=True)

            self.current_turn = TurnHandle(
                session_id=self.session_id,
                turn_id=envelope.turn_id,
                generation_epoch=self.generation_epoch,
                generation_epoch_getter=lambda: self.generation_epoch,
            )
            self.pcm_buffer = bytearray()
            self.pending_binary_metadata = None
            self.turn_started_monotonic = monotonic()
            self.last_non_silent_monotonic = self.turn_started_monotonic
            self.state_machine = TurnStateMachine()
            self.state_machine.transition(TurnState.LISTENING)
            self.watchdog_task = asyncio.create_task(self._watchdog())
            await self._send_json_direct(
                self._factory().server(
                    EventType.SERVER_STATE,
                    envelope.turn_id,
                    {"state": TurnState.LISTENING.value},
                )
            )

    async def _handle_audio_chunk_metadata(self, envelope: EventEnvelope) -> None:
        if not self._is_interrupt_candidate_event(envelope) and self.state_machine.state is not TurnState.LISTENING:
            raise self._violation("Audio chunk received outside LISTENING state")
        if self.pending_binary_metadata is not None:
            raise self._violation("Previous audio chunk metadata is still waiting for binary")
        payload = ClientAudioChunkPayload.model_validate(envelope.payload)
        if not 20 <= payload.duration_ms <= 100:
            raise self._violation("Audio chunk duration must be 20-100 ms")
        self.pending_binary_metadata = PendingBinary(
            turn_id=envelope.turn_id,
            byte_length=payload.byte_length,
            duration_ms=payload.duration_ms,
        )

    async def _handle_binary(self, payload: bytes) -> None:
        pending = self.pending_binary_metadata
        if pending is None:
            await self._protocol_error("Binary audio arrived without chunk metadata")
            return
        self.pending_binary_metadata = None
        if len(payload) != pending.byte_length:
            await self._protocol_error("Binary audio byte_length does not match metadata")
            return
        if self._is_interrupt_candidate_turn(pending.turn_id):
            await self._handle_interrupt_candidate_binary(payload, pending.duration_ms)
            return
        if self.state_machine.state is not TurnState.LISTENING:
            await self._protocol_error("Binary audio arrived outside LISTENING state")
            return
        self.pcm_buffer.extend(payload)
        if self._pcm_rms(payload) > self.container.settings.stream.silence_rms_threshold:
            self.last_non_silent_monotonic = monotonic()

    async def _finalize_current_turn(self) -> None:
        async with self._state_lock:
            if self.current_turn is None or self.state_machine.state is not TurnState.LISTENING:
                return
            if self.pending_binary_metadata is not None:
                raise self._violation("Audio turn ended before expected binary chunk")
            turn = self.current_turn
            if self.watchdog_task is not None:
                self.watchdog_task.cancel()
                self.watchdog_task = None
            turn_dir = self.container.store.create_turn(turn.session_id, turn.turn_id)
            input_wav = turn_dir / "ws_input.wav"
            pcm16le_to_wav(bytes(self.pcm_buffer), input_wav)
            self.state_machine.transition(TurnState.THINKING)
            orchestrator = self.container.make_stream_orchestrator()
            task = asyncio.create_task(
                orchestrator.run(
                    input_wav,
                    turn,
                    self._send_json_direct,
                    self._send_binary_pair_direct,
                    self._factory(),
                )
            )
            turn.active_task = task

    async def _watchdog(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.05)
                if self.current_turn is None or self.state_machine.state is not TurnState.LISTENING:
                    return
                now = monotonic()
                if (
                    self.last_non_silent_monotonic is not None
                    and (now - self.last_non_silent_monotonic) * 1000 >= self.container.settings.stream.silence_timeout_ms
                ):
                    await self._finalize_current_turn()
                    return
                if (
                    self.turn_started_monotonic is not None
                    and now - self.turn_started_monotonic >= self.container.settings.stream.max_turn_duration_seconds
                ):
                    await self._finalize_current_turn()
                    return
        except asyncio.CancelledError:
            return

    async def _cancel_current_turn(self, *, emit: bool, clear_interrupt: bool = True) -> None:
        turn = self.current_turn
        if turn is None:
            return
        await turn.cancel()
        self._schedule_provider_cancel(turn.turn_id)
        if emit:
            self.generation_epoch += 1
        if self.watchdog_task is not None:
            self.watchdog_task.cancel()
            self.watchdog_task = None
        self.pending_binary_metadata = None
        self.pcm_buffer = bytearray()
        if emit:
            await self._send_json_direct(
                self._factory().server(EventType.SERVER_PLAYBACK_STOP, turn.turn_id, {})
            )
        if emit and await turn.emit_terminal_once(EventType.SERVER_TURN_CANCELLED):
            await self._send_json_direct(
                self._factory().server(EventType.SERVER_TURN_CANCELLED, turn.turn_id, {})
            )
        await self._yield_to_provider_cancel_tasks()
        if clear_interrupt:
            self._discard_interrupt_candidate()
        self.current_turn = None
        self.state_machine = TurnStateMachine()

    def _schedule_provider_cancel(self, turn_id: UUID) -> None:
        task: asyncio.Task[list[object]] = asyncio.create_task(
            self.container.cancel_turn_providers(turn_id)
        )
        self._provider_cancel_tasks.add(task)
        task.add_done_callback(self._provider_cancel_tasks.discard)

    async def _yield_to_provider_cancel_tasks(self) -> None:
        # Do not wait on slow provider cleanup here: playback.stop already went out.
        # A few zero-time slices let no-op/fast cancels finish before short-lived
        # test WebSocket loops close, avoiding pending-task noise.
        for _ in range(6):
            if not self._provider_cancel_tasks:
                return
            await asyncio.sleep(0)

    def _start_interrupt_candidate(self, turn_id: UUID) -> None:
        now = monotonic()
        self.interrupt_candidate = PendingInterrupt(
            turn_id=turn_id,
            pcm_buffer=bytearray(),
            speech_ms=0,
            turn_started_monotonic=now,
            last_non_silent_monotonic=None,
        )
        self.pending_binary_metadata = None

    async def _handle_interrupt_candidate_binary(self, payload: bytes, duration_ms: int) -> None:
        candidate = self.interrupt_candidate
        if candidate is None:
            return
        candidate.pcm_buffer.extend(payload)
        if self._pcm_rms(payload) > self.container.settings.vad.interrupt_rms_threshold:
            candidate.speech_ms += duration_ms
            candidate.last_non_silent_monotonic = monotonic()
        if candidate.speech_ms >= self.container.settings.vad.interrupt_min_speech_ms:
            await self._accept_interrupt_candidate(candidate)

    async def _accept_interrupt_candidate(self, candidate: PendingInterrupt) -> None:
        self.interrupt_candidate = None
        await self._cancel_current_turn(emit=True, clear_interrupt=False)
        self.current_turn = TurnHandle(
            session_id=self.session_id,
            turn_id=candidate.turn_id,
            generation_epoch=self.generation_epoch,
            generation_epoch_getter=lambda: self.generation_epoch,
        )
        self.pcm_buffer = bytearray(candidate.pcm_buffer)
        self.pending_binary_metadata = None
        self.turn_started_monotonic = candidate.turn_started_monotonic
        self.last_non_silent_monotonic = candidate.last_non_silent_monotonic
        self.state_machine = TurnStateMachine()
        self.state_machine.transition(TurnState.LISTENING)
        self.watchdog_task = asyncio.create_task(self._watchdog())
        await self._send_json_direct(
            self._factory().server(
                EventType.SERVER_STATE,
                candidate.turn_id,
                {"state": TurnState.LISTENING.value},
            )
        )

    def _discard_interrupt_candidate(self) -> None:
        if self.interrupt_candidate is not None:
            if (
                self.pending_binary_metadata is not None
                and self.pending_binary_metadata.turn_id == self.interrupt_candidate.turn_id
            ):
                self.pending_binary_metadata = None
            self.interrupt_candidate = None

    def _is_interrupt_candidate_event(self, envelope: EventEnvelope) -> bool:
        return self._is_interrupt_candidate_turn(envelope.turn_id)

    def _is_interrupt_candidate_turn(self, turn_id: UUID | None) -> bool:
        return self.interrupt_candidate is not None and turn_id == self.interrupt_candidate.turn_id

    async def _protocol_error(self, message: str) -> None:
        await self._send_pipeline_error(self._violation(message), self.current_turn.turn_id if self.current_turn else None)

    async def _send_pipeline_error(self, exc: PipelineException, turn_id: UUID | None) -> None:
        if self.event_factory is None:
            return
        turn = self.current_turn
        if turn is not None:
            if not await turn.emit_terminal_once(EventType.SERVER_PIPELINE_ERROR):
                return
            await turn.cancel()
        await self._send_json_direct(
            self._factory().server(
                EventType.SERVER_PIPELINE_ERROR,
                turn_id,
                {"error": exc.to_detail()},
            )
        )
        self.current_turn = None
        self.state_machine = TurnStateMachine()

    async def _send_json_direct(self, envelope: EventEnvelope) -> None:
        await self._send_outbound(envelope)

    async def _send_binary_pair_direct(self, envelope: EventEnvelope, binary: bytes) -> None:
        await self._send_outbound(envelope, binary=binary)

    async def _send_outbound(self, envelope: EventEnvelope, *, binary: bytes | None = None) -> None:
        priority = self._outbound_priority(envelope)
        wait_for_send = envelope.type in _WAIT_FOR_SEND_EVENTS
        completion: asyncio.Future[None] | None = None
        if wait_for_send:
            completion = asyncio.get_running_loop().create_future()
        self._outbound_sequence += 1
        await self._outbound_queue.put(
            _OutboundMessage(
                priority=priority,
                sequence=self._outbound_sequence,
                envelope=envelope,
                binary=binary,
                completion=completion,
            )
        )
        self._ensure_outbound_sender()
        if completion is not None:
            await completion

    def _ensure_outbound_sender(self) -> None:
        if self._outbound_sender_task is None or self._outbound_sender_task.done():
            self._outbound_sender_task = asyncio.create_task(self._outbound_sender_loop())

    async def _outbound_sender_loop(self) -> None:
        try:
            while True:
                try:
                    message = self._outbound_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await self._send_outbound_message(message)
        finally:
            self._outbound_sender_task = None

    async def _send_outbound_message(self, message: _OutboundMessage) -> None:
        try:
            if self._should_drop_outbound(message):
                self._outbound_stale_drop_count += 1
                if message.completion is not None and not message.completion.done():
                    message.completion.set_result(None)
                return

            envelope = self._next_outbound_envelope(message.envelope)
            async with self._send_lock:
                await self.websocket.send_json(envelope.model_dump(mode="json"))
                if message.binary is not None:
                    await self.websocket.send_bytes(message.binary)
            self._observe_server_event(envelope)
            if message.completion is not None and not message.completion.done():
                message.completion.set_result(None)
        except Exception as exc:
            if message.completion is not None and not message.completion.done():
                message.completion.set_exception(exc)
            raise

    def _outbound_priority(self, envelope: EventEnvelope) -> int:
        if envelope.type in _HIGH_PRIORITY_EVENTS:
            return 0
        return 10

    def _should_drop_outbound(self, message: _OutboundMessage) -> bool:
        if message.envelope.type in _HIGH_PRIORITY_EVENTS:
            return False
        if message.envelope.turn_id is None:
            return False
        if message.envelope.generation_epoch is None:
            return False
        return message.envelope.generation_epoch != self.generation_epoch

    def _next_outbound_envelope(self, envelope: EventEnvelope) -> EventEnvelope:
        self._outbound_sent_sequence += 1
        return envelope.model_copy(update={"sequence": self._outbound_sent_sequence})

    async def _yield_to_outbound_sender(self) -> None:
        for _ in range(6):
            if self._outbound_queue.empty() and (
                self._outbound_sender_task is None or self._outbound_sender_task.done()
            ):
                return
            await asyncio.sleep(0)

    def _observe_server_event(self, envelope: EventEnvelope) -> None:
        if envelope.type is EventType.SERVER_SEGMENT_READY and self.state_machine.state is TurnState.THINKING:
            self.state_machine.transition(TurnState.SPEAKING)
        if envelope.type in {
            EventType.SERVER_PIPELINE_DONE,
            EventType.SERVER_PIPELINE_ERROR,
            EventType.SERVER_TURN_CANCELLED,
        }:
            self.state_machine = TurnStateMachine()
            self.current_turn = None

    def _factory(self) -> EventFactory:
        if self.event_factory is None:
            raise RuntimeError("session has not started")
        return self.event_factory

    def _violation(self, message: str) -> PipelineException:
        return PipelineException(
            code=ErrorCode.PROTOCOL_VIOLATION,
            stage="websocket",
            provider=None,
            retryable=False,
            message=message,
        )

    def _pcm_rms(self, payload: bytes) -> float:
        if not payload or len(payload) % 2 != 0:
            return 0.0
        samples = array("h")
        samples.frombytes(payload)
        if not samples:
            return 0.0
        mean_square = sum(sample * sample for sample in samples) / len(samples)
        return (mean_square**0.5) / 32768.0
