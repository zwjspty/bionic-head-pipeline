from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID
import asyncio

from bionic_head.core.cancellation import CancellationToken
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.protocol.events import EventType

try:  # Python 3.10 fallback for local verification
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        pass


class TurnState(StrEnum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    CANCELLING = "CANCELLING"
    ERROR = "ERROR"


ALLOWED_TRANSITIONS = {
    TurnState.IDLE: {TurnState.LISTENING, TurnState.ERROR},
    TurnState.LISTENING: {TurnState.THINKING, TurnState.CANCELLING, TurnState.ERROR, TurnState.IDLE},
    TurnState.THINKING: {TurnState.SPEAKING, TurnState.CANCELLING, TurnState.ERROR, TurnState.IDLE},
    TurnState.SPEAKING: {TurnState.CANCELLING, TurnState.ERROR, TurnState.IDLE},
    TurnState.CANCELLING: {TurnState.IDLE, TurnState.ERROR},
    TurnState.ERROR: {TurnState.IDLE},
}


class TurnStateMachine:
    def __init__(self) -> None:
        self.state = TurnState.IDLE

    def transition(self, next_state: TurnState) -> None:
        if next_state not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"Illegal turn state transition: {self.state} -> {next_state}")
        self.state = next_state


@dataclass
class TurnHandle:
    session_id: UUID
    turn_id: UUID
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    active_task: asyncio.Task[object] | None = None

    _current: bool = field(default=True, init=False, repr=False)
    _terminal_event: EventType | None = field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def current(self) -> bool:
        return self._current

    @property
    def terminal_event(self) -> EventType | None:
        return self._terminal_event

    async def emit_if_current(self, operation: Callable[[], Awaitable[None]]) -> bool:
        async with self._lock:
            if not self.current or self.cancellation.cancelled:
                return False
            await operation()
            return True

    async def commit_if_current(self, callback: Callable[[], None]) -> bool:
        async with self._lock:
            if not self.current or self.cancellation.cancelled:
                return False
            callback()
            return True

    async def emit_terminal_once(self, event_type: EventType | str) -> bool:
        event_type = EventType(event_type)
        async with self._lock:
            if self._terminal_event is not None:
                return False
            if not self.current and event_type is not EventType.SERVER_TURN_CANCELLED:
                return False
            self._terminal_event = event_type
            self._current = False
            return True

    async def cancel(self) -> None:
        async with self._lock:
            self.cancellation.cancel()
            self._current = False
            if self.active_task is not None and not self.active_task.done():
                self.active_task.cancel()


class SessionManager:
    def __init__(self, *, max_active_sessions: int) -> None:
        if max_active_sessions < 1:
            raise ValueError("max_active_sessions must be at least 1")
        self.max_active_sessions = max_active_sessions
        self._active_sessions: set[UUID] = set()
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def admit(self, session_id: UUID) -> AsyncIterator[None]:
        async with self._lock:
            if session_id not in self._active_sessions and len(self._active_sessions) >= self.max_active_sessions:
                raise PipelineException(
                    code=ErrorCode.SESSION_LIMIT_REACHED,
                    stage="session",
                    provider=None,
                    retryable=True,
                    message="Session limit reached",
                )
            self._active_sessions.add(session_id)

        try:
            yield
        finally:
            async with self._lock:
                self._active_sessions.discard(session_id)
