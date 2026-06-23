from __future__ import annotations

from uuid import uuid4

import pytest

from bionic_head.core.state import SessionManager, TurnHandle, TurnState, TurnStateMachine
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.protocol.events import EventType


def test_normal_and_cancel_transitions() -> None:
    machine = TurnStateMachine()
    machine.transition(TurnState.LISTENING)
    machine.transition(TurnState.THINKING)
    machine.transition(TurnState.CANCELLING)
    machine.transition(TurnState.IDLE)

    assert machine.state is TurnState.IDLE


def test_illegal_transition_is_rejected() -> None:
    machine = TurnStateMachine()

    with pytest.raises(ValueError):
        machine.transition(TurnState.SPEAKING)


@pytest.mark.asyncio
async def test_turn_handle_suppresses_events_after_cancel() -> None:
    handle = TurnHandle(session_id=uuid4(), turn_id=uuid4())
    emitted: list[str] = []

    async def emit() -> None:
        emitted.append("event")

    assert await handle.emit_if_current(emit) is True
    await handle.cancel()
    assert await handle.emit_if_current(emit) is False
    assert await handle.commit_if_current(lambda: emitted.append("commit")) is False
    assert emitted == ["event"]


@pytest.mark.asyncio
async def test_turn_handle_suppresses_events_after_generation_epoch_changes() -> None:
    current_epoch = 1
    handle = TurnHandle(
        session_id=uuid4(),
        turn_id=uuid4(),
        generation_epoch=1,
        generation_epoch_getter=lambda: current_epoch,
    )
    emitted: list[str] = []

    async def emit() -> None:
        emitted.append("event")

    assert await handle.emit_if_current(emit) is True
    current_epoch = 2

    assert await handle.emit_if_current(emit) is False
    assert await handle.commit_if_current(lambda: emitted.append("commit")) is False
    assert emitted == ["event"]


@pytest.mark.asyncio
async def test_terminal_event_can_only_win_once() -> None:
    handle = TurnHandle(session_id=uuid4(), turn_id=uuid4())

    assert await handle.emit_terminal_once(EventType.SERVER_PIPELINE_DONE) is True
    assert await handle.emit_terminal_once(EventType.SERVER_PIPELINE_ERROR) is False


@pytest.mark.asyncio
async def test_terminal_event_makes_handle_not_current() -> None:
    handle = TurnHandle(session_id=uuid4(), turn_id=uuid4())
    emitted: list[str] = []

    async def emit() -> None:
        emitted.append("late")

    assert await handle.emit_terminal_once(EventType.SERVER_PIPELINE_DONE) is True
    assert await handle.emit_if_current(emit) is False
    assert await handle.commit_if_current(lambda: emitted.append("commit")) is False
    assert emitted == []


@pytest.mark.asyncio
async def test_session_manager_allows_only_one_active_session() -> None:
    manager = SessionManager(max_active_sessions=1)

    async with manager.admit(uuid4()):
        with pytest.raises(PipelineException) as raised:
            async with manager.admit(uuid4()):
                pass

    assert raised.value.code is ErrorCode.SESSION_LIMIT_REACHED
    async with manager.admit(uuid4()):
        pass
