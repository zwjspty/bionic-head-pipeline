from __future__ import annotations

from enum import Enum


class ScriptedAction(str, Enum):
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"
    WAIT_FOR_PLAYBACK = "wait_for_playback"
    CANCEL = "cancel"
    WAIT_FOR_TURN_CANCELLED = "wait_for_turn_cancelled"
    WAIT_FOR_PIPELINE_DONE = "wait_for_pipeline_done"
    QUIT = "quit"


def build_scripted_actions(*, turn_count: int, cancel_first_turn: bool) -> list[ScriptedAction]:
    if turn_count < 1:
        raise ValueError("turn_count must be at least 1")

    actions: list[ScriptedAction] = []
    for turn_index in range(turn_count):
        actions.extend([ScriptedAction.START_RECORDING, ScriptedAction.STOP_RECORDING])
        if turn_index == 0 and cancel_first_turn:
            actions.extend(
                [
                    ScriptedAction.WAIT_FOR_PLAYBACK,
                    ScriptedAction.CANCEL,
                    ScriptedAction.WAIT_FOR_TURN_CANCELLED,
                ]
            )
        elif turn_index == turn_count - 1:
            actions.append(ScriptedAction.WAIT_FOR_PIPELINE_DONE)
    actions.append(ScriptedAction.QUIT)
    return actions
