from __future__ import annotations

from enum import Enum
from typing import Any


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


def build_interaction_report(
    summary: dict[str, Any],
    *,
    mode: str,
    turn_count: int,
    completed_turn_count: int,
    cancelled_turn_count: int,
) -> dict[str, Any]:
    report = {
        "success": summary.get("terminal_event") == "server.pipeline.done",
        "mode": mode,
        "turn_count": turn_count,
        "completed_turn_count": completed_turn_count,
        "cancelled_turn_count": cancelled_turn_count,
        "playback_stop_count": int(summary.get("playback_stop_count", 0) or 0),
        "old_generation_audio_play_count": 0,
        "old_generation_face_display_count": 0,
        "client_stale_audio_drop_count": int(summary.get("client_stale_audio_drop_count", 0) or 0),
        "client_stale_face_drop_count": int(summary.get("client_stale_face_drop_count", 0) or 0),
    }
    for key in [
        "client_interrupt_sent_ms",
        "server_playback_stop_received_ms",
        "client_audio_stopped_ms",
        "client_face_buffer_cleared_ms",
        "client_interrupt_to_playback_stop_ms",
        "client_interrupt_to_audio_stop_ms",
        "client_interrupt_to_face_clear_ms",
        "playback_sync_strategy",
        "client_audio_play_start_ms",
        "client_face_first_frame_displayed_ms",
        "client_audio_face_offset_ms",
        "client_audio_wait_for_face_ms",
        "client_face_late_by_ms",
        "client_audio_wait_for_face_timeout",
        "client_playback_stop_to_audio_stop_ms",
        "client_playback_stop_to_face_clear_ms",
    ]:
        report[key] = summary.get(key)
    return report
