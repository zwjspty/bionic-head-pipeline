from bionic_head.client.scripted import ScriptedAction, build_scripted_actions


def test_scripted_controller_generates_two_turn_cancel_flow() -> None:
    actions = build_scripted_actions(turn_count=2, cancel_first_turn=True)

    assert actions == [
        ScriptedAction.START_RECORDING,
        ScriptedAction.STOP_RECORDING,
        ScriptedAction.WAIT_FOR_PLAYBACK,
        ScriptedAction.CANCEL,
        ScriptedAction.WAIT_FOR_TURN_CANCELLED,
        ScriptedAction.START_RECORDING,
        ScriptedAction.STOP_RECORDING,
        ScriptedAction.WAIT_FOR_PIPELINE_DONE,
        ScriptedAction.QUIT,
    ]
