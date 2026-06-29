from pathlib import Path
import json

from bionic_head.client.history_smoke import (
    HistorySmokeTurn,
    build_history_smoke_report,
    write_history_smoke_report,
)


def test_history_smoke_report_succeeds_when_second_turn_uses_history_and_reply_contains_expected() -> None:
    report = build_history_smoke_report(
        mode="mock",
        session_id="session-1",
        expected_text="小张",
        turns=[
            HistorySmokeTurn(
                turn_index=1,
                turn_id="turn-1",
                terminal_event="server.pipeline.done",
                asr_text="我叫小张。",
                llm_reply="你好小张。",
                history_enabled=True,
                history_turn_count_before=0,
                history_turn_count_after=1,
            ),
            HistorySmokeTurn(
                turn_index=2,
                turn_id="turn-2",
                terminal_event="server.pipeline.done",
                asr_text="我叫什么？",
                llm_reply="你叫小张。",
                history_enabled=True,
                history_turn_count_before=1,
                history_turn_count_after=2,
            ),
        ],
    )

    assert report.success is True
    assert report.failure_reasons == []
    assert report.turns[1].reply_contains_expected is True


def test_history_smoke_report_fails_when_second_turn_history_is_empty() -> None:
    report = build_history_smoke_report(
        mode="mock",
        session_id="session-1",
        expected_text="小张",
        turns=[
            HistorySmokeTurn(
                turn_index=1,
                turn_id="turn-1",
                terminal_event="server.pipeline.done",
                asr_text="我叫小张。",
                llm_reply="你好小张。",
                history_enabled=True,
                history_turn_count_before=0,
                history_turn_count_after=1,
            ),
            HistorySmokeTurn(
                turn_index=2,
                turn_id="turn-2",
                terminal_event="server.pipeline.done",
                asr_text="我叫什么？",
                llm_reply="你叫小张。",
                history_enabled=True,
                history_turn_count_before=0,
                history_turn_count_after=1,
            ),
        ],
    )

    assert report.success is False
    assert "turn2_history_empty" in report.failure_reasons


def test_history_smoke_report_fails_when_expected_text_is_missing() -> None:
    report = build_history_smoke_report(
        mode="real",
        session_id="session-1",
        expected_text="小张",
        turns=[
            HistorySmokeTurn(
                turn_index=1,
                turn_id="turn-1",
                terminal_event="server.pipeline.done",
                asr_text="我叫小张。",
                llm_reply="你好。",
                history_enabled=True,
                history_turn_count_before=0,
                history_turn_count_after=1,
            ),
            HistorySmokeTurn(
                turn_index=2,
                turn_id="turn-2",
                terminal_event="server.pipeline.done",
                asr_text="我叫什么？",
                llm_reply="我不知道。",
                history_enabled=True,
                history_turn_count_before=1,
                history_turn_count_after=2,
            ),
        ],
    )

    assert report.success is False
    assert "expected_text_missing" in report.failure_reasons
    assert report.turns[1].reply_contains_expected is False


def test_write_history_smoke_report_writes_json(tmp_path: Path) -> None:
    report = build_history_smoke_report(
        mode="mock",
        session_id="session-1",
        expected_text="小张",
        turns=[
            HistorySmokeTurn(
                turn_index=1,
                turn_id="turn-1",
                terminal_event="server.pipeline.done",
                asr_text="我叫小张。",
                llm_reply="你好小张。",
                history_enabled=True,
                history_turn_count_before=0,
                history_turn_count_after=1,
            ),
            HistorySmokeTurn(
                turn_index=2,
                turn_id="turn-2",
                terminal_event="server.pipeline.done",
                asr_text="我叫什么？",
                llm_reply="你叫小张。",
                history_enabled=True,
                history_turn_count_before=1,
                history_turn_count_after=2,
            ),
        ],
    )

    output = tmp_path / "history_smoke_report.json"
    write_history_smoke_report(output, report)

    body = json.loads(output.read_text(encoding="utf-8"))
    assert body["success"] is True
    assert body["turns"][1]["reply_contains_expected"] is True
