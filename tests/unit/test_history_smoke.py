from pathlib import Path
import json
import subprocess
import sys

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


def test_history_smoke_parser_accepts_real_mode_args() -> None:
    import scripts.history_smoke as history_smoke_script

    parser = history_smoke_script.build_parser()

    args = parser.parse_args(
        [
            "--url",
            "ws://127.0.0.1:8005/pipeline/stream",
            "--output-dir",
            "/tmp/history-smoke",
            "--mode",
            "real",
            "--turn1-wav",
            "/tmp/turn1.wav",
            "--turn2-wav",
            "/tmp/turn2.wav",
            "--expect",
            "小张",
        ]
    )

    assert args.mode == "real"
    assert args.turn1_wav == Path("/tmp/turn1.wav")
    assert args.turn2_wav == Path("/tmp/turn2.wav")
    assert args.expect == "小张"


def test_history_smoke_help_runs_by_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/history_smoke.py", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--mode" in result.stdout
    assert "--expect" in result.stdout
