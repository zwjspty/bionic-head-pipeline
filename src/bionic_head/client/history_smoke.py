from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class HistorySmokeTurn:
    turn_index: int
    turn_id: str
    terminal_event: str | None
    asr_text: str | None
    llm_reply: str | None
    history_enabled: bool | None
    history_turn_count_before: int | None
    history_turn_count_after: int | None
    reply_contains_expected: bool = False


@dataclass
class HistorySmokeReport:
    success: bool
    mode: str
    session_id: str
    expected_text: str
    turns: list[HistorySmokeTurn]
    failure_reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "mode": self.mode,
            "session_id": self.session_id,
            "expected_text": self.expected_text,
            "turns": [asdict(turn) for turn in self.turns],
            "failure_reasons": self.failure_reasons,
        }


def build_history_smoke_report(
    *,
    mode: str,
    session_id: str,
    expected_text: str,
    turns: list[HistorySmokeTurn],
) -> HistorySmokeReport:
    failure_reasons: list[str] = []
    if len(turns) != 2:
        failure_reasons.append("expected_two_turns")

    for turn in turns:
        if turn.terminal_event != "server.pipeline.done":
            failure_reasons.append(f"turn{turn.turn_index}_not_done")

    if len(turns) >= 2:
        second = turns[1]
        second.reply_contains_expected = bool(
            expected_text and second.llm_reply and expected_text in second.llm_reply
        )
        if second.history_enabled is not True:
            failure_reasons.append("history_disabled")
        if not second.history_turn_count_before or second.history_turn_count_before < 1:
            failure_reasons.append("turn2_history_empty")
        if not second.reply_contains_expected:
            failure_reasons.append("expected_text_missing")

    return HistorySmokeReport(
        success=not failure_reasons,
        mode=mode,
        session_id=session_id,
        expected_text=expected_text,
        turns=turns,
        failure_reasons=failure_reasons,
    )


def write_history_smoke_report(path: Path, report: HistorySmokeReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
