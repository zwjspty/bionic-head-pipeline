from __future__ import annotations

import json
import wave
from pathlib import Path

from bionic_head.client.demo_acceptance import (
    AcceptanceCheckResult,
    DemoAcceptanceReport,
    build_demo_acceptance_report,
    write_demo_input_wav,
    write_json,
)


def test_build_report_fails_when_required_check_fails() -> None:
    report = build_demo_acceptance_report(
        mode="fake",
        server={"health_ok": True, "diagnostics_ok": True},
        checks={
            "scripted_interactive_smoke": AcceptanceCheckResult(
                success=True,
                artifacts={"interaction_report": "scripted/interaction_report.json"},
            ),
            "history_smoke": AcceptanceCheckResult(
                success=False,
                failure_code="history_smoke_failed",
                failure_message="History smoke did not preserve expected context.",
                artifacts={"history_smoke_report": "history/history_smoke_report.json"},
            ),
        },
        artifacts={},
    )

    assert isinstance(report, DemoAcceptanceReport)
    body = report.to_dict()
    assert body["success"] is False
    assert "history_smoke:history_smoke_failed" in body["failure_reasons"]
    assert body["checks"]["history_smoke"]["success"] is False
    assert body["checks"]["history_smoke"]["failure_code"] == "history_smoke_failed"
    assert body["checks"]["history_smoke"]["failure_message"] == "History smoke did not preserve expected context."


def test_build_report_succeeds_when_server_and_checks_pass() -> None:
    report = build_demo_acceptance_report(
        mode="fake",
        server={"health_ok": True, "diagnostics_ok": True},
        checks={
            "scripted_interactive_smoke": AcceptanceCheckResult(success=True),
            "history_smoke": AcceptanceCheckResult(success=True),
        },
        artifacts={"latest_pipeline": "artifacts/latest_pipeline.json"},
    )

    body = report.to_dict()
    assert body["success"] is True
    assert body["failure_reasons"] == []
    assert body["artifacts"]["latest_pipeline"] == "artifacts/latest_pipeline.json"


def test_build_report_fails_when_server_health_fails() -> None:
    report = build_demo_acceptance_report(
        mode="fake",
        server={"health_ok": False, "diagnostics_ok": True},
        checks={"scripted_interactive_smoke": AcceptanceCheckResult(success=True)},
        artifacts={},
    )

    assert report.success is False
    assert "server:server_health_unreachable" in report.failure_reasons


def test_write_json_creates_parent_and_writes_utf8(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "report.json"

    write_json(output, {"success": True, "message": "你好"})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "success": True,
        "message": "你好",
    }


def test_write_json_accepts_report_dataclass(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    report = build_demo_acceptance_report(
        mode="fake",
        server={"health_ok": True, "diagnostics_ok": True},
        checks={"scripted_interactive_smoke": AcceptanceCheckResult(success=True)},
        artifacts={},
    )

    write_json(output, report)

    assert json.loads(output.read_text(encoding="utf-8"))["success"] is True


def test_write_demo_input_wav_creates_16k_mono_pcm(tmp_path: Path) -> None:
    wav_path = write_demo_input_wav(tmp_path / "generated-input.wav")

    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 16000
