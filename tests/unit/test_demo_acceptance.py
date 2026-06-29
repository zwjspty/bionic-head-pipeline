from __future__ import annotations

import json
import asyncio
import argparse
import subprocess
import sys
from pathlib import Path
import urllib.error
import wave

import pytest

from bionic_head.client.demo_acceptance import (
    AcceptanceCheckResult,
    DemoAcceptanceReport,
    build_demo_acceptance_report,
    write_demo_input_wav,
    write_json,
)
from bionic_head.client import demo_artifacts
from bionic_head.client.demo_artifacts import (
    collect_existing_artifacts,
    collect_latest_artifacts,
    http_get_json,
)


class FakeHTTPResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_run_demo_acceptance_help_runs_by_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_demo_acceptance.py", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--playback-sync" in result.stdout
    assert "--history-turn1-wav" in result.stdout


def test_collect_demo_artifacts_help_runs_by_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/collect_demo_artifacts.py", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--output-dir" in result.stdout


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
    wav_path = write_demo_input_wav(tmp_path / "generated-input.wav", 8000, 0.5)

    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getframerate() == 8000
        assert wav.getnframes() == 4000


def test_http_get_json_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        demo_artifacts.urllib.request,
        "urlopen",
        lambda request, timeout: FakeHTTPResponse(b'{"status":"ok"}'),
    )

    ok, payload, error = http_get_json("http://127.0.0.1:8005/health")

    assert ok is True
    assert payload == {"status": "ok"}
    assert error is None


def test_http_get_json_handles_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_error(request, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(demo_artifacts.urllib.request, "urlopen", raise_error)

    ok, payload, error = http_get_json("http://127.0.0.1:8005/health")

    assert ok is False
    assert payload is None
    assert "refused" in str(error)


def test_http_get_json_returns_false_none_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        demo_artifacts.urllib.request,
        "urlopen",
        lambda request, timeout: FakeHTTPResponse(b"not-json"),
    )

    ok, payload, error = http_get_json("http://127.0.0.1:8005/health")

    assert ok is False
    assert payload is None
    assert error is not None


def test_http_get_json_returns_false_none_on_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_error(request, timeout):
        raise OSError("filesystem unavailable")

    monkeypatch.setattr(demo_artifacts.urllib.request, "urlopen", raise_error)

    ok, payload, error = http_get_json("http://127.0.0.1:8005/health")

    assert ok is False
    assert payload is None
    assert error is not None


def test_collect_latest_artifacts_copies_local_latest(tmp_path: Path) -> None:
    output_dir = tmp_path / "acceptance"
    latest_dir = tmp_path / "latest"
    latest_dir.mkdir()
    (latest_dir / "latest_pipeline.json").write_text('{"ok": true}', encoding="utf-8")
    (latest_dir / "latest_ue5_blendshape.json").write_text('{"frames": []}', encoding="utf-8")

    artifacts = collect_latest_artifacts(
        output_dir=output_dir,
        http_base_url=None,
        data_latest_dir=latest_dir,
    )

    assert artifacts == {
        "latest_pipeline": "artifacts/latest_pipeline.json",
        "latest_ue5": "artifacts/latest_ue5_blendshape.json",
    }
    assert (output_dir / "artifacts" / "latest_pipeline.json").exists()
    assert (output_dir / "artifacts" / "latest_ue5_blendshape.json").exists()


def test_collect_latest_artifacts_collects_http_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "acceptance"

    def fake_urlopen(request, timeout):
        assert isinstance(request.full_url, str)
        if request.full_url.endswith("/pipeline/latest"):
            return FakeHTTPResponse(json.dumps({"pipeline": "ok"}).encode("utf-8"))
        if request.full_url.endswith("/ue5/latest"):
            return FakeHTTPResponse(json.dumps({"frames": []}).encode("utf-8"))
        raise AssertionError(f"unexpected url: {request.full_url}")

    monkeypatch.setattr(demo_artifacts.urllib.request, "urlopen", fake_urlopen)

    artifacts = collect_latest_artifacts(
        output_dir=output_dir,
        http_base_url="http://127.0.0.1:8005",
        data_latest_dir=None,
    )

    assert artifacts == {
        "latest_pipeline": "artifacts/latest_pipeline.json",
        "latest_ue5": "artifacts/latest_ue5_blendshape.json",
    }
    assert json.loads((output_dir / "artifacts" / "latest_pipeline.json").read_text(encoding="utf-8")) == {"pipeline": "ok"}
    assert json.loads((output_dir / "artifacts" / "latest_ue5_blendshape.json").read_text(encoding="utf-8")) == {"frames": []}


def test_collect_existing_artifacts_tracks_present_and_missing_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "acceptance"
    output_dir.mkdir()
    present = output_dir / "scripted" / "summary.json"
    present.parent.mkdir()
    present.write_text("{}", encoding="utf-8")
    missing = output_dir / "history" / "events.jsonl"

    artifacts = collect_existing_artifacts(
        output_dir,
        {
            "scripted_summary": present,
            "history_events": missing,
        },
    )

    assert artifacts == {"scripted_summary": "scripted/summary.json"}


def test_collect_demo_artifacts_parser_accepts_paths() -> None:
    import scripts.collect_demo_artifacts as collect_script

    parser = collect_script.build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--http-base-url",
            "http://127.0.0.1:8005",
            "--data-latest-dir",
            "data/latest",
        ]
    )

    assert args.output_dir == Path("/tmp/out")
    assert args.http_base_url == "http://127.0.0.1:8005"
    assert args.data_latest_dir == Path("data/latest")


def test_run_demo_acceptance_parser_accepts_fake_mode() -> None:
    import scripts.run_demo_acceptance as runner

    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--url",
            "ws://127.0.0.1:8005/pipeline/stream",
            "--http-base-url",
            "http://127.0.0.1:8005",
            "--output-dir",
            "/tmp/acceptance",
            "--mode",
            "fake",
            "--audio-backend",
            "null",
            "--playback-sync",
            "immediate_audio",
            "wait_for_face",
        ]
    )

    assert args.mode == "fake"
    assert args.playback_sync == ["immediate_audio", "wait_for_face"]


def test_run_demo_acceptance_parser_rejects_real_without_history_wavs() -> None:
    import scripts.run_demo_acceptance as runner

    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--url",
            "ws://127.0.0.1:8005/pipeline/stream",
            "--http-base-url",
            "http://127.0.0.1:8005",
            "--output-dir",
            "/tmp/acceptance",
            "--mode",
            "real",
        ]
    )

    with pytest.raises(SystemExit, match="real mode requires"):
        runner.validate_args(args)


def test_run_demo_acceptance_collects_health_diagnostics_artifacts_and_returns_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_demo_acceptance as runner

    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--url",
            "ws://127.0.0.1:8005/pipeline/stream",
            "--http-base-url",
            "http://127.0.0.1:8005",
            "--output-dir",
            str(tmp_path / "acceptance" / "output"),
        ]
    )

    output_dir = args.output_dir
    assert not output_dir.exists()
    args.data_latest_dir = tmp_path / "data" / "latest"
    args.data_latest_dir.mkdir(parents=True, exist_ok=True)

    health_requests: list[str] = []
    diag_requests: list[str] = []

    def fake_http_get_json(url: str, timeout_sec: float) -> tuple[bool, object | None, str | None]:
        if url.endswith("/health"):
            health_requests.append(url)
            return True, {"status": "ok"}, None
        if url.endswith("/diagnostics"):
            diag_requests.append(url)
            return True, {"status": "ok"}, None
        raise AssertionError(f"unexpected endpoint: {url}")

    monkeypatch.setattr(runner, "http_get_json", fake_http_get_json)
    async def fake_scripted_check(args) -> AcceptanceCheckResult:
        return AcceptanceCheckResult(success=True)

    async def fake_history_check(args) -> AcceptanceCheckResult:
        return AcceptanceCheckResult(success=True)

    async def fake_playback_interrupt_check(args) -> AcceptanceCheckResult:
        return AcceptanceCheckResult(success=True)

    monkeypatch.setattr(runner, "_run_scripted_interactive_check", fake_scripted_check)
    monkeypatch.setattr(runner, "_run_history_check", fake_history_check)
    monkeypatch.setattr(runner, "_run_playback_interrupt_check", fake_playback_interrupt_check)

    async def fake_av_sync_check(args, *, playback_sync: str) -> AcceptanceCheckResult:
        return AcceptanceCheckResult(success=True, metrics={"playback_sync_strategy": playback_sync})

    monkeypatch.setattr(runner, "_run_av_sync_check", fake_av_sync_check)

    collect_calls: dict[str, object] = {}

    def fake_collect_latest_artifacts(
        *,
        output_dir: Path,
        http_base_url: str,
        data_latest_dir: Path | None,
        timeout_sec: float,
    ) -> dict[str, str]:
        collect_calls["output_dir"] = output_dir
        collect_calls["http_base_url"] = http_base_url
        collect_calls["data_latest_dir"] = data_latest_dir
        collect_calls["timeout_sec"] = timeout_sec
        return {"latest_pipeline": "artifacts/latest_pipeline.json"}

    monkeypatch.setattr(runner, "collect_latest_artifacts", fake_collect_latest_artifacts)

    report = asyncio.run(runner.run_demo_acceptance(args))

    assert isinstance(report, dict)
    assert report["success"] is True
    assert "latest_pipeline" in report["artifacts"]
    assert output_dir.exists()
    assert health_requests and diag_requests
    assert any(url.endswith("/health") for url in health_requests)
    assert any(url.endswith("/diagnostics") for url in diag_requests)

    assert collect_calls == {
        "output_dir": output_dir,
        "http_base_url": "http://127.0.0.1:8005",
        "data_latest_dir": args.data_latest_dir,
        "timeout_sec": args.timeout_sec,
    }

    report_path = output_dir / "demo_acceptance_report.json"
    assert report_path.exists()
    file_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert file_payload == report


@pytest.mark.asyncio
async def test_run_demo_acceptance_aggregates_fake_checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.run_demo_acceptance as runner

    async def fake_check_server(http_base_url: str, *, timeout_sec: float):
        return {"health_ok": True, "diagnostics_ok": True}

    async def fake_scripted(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "interaction_report.json",
            {
                "success": True,
                "turn_count": kwargs.get("scripted_turns", 1),
                "completed_turn_count": 1,
                "cancelled_turn_count": 1 if kwargs.get("scripted_turns", 1) > 1 else 0,
                "old_generation_audio_play_count": 0,
                "old_generation_face_display_count": 0,
                "playback_sync_strategy": kwargs.get("playback_sync", "immediate_audio"),
                "client_audio_face_offset_ms": 0.5,
                "client_audio_wait_for_face_ms": 10.0 if kwargs.get("playback_sync") == "wait_for_face" else 0.0,
                "client_audio_wait_for_face_timeout": False,
            },
        )
        write_json(output_dir / "summary.json", {"terminal_event": "server.pipeline.done"})
        return "server.pipeline.done"

    async def fake_history(**kwargs):
        from bionic_head.client.history_smoke import HistorySmokeReport

        report = HistorySmokeReport(
            success=True,
            mode=kwargs["mode"],
            session_id="session-1",
            expected_text=kwargs["expected_text"],
            failure_reasons=[],
            turns=[],
        )
        write_json(kwargs["output_dir"] / "history_smoke_report.json", report.to_dict())
        write_json(kwargs["output_dir"] / "summary.json", {"success": True})
        return report

    async def fake_local(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "summary.json",
            {
                "terminal_event": "server.turn.cancelled",
                "playback_stop_count": 1,
                "client_interrupt_sent_ms": 10.0,
            },
        )
        return "server.turn.cancelled"

    monkeypatch.setattr(runner, "_check_server", fake_check_server)
    monkeypatch.setattr(runner.interactive_demo_client, "run_scripted_demo", fake_scripted)
    monkeypatch.setattr(runner.history_smoke, "run_history_smoke", fake_history)
    monkeypatch.setattr(runner.local_demo_client, "run_local_demo", fake_local)
    monkeypatch.setattr(runner, "collect_latest_artifacts", lambda **kwargs: {})

    args = argparse.Namespace(
        url="ws://127.0.0.1:8005/pipeline/stream",
        http_base_url="http://127.0.0.1:8005",
        output_dir=tmp_path,
        mode="fake",
        audio_backend="null",
        playback_sync=["immediate_audio", "wait_for_face"],
        wait_for_face_timeout_ms=800,
        history_turn1_wav=None,
        history_turn2_wav=None,
        expect="小张",
        chunk_ms=40,
        timeout_sec=30.0,
        data_latest_dir=None,
    )

    report = await runner.run_demo_acceptance(args)

    assert report["success"] is True
    assert set(report["checks"]) == {
        "scripted_interactive_smoke",
        "history_smoke",
        "playback_interrupt_smoke",
        "av_sync_immediate_audio",
        "av_sync_wait_for_face",
    }
    assert (tmp_path / "demo_acceptance_report.json").exists()
    assert report["checks"]["av_sync_wait_for_face"]["metrics"]["client_audio_wait_for_face_ms"] == 10.0


@pytest.mark.asyncio
async def test_run_demo_acceptance_writes_report_when_history_smoke_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.run_demo_acceptance as runner

    async def fake_check_server(http_base_url: str, *, timeout_sec: float):
        return {"health_ok": True, "diagnostics_ok": True}

    async def passing_check(args) -> AcceptanceCheckResult:
        return AcceptanceCheckResult(success=True)

    async def failing_history(**kwargs):
        raise SystemExit("history smoke exited")

    monkeypatch.setattr(runner, "_check_server", fake_check_server)
    monkeypatch.setattr(runner, "_run_scripted_interactive_check", passing_check)
    monkeypatch.setattr(runner, "_run_playback_interrupt_check", passing_check)
    monkeypatch.setattr(runner, "_run_av_sync_check", lambda args, *, playback_sync: passing_check(args))
    monkeypatch.setattr(runner.history_smoke, "run_history_smoke", failing_history)
    monkeypatch.setattr(runner, "collect_latest_artifacts", lambda **kwargs: {})

    args = argparse.Namespace(
        url="ws://127.0.0.1:8005/pipeline/stream",
        http_base_url="http://127.0.0.1:8005",
        output_dir=tmp_path,
        mode="fake",
        audio_backend="null",
        playback_sync=["immediate_audio"],
        wait_for_face_timeout_ms=800,
        history_turn1_wav=None,
        history_turn2_wav=None,
        expect="小张",
        chunk_ms=40,
        timeout_sec=30.0,
        data_latest_dir=None,
    )

    report = await runner.run_demo_acceptance(args)

    assert report["success"] is False
    failed_check = report["checks"]["history_smoke"]
    assert failed_check["success"] is False
    assert failed_check["failure_code"] == "history_smoke_exception"
    assert failed_check["failure_message"] == "history smoke exited"
    assert (tmp_path / "demo_acceptance_report.json").exists()
    assert "history_smoke:history_smoke_exception" in report["failure_reasons"]


@pytest.mark.parametrize(
    ("mode", "history_turn1_wav", "history_turn2_wav", "expected_internal_mode"),
    [
        ("fake", None, None, "mock"),
        ("real", Path("/tmp/turn1.wav"), Path("/tmp/turn2.wav"), "real"),
    ],
)
@pytest.mark.asyncio
async def test_run_demo_acceptance_maps_history_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    history_turn1_wav: Path | None,
    history_turn2_wav: Path | None,
    expected_internal_mode: str,
) -> None:
    import scripts.run_demo_acceptance as runner
    from bionic_head.client.history_smoke import HistorySmokeReport

    async def fake_check_server(http_base_url: str, *, timeout_sec: float):
        return {"health_ok": True, "diagnostics_ok": True}

    async def passing_check(args) -> AcceptanceCheckResult:
        return AcceptanceCheckResult(success=True)

    history_calls: list[dict[str, object]] = []

    async def fake_history(**kwargs):
        history_calls.append(kwargs)
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        report = HistorySmokeReport(
            success=True,
            mode=kwargs["mode"],
            session_id="session-1",
            expected_text=kwargs["expected_text"],
            failure_reasons=[],
            turns=[],
        )
        write_json(output_dir / "history_smoke_report.json", report.to_dict())
        write_json(output_dir / "summary.json", {"success": True})
        return report

    monkeypatch.setattr(runner, "_check_server", fake_check_server)
    monkeypatch.setattr(runner, "_run_scripted_interactive_check", passing_check)
    monkeypatch.setattr(runner, "_run_playback_interrupt_check", passing_check)
    monkeypatch.setattr(runner, "_run_av_sync_check", lambda args, *, playback_sync: passing_check(args))
    monkeypatch.setattr(runner.history_smoke, "run_history_smoke", fake_history)
    monkeypatch.setattr(runner, "collect_latest_artifacts", lambda **kwargs: {})

    args = argparse.Namespace(
        url="ws://127.0.0.1:8005/pipeline/stream",
        http_base_url="http://127.0.0.1:8005",
        output_dir=tmp_path,
        mode=mode,
        audio_backend="null",
        playback_sync=["immediate_audio"],
        wait_for_face_timeout_ms=800,
        history_turn1_wav=history_turn1_wav,
        history_turn2_wav=history_turn2_wav,
        expect="小张",
        chunk_ms=40,
        timeout_sec=30.0,
        data_latest_dir=None,
    )

    report = await runner.run_demo_acceptance(args)

    assert report["success"] is True
    assert history_calls
    assert history_calls[0]["mode"] == expected_internal_mode
