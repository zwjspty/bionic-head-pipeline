# Task 19 Demo Acceptance Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable local demo acceptance package that runs current scripted/history/interrupt/AV-sync checks, collects artifacts, and writes `demo_acceptance_report.json`.

**Architecture:** Add a small testable acceptance report module under `src/bionic_head/client/demo_acceptance.py`, a focused artifact collector module under `src/bionic_head/client/demo_artifacts.py`, and thin CLI scripts in `scripts/run_demo_acceptance.py` and `scripts/collect_demo_artifacts.py`. Reuse existing smoke functions instead of reimplementing WebSocket protocol loops.

**Tech Stack:** Python 3.10+/3.11-compatible code, asyncio, stdlib `urllib.request` for HTTP JSON checks, existing `websockets`-based smoke scripts, pytest.

## Global Constraints

- Default `fake` mode must not require real microphone, speaker, GPU, Ollama, Piper, EmoTalk, Blender, or real WAV files.
- Do not change `/pipeline/stream`, `/pipeline/audio`, WebSocket events, ASR, TTS, LLM, EmoTalk, UE5 formatter, or backend protocol.
- Do not make grey-head rendering mandatory.
- Existing smoke scripts remain source of truth for protocol behavior.
- All failure paths must write structured failure reasons in `demo_acceptance_report.json`.
- Full pytest must pass before merge.

---

## File map

- Create `src/bionic_head/client/demo_acceptance.py`
  - dataclasses for check results and reports;
  - report aggregation;
  - generated WAV helper.
- Create `src/bionic_head/client/demo_artifacts.py`
  - artifact collection helpers;
  - HTTP JSON helper.
  - latest artifact collection.
- Create `scripts/run_demo_acceptance.py`
  - CLI parser;
  - async orchestration;
  - direct calls to existing smoke functions;
  - writes `demo_acceptance_report.json`;
  - exits non-zero on required failure.
- Create `scripts/collect_demo_artifacts.py`
  - CLI for collecting latest artifacts separately;
  - useful when a user has already run checks manually.
- Create `tests/unit/test_demo_acceptance.py`
  - unit tests for report aggregation, artifact collection, generated WAV, HTTP checks, CLI validation, and monkeypatched runner.
- Create `docs/operations/demo-acceptance.md`
  - how to run mock and real acceptance.
- Create `data/demo/README.md`
  - explains where demo acceptance artifacts should be stored and what is intentionally not committed.

---

## Task 1: Acceptance report core

**Files:**
- Create: `src/bionic_head/client/demo_acceptance.py`
- Test: `tests/unit/test_demo_acceptance.py`

**Interfaces:**
- Produces:
  - `AcceptanceCheckResult`
  - `DemoAcceptanceReport`
  - `build_demo_acceptance_report(...) -> DemoAcceptanceReport`
  - `write_json(path: Path, payload: Mapping[str, object] | DemoAcceptanceReport) -> None`
  - `write_demo_input_wav(path: Path, sample_rate: int = 16000, duration_seconds: float = 1.0) -> Path`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_demo_acceptance.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_demo_acceptance.py -q
```

Expected: import failure for `bionic_head.client.demo_acceptance`.

- [ ] **Step 3: Implement core module**

Create `src/bionic_head/client/demo_acceptance.py`:

```python
from __future__ import annotations

import json
import math
import wave
from array import array
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AcceptanceCheckResult:
    success: bool
    failure_code: str | None = None
    failure_message: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": self.success,
            "artifacts": self.artifacts,
        }
        if self.failure_code is not None:
            payload["failure_code"] = self.failure_code
        if self.failure_message is not None:
            payload["failure_message"] = self.failure_message
        if self.metrics:
            payload["metrics"] = self.metrics
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        return payload


@dataclass
class DemoAcceptanceReport:
    success: bool
    generated_at: str
    mode: str
    server: dict[str, Any]
    checks: dict[str, AcceptanceCheckResult]
    artifacts: dict[str, str]
    failure_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "generated_at": self.generated_at,
            "mode": self.mode,
            "server": self.server,
            "checks": {name: check.to_dict() for name, check in self.checks.items()},
            "artifacts": self.artifacts,
            "failure_reasons": self.failure_reasons,
        }


def build_demo_acceptance_report(
    *,
    mode: str,
    server: Mapping[str, Any],
    checks: Mapping[str, AcceptanceCheckResult],
    artifacts: Mapping[str, str],
) -> DemoAcceptanceReport:
    failure_reasons: list[str] = []
    if not bool(server.get("health_ok")):
        failure_reasons.append("server:server_health_unreachable")
    if not bool(server.get("diagnostics_ok")):
        failure_reasons.append("server:server_diagnostics_failed")
    for name, check in checks.items():
        if not check.success:
            failure_reasons.append(f"{name}:{check.failure_code or 'check_failed'}")

    return DemoAcceptanceReport(
        success=not failure_reasons,
        generated_at=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        server=dict(server),
        checks=dict(checks),
        artifacts=dict(artifacts),
        failure_reasons=failure_reasons,
    )


def write_json(path: Path, payload: Mapping[str, Any] | DemoAcceptanceReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, DemoAcceptanceReport):
        payload = payload.to_dict()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_demo_input_wav(
    path: Path,
    *,
    sample_rate: int = 16000,
    duration_seconds: float = 1.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = int(sample_rate * duration_seconds)
    samples = array(
        "h",
        (
            int(2500 * math.sin(2 * math.pi * 220 * index / sample_rate))
            for index in range(total_samples)
        ),
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return path
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_demo_acceptance.py -q
```

Expected: tests in Task 1 pass.

- [ ] **Step 5: Commit**

```bash
git add src/bionic_head/client/demo_acceptance.py tests/unit/test_demo_acceptance.py
git commit -m "feat: add demo acceptance report builder"
```

---

## Task 2: HTTP checks and artifact collection CLI

**Files:**
- Create: `src/bionic_head/client/demo_artifacts.py`
- Create: `scripts/collect_demo_artifacts.py`
- Test: `tests/unit/test_demo_acceptance.py`

**Interfaces:**
- Consumes: `write_json` from `demo_acceptance`
- Produces:
  - `collect_existing_artifacts(output_dir: Path, names: Mapping[str, Path]) -> dict[str, str]`
  - `http_get_json(url: str, timeout_sec: float = 5.0) -> tuple[bool, object | None, str | None]`
  - `collect_latest_artifacts(output_dir: Path, http_base_url: str | None, data_latest_dir: Path | None) -> dict[str, str]`
  - `scripts.collect_demo_artifacts.build_parser()`

- [ ] **Step 1: Write failing tests**

Append:

```python
from types import SimpleNamespace
import urllib.error

import pytest

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
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_demo_acceptance.py -q
```

Expected: missing `http_get_json`, `collect_latest_artifacts`, or script import failure.

- [ ] **Step 3: Implement HTTP and collector helpers**

Create `src/bionic_head/client/demo_artifacts.py`:

```python
from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from bionic_head.client.demo_acceptance import write_json


def http_get_json(url: str, *, timeout_sec: float = 5.0) -> tuple[bool, object | None, str | None]:
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return True, payload, None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return False, None, str(exc)


def collect_latest_artifacts(
    *,
    output_dir: Path,
    http_base_url: str | None,
    data_latest_dir: Path | None,
    timeout_sec: float = 5.0,
) -> dict[str, str]:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    if data_latest_dir is not None:
        local_sources = {
            "latest_pipeline": data_latest_dir / "latest_pipeline.json",
            "latest_ue5": data_latest_dir / "latest_ue5_blendshape.json",
        }
        for name, source in local_sources.items():
            if source.exists():
                destination = artifact_dir / source.name
                shutil.copy2(source, destination)
                artifacts[name] = destination.relative_to(output_dir).as_posix()

    if http_base_url:
        endpoints = {
            "latest_pipeline_http": "/pipeline/latest",
            "latest_ue5_http": "/ue5/latest",
        }
        for name, path in endpoints.items():
            ok, payload, _ = http_get_json(http_base_url.rstrip("/") + path, timeout_sec=timeout_sec)
            if ok and payload is not None:
                destination = artifact_dir / f"{name}.json"
                write_json(destination, {"payload": payload})
                artifacts[name] = destination.relative_to(output_dir).as_posix()
    return artifacts


def collect_existing_artifacts(output_dir: Path, names: Mapping[str, Path]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for name, path in names.items():
        if path.exists() and path.is_file():
            artifacts[name] = path.relative_to(output_dir).as_posix()
    return artifacts
```

Create `scripts/collect_demo_artifacts.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bionic_head.client.demo_artifacts import collect_latest_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect demo acceptance artifacts.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--http-base-url")
    parser.add_argument("--data-latest-dir", type=Path, default=Path("data/latest"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = collect_latest_artifacts(
        output_dir=args.output_dir,
        http_base_url=args.http_base_url,
        data_latest_dir=args.data_latest_dir,
    )
    print(json.dumps({"artifacts": artifacts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_demo_acceptance.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/bionic_head/client/demo_artifacts.py scripts/collect_demo_artifacts.py tests/unit/test_demo_acceptance.py
git commit -m "feat: collect demo acceptance artifacts"
```

---

## Task 3: Acceptance runner CLI skeleton and validation

**Files:**
- Create: `scripts/run_demo_acceptance.py`
- Modify: `tests/unit/test_demo_acceptance.py`

**Interfaces:**
- Consumes: `build_demo_acceptance_report`, `write_json`, `http_get_json`
- Produces:
  - `scripts.run_demo_acceptance.build_parser()`
  - `scripts.run_demo_acceptance.run_demo_acceptance(args: argparse.Namespace) -> dict[str, object]`

- [ ] **Step 1: Write failing tests**

Append:

```python
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
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_demo_acceptance.py -q
```

Expected: `scripts.run_demo_acceptance` import failure.

- [ ] **Step 3: Implement parser and argument validation**

Create `scripts/run_demo_acceptance.py`:

```python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bionic_head.client.demo_acceptance import (
    AcceptanceCheckResult,
    build_demo_acceptance_report,
    write_json,
)
from bionic_head.client.demo_artifacts import collect_latest_artifacts, http_get_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local demo acceptance checks.")
    parser.add_argument("--url", required=True, help="WebSocket URL, e.g. ws://127.0.0.1:8005/pipeline/stream")
    parser.add_argument("--http-base-url", required=True, help="HTTP base URL, e.g. http://127.0.0.1:8005")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["fake", "real"], default="fake")
    parser.add_argument("--audio-backend", choices=["null", "sounddevice"], default="null")
    parser.add_argument(
        "--playback-sync",
        nargs="+",
        choices=["immediate_audio", "wait_for_face"],
        default=["immediate_audio", "wait_for_face"],
    )
    parser.add_argument("--wait-for-face-timeout-ms", type=int, default=800)
    parser.add_argument("--history-turn1-wav", type=Path)
    parser.add_argument("--history-turn2-wav", type=Path)
    parser.add_argument("--expect", default="小张")
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--data-latest-dir", type=Path, default=Path("data/latest"))
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "real" and (args.history_turn1_wav is None or args.history_turn2_wav is None):
        raise SystemExit("real mode requires --history-turn1-wav and --history-turn2-wav")


async def run_demo_acceptance(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    server = await _check_server(args.http_base_url, timeout_sec=args.timeout_sec)
    checks: dict[str, AcceptanceCheckResult] = {}
    artifacts = collect_latest_artifacts(
        output_dir=args.output_dir,
        http_base_url=args.http_base_url,
        data_latest_dir=args.data_latest_dir,
        timeout_sec=args.timeout_sec,
    )
    report = build_demo_acceptance_report(
        mode=args.mode,
        server=server,
        checks=checks,
        artifacts=artifacts,
    )
    write_json(args.output_dir / "demo_acceptance_report.json", report)
    return report


async def _check_server(http_base_url: str, *, timeout_sec: float) -> dict[str, object]:
    health_ok, health_payload, health_error = http_get_json(
        http_base_url.rstrip("/") + "/health",
        timeout_sec=timeout_sec,
    )
    diagnostics_ok, diagnostics_payload, diagnostics_error = http_get_json(
        http_base_url.rstrip("/") + "/diagnostics",
        timeout_sec=timeout_sec,
    )
    return {
        "health_ok": bool(health_ok and isinstance(health_payload, dict) and health_payload.get("status") == "ok"),
        "health": health_payload,
        "health_error": health_error,
        "diagnostics_ok": diagnostics_ok,
        "diagnostics": diagnostics_payload,
        "diagnostics_error": diagnostics_error,
    }


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(run_demo_acceptance(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_demo_acceptance.py -q
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_demo_acceptance.py tests/unit/test_demo_acceptance.py
git commit -m "feat: add demo acceptance runner cli"
```

---

## Task 4: Wire smoke checks into runner

**Files:**
- Modify: `scripts/run_demo_acceptance.py`
- Modify: `tests/unit/test_demo_acceptance.py`

**Interfaces:**
- Consumes:
  - `scripts.interactive_demo_client.run_scripted_demo`
  - `scripts.history_smoke.run_history_smoke`
  - `scripts.local_demo_client.run_local_demo`
  - `write_demo_input_wav`
- Produces:
  - required checks in `demo_acceptance_report.json`

- [ ] **Step 1: Write failing tests**

Append:

```python
import argparse


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
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_demo_acceptance.py::test_run_demo_acceptance_aggregates_fake_checks -q
```

Expected: missing smoke integration.

- [ ] **Step 3: Import smoke modules and implement check runners**

Modify `scripts/run_demo_acceptance.py`:

```python
import scripts.history_smoke as history_smoke
import scripts.interactive_demo_client as interactive_demo_client
import scripts.local_demo_client as local_demo_client
from bionic_head.client.demo_acceptance import write_demo_input_wav
```

In `run_demo_acceptance`, replace `checks = {}` with:

```python
    checks: dict[str, AcceptanceCheckResult] = {}
    checks["scripted_interactive_smoke"] = await _run_scripted_interactive_check(args)
    checks["history_smoke"] = await _run_history_check(args)
    checks["playback_interrupt_smoke"] = await _run_playback_interrupt_check(args)
    for strategy in args.playback_sync:
        checks[f"av_sync_{strategy}"] = await _run_av_sync_check(args, playback_sync=strategy)
```

Add helpers:

```python
async def _run_scripted_interactive_check(args: argparse.Namespace) -> AcceptanceCheckResult:
    output_dir = args.output_dir / "scripted_interactive_smoke"
    try:
        await interactive_demo_client.run_scripted_demo(
            url=args.url,
            output_dir=output_dir,
            scripted_turns=2,
            scripted_cancel_after_ms=300,
            chunk_ms=args.chunk_ms,
            sample_rate=16000,
            audio_backend=args.audio_backend,
            playback_sync="immediate_audio",
            wait_for_face_timeout_ms=args.wait_for_face_timeout_ms,
            wait_timeout_sec=args.timeout_sec,
        )
        report = _read_json(output_dir / "interaction_report.json")
        reasons: list[str] = []
        if not bool(report.get("success")):
            reasons.append("scripted_interactive_failed")
        if int(report.get("old_generation_audio_play_count", 0) or 0) != 0:
            reasons.append("old_generation_audio_played")
        if int(report.get("old_generation_face_display_count", 0) or 0) != 0:
            reasons.append("old_generation_face_displayed")
        return AcceptanceCheckResult(
            success=not reasons,
            failure_code=reasons[0] if reasons else None,
            failure_message="; ".join(reasons) if reasons else None,
            artifacts=collect_existing_artifacts(
                args.output_dir,
                {
                    "interaction_report": output_dir / "interaction_report.json",
                    "summary": output_dir / "summary.json",
                },
            ),
            metrics={key: report.get(key) for key in ("turn_count", "completed_turn_count", "cancelled_turn_count")},
        )
    except Exception as exc:  # noqa: BLE001
        return AcceptanceCheckResult(False, failure_code="scripted_interactive_exception", error_message=str(exc))


async def _run_history_check(args: argparse.Namespace) -> AcceptanceCheckResult:
    output_dir = args.output_dir / "history_smoke"
    try:
        report = await history_smoke.run_history_smoke(
            url=args.url,
            output_dir=output_dir,
            mode="mock" if args.mode == "fake" else "real",
            turn1_wav=args.history_turn1_wav,
            turn2_wav=args.history_turn2_wav,
            expected_text=args.expect,
            chunk_ms=args.chunk_ms,
            timeout_sec=args.timeout_sec,
        )
        return AcceptanceCheckResult(
            success=report.success,
            failure_code=report.failure_reasons[0] if report.failure_reasons else None,
            failure_message="; ".join(report.failure_reasons) if report.failure_reasons else None,
            artifacts=collect_existing_artifacts(
                args.output_dir,
                {
                    "history_smoke_report": output_dir / "history_smoke_report.json",
                    "summary": output_dir / "summary.json",
                    "events": output_dir / "events.jsonl",
                },
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return AcceptanceCheckResult(False, failure_code="history_smoke_exception", error_message=str(exc))


async def _run_playback_interrupt_check(args: argparse.Namespace) -> AcceptanceCheckResult:
    output_dir = args.output_dir / "playback_interrupt_smoke"
    wav_path = write_demo_input_wav(args.output_dir / "generated-input.wav")
    try:
        terminal = await local_demo_client.run_local_demo(
            url=args.url,
            wav_path=wav_path,
            output_dir=output_dir,
            chunk_ms=args.chunk_ms,
            play_audio=False,
            cancel_after_ms=300,
            playback_sync="immediate_audio",
            wait_for_face_timeout_ms=args.wait_for_face_timeout_ms,
        )
        summary = _read_json(output_dir / "summary.json")
        evidence = terminal == "server.turn.cancelled" or (
            int(summary.get("playback_stop_count", 0) or 0) >= 1
            and summary.get("client_interrupt_sent_ms") is not None
        )
        return AcceptanceCheckResult(
            success=evidence,
            failure_code=None if evidence else "playback_interrupt_failed",
            failure_message=None if evidence else "Playback interrupt did not produce cancel or playback-stop evidence.",
            artifacts=collect_existing_artifacts(
                args.output_dir,
                {
                    "summary": output_dir / "summary.json",
                    "client_playback_metrics": output_dir / "client_playback_metrics.json",
                },
            ),
            metrics={
                "terminal_event": terminal,
                "playback_stop_count": summary.get("playback_stop_count"),
                "client_interrupt_sent_ms": summary.get("client_interrupt_sent_ms"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return AcceptanceCheckResult(False, failure_code="playback_interrupt_exception", error_message=str(exc))


async def _run_av_sync_check(args: argparse.Namespace, *, playback_sync: str) -> AcceptanceCheckResult:
    output_dir = args.output_dir / f"av_sync_{playback_sync}"
    try:
        await interactive_demo_client.run_scripted_demo(
            url=args.url,
            output_dir=output_dir,
            scripted_turns=1,
            scripted_cancel_after_ms=0,
            chunk_ms=args.chunk_ms,
            sample_rate=16000,
            audio_backend=args.audio_backend,
            playback_sync=playback_sync,
            wait_for_face_timeout_ms=args.wait_for_face_timeout_ms,
            wait_timeout_sec=args.timeout_sec,
        )
        report = _read_json(output_dir / "interaction_report.json")
        reasons: list[str] = []
        if not bool(report.get("success")):
            reasons.append("av_sync_failed")
        if report.get("playback_sync_strategy") != playback_sync:
            reasons.append("av_sync_strategy_mismatch")
        if report.get("client_audio_face_offset_ms") is None:
            reasons.append("av_sync_offset_missing")
        if playback_sync == "wait_for_face":
            if report.get("client_audio_wait_for_face_ms") is None:
                reasons.append("av_sync_wait_missing")
            if report.get("client_audio_wait_for_face_timeout") is True:
                reasons.append("av_sync_wait_for_face_timeout")
        return AcceptanceCheckResult(
            success=not reasons,
            failure_code=reasons[0] if reasons else None,
            failure_message="; ".join(reasons) if reasons else None,
            artifacts=collect_existing_artifacts(
                args.output_dir,
                {
                    "interaction_report": output_dir / "interaction_report.json",
                    "summary": output_dir / "summary.json",
                },
            ),
            metrics={
                "playback_sync_strategy": report.get("playback_sync_strategy"),
                "client_audio_face_offset_ms": report.get("client_audio_face_offset_ms"),
                "client_audio_wait_for_face_ms": report.get("client_audio_wait_for_face_ms"),
                "client_audio_wait_for_face_timeout": report.get("client_audio_wait_for_face_timeout"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return AcceptanceCheckResult(False, failure_code=f"av_sync_{playback_sync}_exception", error_message=str(exc))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_demo_acceptance.py -q
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_demo_acceptance.py tests/unit/test_demo_acceptance.py
git commit -m "feat: run demo acceptance smoke checks"
```

---

## Task 5: CLI smoke and documentation

**Files:**
- Create: `docs/operations/demo-acceptance.md`
- Create: `data/demo/README.md`
- Modify: `tests/unit/test_demo_acceptance.py`

**Interfaces:**
- Consumes: runner and collector CLI scripts
- Produces: documented commands and durable artifact guidance

- [ ] **Step 1: Add CLI help tests**

Append:

```python
import subprocess
import sys


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
```

- [ ] **Step 2: Run help tests and verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_demo_acceptance.py::test_run_demo_acceptance_help_runs_by_path tests/unit/test_demo_acceptance.py::test_collect_demo_artifacts_help_runs_by_path -q
```

- [ ] **Step 3: Write operations documentation**

Create `docs/operations/demo-acceptance.md`:

```markdown
# Demo acceptance package

Task 19 provides a repeatable local demo acceptance command.

## Start a server

Mock provider example:

```bash
cd /home/user/code/端到端
source .venv/bin/activate

PYTHONPATH=src BIONIC_CONFIG=config/mock.json \
  .venv/bin/uvicorn bionic_head.api.app:create_app \
  --factory --host 127.0.0.1 --port 8005
```

## Fake acceptance

```bash
.venv/bin/python scripts/run_demo_acceptance.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --http-base-url http://127.0.0.1:8005 \
  --output-dir /tmp/bionic-demo-acceptance \
  --mode fake \
  --audio-backend null \
  --playback-sync immediate_audio wait_for_face
```

Expected output:

```text
demo_acceptance_report.json
success: true
checks.scripted_interactive_smoke.success: true
checks.history_smoke.success: true
checks.playback_interrupt_smoke.success: true
checks.av_sync_immediate_audio.success: true
checks.av_sync_wait_for_face.success: true
```

## Real acceptance

```bash
.venv/bin/python scripts/run_demo_acceptance.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --http-base-url http://127.0.0.1:8005 \
  --output-dir /tmp/bionic-demo-acceptance-real \
  --mode real \
  --history-turn1-wav /path/to/wo-jiao-xiaozhang.wav \
  --history-turn2-wav /path/to/wo-jiao-shenme.wav \
  --expect 小张
```

## Report and artifacts

The runner writes:

- `demo_acceptance_report.json`
- `generated-input.wav`
- `scripted_interactive_smoke/`
- `history_smoke/`
- `playback_interrupt_smoke/`
- `av_sync_immediate_audio/`
- `av_sync_wait_for_face/`
- `artifacts/`

Inspect `failure_reasons` first when a run fails.

## Non-goals

This acceptance package does not start the server, install providers, connect UE5, add AEC, or run Blender rendering by default.
```

Create `data/demo/README.md`:

```markdown
# Demo data

This directory is reserved for small checked-in demo documentation and placeholder notes.

Generated acceptance outputs should normally go to `/tmp/...` or another ignored local directory.

Do not commit large generated files such as WAV chunks, MP4 renders, `events.jsonl`, or full acceptance output folders unless a future task explicitly requests a small fixture.
```

- [ ] **Step 4: Run full tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: full suite passes.

- [ ] **Step 5: Commit**

```bash
git add docs/operations/demo-acceptance.md data/demo/README.md tests/unit/test_demo_acceptance.py
git commit -m "docs: document demo acceptance package"
```

---

## Completion checklist

- [ ] `scripts/run_demo_acceptance.py --help` works.
- [ ] `scripts/collect_demo_artifacts.py --help` works.
- [ ] Fake mode does not require real microphone, speaker, GPU, Ollama, Piper, EmoTalk, or real WAV files.
- [ ] `demo_acceptance_report.json` includes server, checks, artifacts, and failure reasons.
- [ ] Checks include scripted interactive, history, playback interrupt, immediate_audio, and wait_for_face.
- [ ] Failure paths are represented as structured check failures.
- [ ] Docs explain mock and real acceptance commands.
- [ ] Full pytest passes.
