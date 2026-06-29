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
