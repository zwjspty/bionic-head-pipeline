from pathlib import Path

import pytest

from bionic_head.core.timeline import Timeline
from bionic_head.domain.errors import ErrorCode, PipelineException


def test_timeline_records_completed_stage() -> None:
    timeline = Timeline()
    with timeline.stage("asr", "mock"):
        pass
    item = timeline.snapshot()["stages"][0]
    assert item["status"] == "completed"
    assert item["duration_ms"] >= 0


def test_timeline_records_failed_stage() -> None:
    timeline = Timeline()
    with pytest.raises(PipelineException):
        with timeline.stage("tts", "mock"):
            raise PipelineException(
                code=ErrorCode.PROVIDER_FAILED,
                stage="tts",
                provider="mock",
                retryable=False,
                message="TTS failed",
            )
    item = timeline.snapshot()["stages"][0]
    assert item["status"] == "failed"
    assert item["error_code"] == "provider_failed"


def test_timeline_marks_and_metrics() -> None:
    timeline = Timeline()
    timeline.mark("start")
    timeline.mark("end")
    metric = timeline.metric("elapsed", "start", "end")
    assert metric["name"] == "elapsed"
    assert metric["duration_ms"] >= 0


def test_timeline_writes_json(tmp_path: Path) -> None:
    timeline = Timeline()
    with timeline.stage("ue5", "mock"):
        pass
    path = tmp_path / "timeline.json"
    timeline.write(path)
    assert path.read_text(encoding="utf-8").startswith("{")
