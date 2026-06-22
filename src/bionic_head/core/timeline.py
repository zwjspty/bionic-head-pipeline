from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic_ns
from typing import Iterator
import json

from bionic_head.domain.errors import PipelineException


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Timeline:
    def __init__(self) -> None:
        self._stages: list[dict[str, object]] = []
        self._marks: dict[str, tuple[str, int]] = {}
        self._metrics: list[dict[str, object]] = []

    @contextmanager
    def stage(self, name: str, provider: str) -> Iterator[None]:
        started_at = _utc_now()
        started_ns = monotonic_ns()
        item: dict[str, object] = {
            "name": name,
            "provider": provider,
            "started_at": started_at,
            "finished_at": None,
            "duration_ms": None,
            "status": "running",
            "error_code": None,
        }
        self._stages.append(item)
        try:
            yield
        except BaseException as exc:
            item["status"] = "failed"
            if isinstance(exc, PipelineException):
                item["error_code"] = exc.code.value
            raise
        else:
            item["status"] = "completed"
        finally:
            item["finished_at"] = _utc_now()
            item["duration_ms"] = (monotonic_ns() - started_ns) / 1_000_000.0

    def mark(self, name: str) -> dict[str, object]:
        timestamp = _utc_now()
        self._marks[name] = (timestamp, monotonic_ns())
        return {"name": name, "timestamp": timestamp}

    def metric(self, name: str, start_mark: str, end_mark: str) -> dict[str, object]:
        start = self._marks[start_mark]
        end = self._marks[end_mark]
        metric = {
            "name": name,
            "start_mark": start_mark,
            "end_mark": end_mark,
            "duration_ms": (end[1] - start[1]) / 1_000_000.0,
        }
        self._metrics.append(metric)
        return metric

    def snapshot(self) -> dict[str, object]:
        return {
            "stages": self._stages,
            "marks": {name: timestamp for name, (timestamp, _) in self._marks.items()},
            "metrics": self._metrics,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
