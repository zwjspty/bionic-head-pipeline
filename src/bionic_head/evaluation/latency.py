from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


STREAM_MARK_METRICS = {
    "audio_end_to_asr_final_ms": ("audio_end", "asr_final"),
    "audio_end_to_llm_first_token_ms": ("audio_end", "llm_first_token"),
    "audio_end_to_first_tts_ready_ms": ("audio_end", "first_tts_ready"),
    "audio_end_to_first_face_ready_ms": ("audio_end", "first_face_ready"),
    "audio_end_to_first_segment_ready_ms": ("audio_end", "first_segment_ready"),
    "total_turn_duration_ms": ("audio_end", "completed"),
}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    collected = list(values)
    if not collected:
        return {
            "count": 0,
            "p50_ms": None,
            "p90_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(collected),
        "p50_ms": percentile(collected, 0.5),
        "p90_ms": percentile(collected, 0.9),
        "min_ms": min(collected),
        "max_ms": max(collected),
    }


def build_latency_report(
    runs: list[dict[str, object]],
    *,
    source_wav: Path,
    mode: str,
) -> dict[str, object]:
    success_runs = [run for run in runs if run.get("success") is True]
    failure_codes = Counter(
        str(run.get("failure_code", "unknown"))
        for run in runs
        if run.get("success") is not True
    )
    values_by_metric: dict[str, list[float]] = {
        name: [] for name in STREAM_MARK_METRICS
    }
    providers: dict[str, object] = {}

    for run in success_runs:
        if isinstance(run.get("providers"), dict):
            providers.update(run["providers"])  # type: ignore[arg-type]
        for name, value in extract_metrics(run).items():
            if name in values_by_metric:
                values_by_metric[name].append(value)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": mode,
        "source_wav": str(source_wav),
        "run_count": len(runs),
        "success_count": len(success_runs),
        "failure_count": len(runs) - len(success_runs),
        "failure_codes": dict(failure_codes),
        "providers": providers,
        "metrics": {
            name: summarize(values)
            for name, values in values_by_metric.items()
        },
    }


def extract_metrics(run: dict[str, object]) -> dict[str, float]:
    direct = run.get("metrics")
    if isinstance(direct, dict):
        return {
            str(name): float(value)
            for name, value in direct.items()
            if isinstance(value, (int, float))
        }

    timeline = run.get("timeline")
    if not isinstance(timeline, dict):
        return {}
    marks = timeline.get("marks")
    if not isinstance(marks, dict):
        return {}

    metrics: dict[str, float] = {}
    for name, (start_mark, end_mark) in STREAM_MARK_METRICS.items():
        if start_mark not in marks or end_mark not in marks:
            continue
        metrics[name] = _delta_ms(str(marks[start_mark]), str(marks[end_mark]))
    return metrics


def _delta_ms(start: str, end: str) -> float:
    start_dt = _parse_timestamp(start)
    end_dt = _parse_timestamp(end)
    return (end_dt - start_dt).total_seconds() * 1000.0


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
