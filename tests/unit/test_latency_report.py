from pathlib import Path

from bionic_head.evaluation.latency import build_latency_report, summarize


def test_summary_reports_nearest_rank_p50_p90() -> None:
    report = summarize([100, 200, 300, 400, 500])

    assert report == {
        "count": 5,
        "p50_ms": 300,
        "p90_ms": 500,
        "min_ms": 100,
        "max_ms": 500,
    }


def test_report_uses_only_successful_runs_with_present_metrics() -> None:
    runs = [
        {
            "success": True,
            "providers": {"asr": "mock"},
            "timeline": {
                "marks": {
                    "audio_end": "2026-06-22T00:00:00.000Z",
                    "asr_final": "2026-06-22T00:00:00.100Z",
                    "llm_first_token": "2026-06-22T00:00:00.200Z",
                    "first_tts_ready": "2026-06-22T00:00:00.300Z",
                    "first_face_ready": "2026-06-22T00:00:00.400Z",
                    "first_segment_ready": "2026-06-22T00:00:00.500Z",
                    "completed": "2026-06-22T00:00:00.600Z",
                }
            },
        },
        {"success": False, "failure_code": "provider_failed", "timeline": {"marks": {}}},
    ]

    report = build_latency_report(
        runs,
        source_wav=Path("/tmp/input.wav"),
        mode="stream",
    )

    assert report["run_count"] == 2
    assert report["success_count"] == 1
    assert report["failure_count"] == 1
    assert report["failure_codes"] == {"provider_failed": 1}
    assert report["providers"] == {"asr": "mock"}
    assert report["metrics"]["audio_end_to_asr_final_ms"]["p50_ms"] == 100
    assert report["metrics"]["total_turn_duration_ms"]["p90_ms"] == 600
