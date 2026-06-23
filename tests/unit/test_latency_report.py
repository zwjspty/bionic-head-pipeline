from pathlib import Path

from bionic_head.evaluation.latency import build_latency_report, extract_metrics, summarize


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
                "stages": [
                    {"name": "asr", "provider": "mock", "duration_ms": 10.0},
                    {"name": "llm", "provider": "mock", "duration_ms": 20.0},
                    {"name": "tts", "provider": "mock", "duration_ms": 30.0},
                    {"name": "audio2face", "provider": "mock", "duration_ms": 40.0},
                    {"name": "ue5", "provider": "mock", "duration_ms": 5.0},
                ],
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
    assert report["metrics"]["asr_ms"]["p50_ms"] == 10.0
    assert report["metrics"]["llm_total_ms"]["p50_ms"] == 20.0
    assert report["metrics"]["tts_total_ms"]["p50_ms"] == 30.0
    assert report["metrics"]["face_total_ms"]["p50_ms"] == 40.0
    assert report["metrics"]["ue5_format_ms"]["p50_ms"] == 5.0
    assert report["metrics"]["llm_ttft_ms"]["p50_ms"] == 100
    assert report["metrics"]["tts_first_audio_ms"]["p50_ms"] == 100
    assert report["metrics"]["face_first_chunk_ms"]["p50_ms"] == 100
    assert report["metrics"]["e2e_first_audible_ms"]["p50_ms"] == 300
    assert report["metrics"]["e2e_first_visible_face_ms"]["p50_ms"] == 400
    assert report["metrics"]["audio_end_to_asr_final_ms"]["p50_ms"] == 100
    assert report["metrics"]["total_turn_duration_ms"]["p90_ms"] == 600


def test_extract_metrics_merges_direct_wall_clock_with_timeline_metrics() -> None:
    metrics = extract_metrics(
        {
            "metrics": {"total_turn_duration_ms": 1234.0},
            "timeline": {
                "stages": [
                    {"name": "asr", "duration_ms": 11.0},
                    {"name": "tts", "duration_ms": 33.0},
                ],
                "marks": {
                    "asr_final": "2026-06-22T00:00:00.100Z",
                    "llm_first_token": "2026-06-22T00:00:00.250Z",
                    "first_tts_ready": "2026-06-22T00:00:00.450Z",
                },
            },
        }
    )

    assert metrics["total_turn_duration_ms"] == 1234.0
    assert metrics["asr_ms"] == 11.0
    assert metrics["tts_total_ms"] == 33.0
    assert metrics["llm_ttft_ms"] == 150
    assert metrics["tts_first_audio_ms"] == 200
