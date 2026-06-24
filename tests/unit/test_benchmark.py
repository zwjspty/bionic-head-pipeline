import os
import subprocess
import sys
import wave
from array import array
from pathlib import Path

from scripts.benchmark import stream_metrics_from_summary


def test_benchmark_script_help_runs_when_executed_by_path() -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [sys.executable, "scripts/benchmark.py", "--help"],
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_benchmark_stream_mode_imports_client_when_executed_by_path(tmp_path: Path) -> None:
    wav_path = tmp_path / "input.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(array("h", [0] * 320).tobytes())

    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark.py",
            "--mode",
            "stream",
            "--ws-url",
            "ws://127.0.0.1:9/pipeline/stream",
            "--wav",
            str(wav_path),
            "--runs",
            "1",
            "--output",
            str(tmp_path / "report.json"),
        ],
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "ModuleNotFoundError" not in result.stderr
    assert "No module named 'scripts'" not in result.stderr


def test_stream_metrics_from_summary_uses_first_tts_binary_and_ue5_event() -> None:
    metrics = stream_metrics_from_summary(
        {
            "first_tts_binary_ms": 321.0,
            "event_first_ms": {
                "server.tts.audio": 300.0,
                "server.face.frames": 450.0,
                "server.ue5.frames": 480.0,
                "server.pipeline.done": 900.0,
            },
        },
        wall_ms=1000.0,
    )

    assert metrics["total_turn_duration_ms"] == 1000.0
    assert metrics["tts_first_audio_ms"] == 321.0
    assert metrics["e2e_first_audible_ms"] == 321.0
    assert metrics["face_first_chunk_ms"] == 450.0
    assert metrics["e2e_first_visible_face_ms"] == 480.0


def test_stream_metrics_from_summary_falls_back_to_tts_metadata_time() -> None:
    metrics = stream_metrics_from_summary(
        {
            "first_tts_binary_ms": None,
            "event_first_ms": {
                "server.tts.audio": 300.0,
            },
        },
        wall_ms=1000.0,
    )

    assert metrics["tts_first_audio_ms"] == 300.0
    assert metrics["e2e_first_audible_ms"] == 300.0


def test_stream_metrics_from_summary_uses_ue5_as_face_fallback() -> None:
    metrics = stream_metrics_from_summary(
        {
            "event_first_ms": {
                "server.ue5.frames": 480.0,
            },
        },
        wall_ms=1000.0,
    )

    assert metrics["face_first_chunk_ms"] == 480.0
    assert metrics["e2e_first_visible_face_ms"] == 480.0


def test_stream_metrics_from_summary_extracts_playback_stop_latency() -> None:
    metrics = stream_metrics_from_summary(
        {
            "event_first_ms": {
                "server.playback.stop": 88.0,
            },
        },
        wall_ms=1000.0,
    )

    assert metrics["interrupt_to_playback_stop_ms"] == 88.0


def test_stream_metrics_from_summary_extracts_segment_face_timing_and_stale_counts() -> None:
    metrics = stream_metrics_from_summary(
        {
            "old_turn_face_leak_count": 0,
            "stale_face_drop_count": 2,
            "segments": {
                "chunk-0001": {
                    "tts_audio_event_ms": 100.0,
                    "tts_binary_ms": 125.0,
                    "face_total_ms": 410.0,
                    "ue5_first_frame_after_tts_ms": 430.0,
                    "ue5_first_frame_ms": 530.0,
                },
                "chunk-0002": {
                    "tts_audio_event_ms": 200.0,
                    "tts_binary_ms": 225.0,
                    "face_total_ms": 390.0,
                    "ue5_first_frame_after_tts_ms": 405.0,
                    "ue5_first_frame_ms": 605.0,
                },
            },
        },
        wall_ms=1000.0,
    )

    assert metrics["tts_audio_ready_ms"] == 100.0
    assert metrics["face_total_ms"] == 410.0
    assert metrics["ue5_first_frame_after_tts_ms"] == 430.0
    assert metrics["e2e_first_visible_face_ms"] == 530.0
    assert metrics["old_turn_face_leak_count"] == 0.0
    assert metrics["stale_face_drop_count"] == 2.0


def test_stream_metrics_from_summary_extracts_face_stitch_metrics() -> None:
    metrics = stream_metrics_from_summary(
        {
            "segments": {
                "chunk-0001": {
                    "tts_audio_event_ms": 100.0,
                    "face_stitch_enabled": True,
                    "face_stitch_applied": True,
                    "face_stitch_reset": False,
                    "face_stitch_overlap_frames": 8.0,
                    "face_stitch_actual_overlap_frames": 5.0,
                    "face_boundary_delta_before": 0.4,
                    "face_boundary_delta_after": 0.1,
                }
            }
        },
        wall_ms=1000.0,
    )

    assert metrics["face_stitch_overlap_frames"] == 8.0
    assert metrics["face_stitch_actual_overlap_frames"] == 5.0
    assert metrics["face_boundary_delta_before"] == 0.4
    assert metrics["face_boundary_delta_after"] == 0.1
    assert metrics["face_stitch_applied_count"] == 1.0
    assert metrics["face_stitch_reset_count"] == 0.0
