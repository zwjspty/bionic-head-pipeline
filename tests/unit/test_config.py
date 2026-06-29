from pathlib import Path

import pytest
from pydantic import ValidationError

from bionic_head.config import AppSettings, load_settings


def test_load_mock_settings() -> None:
    settings = load_settings(Path("config/mock.json"))
    assert settings.stream.input_sample_rate == 16000
    assert settings.adapters.audio2face.provider == "mock"
    assert settings.server.max_active_sessions == 1
    assert settings.providers.ollama.model == "qwen2.5:3b"
    assert settings.vad.engine == "rms"
    assert settings.vad.interrupt_min_speech_ms == 80
    assert settings.vad.interrupt_rms_threshold == pytest.approx(0.02)
    assert settings.face_stitching.enabled is True
    assert settings.face_stitching.overlap_frames == 8
    assert settings.face_stitching.reset_on_new_turn is True
    assert settings.face_stitching.record_boundary_metrics is True
    assert settings.eye_continuity.enabled is True
    assert settings.eye_continuity.eye_smooth_channel_indices == []
    assert settings.eye_continuity.blink_enabled is False
    assert settings.eye_continuity.blink_channel_indices == []
    assert settings.eye_continuity.overlap_frames == 6
    assert settings.eye_continuity.record_boundary_metrics is True
    assert settings.eye_continuity.blink_interval_min_sec == pytest.approx(2.5)
    assert settings.eye_continuity.blink_interval_max_sec == pytest.approx(6.0)
    assert settings.eye_continuity.blink_duration_frames == 5
    assert settings.eye_continuity.blink_strength == pytest.approx(1.0)
    assert settings.eye_continuity.seed == 42
    assert settings.eye_continuity.reset_blink_on_new_turn is False
    assert settings.history.enabled is True
    assert settings.history.max_turn_pairs == 6
    assert settings.history.max_chars == 3000


def test_load_real_example_settings() -> None:
    settings = load_settings(Path("config/real.example.json"))
    assert settings.adapters.asr.provider == "faster-whisper"
    assert settings.adapters.llm.provider == "ollama"
    assert settings.adapters.tts.provider == "piper"
    assert settings.adapters.audio2face.provider == "morpheus"
    assert settings.adapters.ue5.provider == "morpheus-raw"
    assert settings.providers.faster_whisper.model == "base"
    assert settings.providers.faster_whisper.device == "cpu"
    assert settings.providers.faster_whisper.compute_type == "int8"
    assert settings.providers.faster_whisper.language == "zh"
    assert str(settings.providers.ollama.base_url) == "http://127.0.0.1:11434/"
    assert settings.providers.ollama.model == "qwen2.5:3b"
    assert settings.providers.ollama.keep_alive == "30m"
    assert settings.providers.ollama.num_ctx == 2048
    assert settings.providers.ollama.num_predict == 96
    assert settings.providers.ollama.temperature == pytest.approx(0.3)
    assert settings.providers.ollama.prewarm is True
    assert settings.providers.piper.executable == ""
    assert settings.providers.piper.runtime == "python"
    assert settings.providers.piper.args == [
        "--model",
        "{model_path}",
        "--output_file",
        "{output_path}",
    ]
    assert settings.providers.piper.model_path is None
    assert settings.providers.morpheus.executable == "conda"
    assert settings.providers.morpheus.args == [
        "run",
        "-n",
        "lyyMor",
        "",
        "--input",
        "{input_path}",
        "--output-dir",
        "{output_dir}",
    ]
    assert settings.providers.morpheus.cwd == Path(
        "/home/hailab/liuyiyu/head-project/Morpheus-Software"
    )
    assert settings.providers.morpheus.timeout_seconds == 300
    assert settings.vad.engine == "rms"
    assert settings.vad.interrupt_min_speech_ms == 80
    assert settings.vad.interrupt_rms_threshold == pytest.approx(0.02)


def test_load_emotalk_example_settings() -> None:
    settings = load_settings(Path("config/emotalk.example.json"))

    assert settings.adapters.audio2face.provider == "emotalk_sidecar"
    assert settings.adapters.ue5.provider == "morpheus-raw"
    assert settings.providers.ollama.keep_alive == "30m"
    assert settings.providers.ollama.num_ctx == 2048
    assert settings.providers.ollama.num_predict == 96
    assert settings.providers.ollama.temperature == pytest.approx(0.3)
    assert settings.providers.ollama.prewarm is True
    assert settings.providers.emotalk_sidecar.sidecar_command == [
        "/path/to/miniconda3/envs/emotalk/bin/python",
        "-m",
        "bionic_head.emotalk_sidecar_worker",
    ]
    assert settings.providers.emotalk_sidecar.sidecar_cwd == Path("/path/to/bionic-head-pipeline")
    assert settings.providers.emotalk_sidecar.sidecar_env == {"PYTHONPATH": "src:."}
    assert settings.providers.emotalk_sidecar.sample_rate == 16000
    assert settings.providers.emotalk_sidecar.fps == 30
    assert settings.providers.emotalk_sidecar.channel_count == 52
    assert settings.providers.emotalk_sidecar.output_npy_name == "emotalk.npy"
    assert settings.providers.emotalk_sidecar.timeout_seconds == 20
    assert settings.providers.emotalk_sidecar.prewarm_on_startup is True
    assert settings.providers.emotalk_sidecar.prewarm_on_session_start is False
    assert settings.providers.emotalk_sidecar.prewarm_required is True
    assert settings.providers.emotalk_sidecar.prewarm_audio_seconds == pytest.approx(1.0)
    assert settings.providers.emotalk_sidecar.prewarm_timeout_seconds == pytest.approx(30.0)
    assert settings.providers.emotalk.executable == "/path/to/miniconda3/bin/conda"
    assert settings.providers.emotalk.args == [
        "run",
        "-n",
        "emotalk",
        "python",
        "/path/to/EmoTalk_release/scripts/export_blendshape_from_audio.py",
        "--wav_path",
        "{input_path}",
        "--out_path",
        "{output_dir}/emotalk.npy",
        "--device",
        "cpu",
    ]
    assert settings.providers.emotalk.cwd == Path("/path/to/EmoTalk_release")
    assert settings.providers.emotalk.output_npy_glob == "*.npy"
    assert settings.providers.emotalk.timeout_seconds == 300
    assert settings.providers.piper.runtime == "python"
    assert settings.vad.engine == "rms"
    assert settings.vad.interrupt_min_speech_ms == 80
    assert settings.vad.interrupt_rms_threshold == pytest.approx(0.02)


def test_rejects_unsupported_sample_width() -> None:
    with pytest.raises(ValidationError):
        AppSettings.model_validate(
            {
                "server": {},
                "stream": {"input_sample_width_bytes": 4},
                "retention": {},
                "limits": {},
                "vad": {},
                "adapters": {
                    name: {"provider": "mock"}
                    for name in ("asr", "llm", "tts", "audio2face", "ue5")
                },
                "mock": {},
                "providers": {},
                "storage": {},
            }
        )


def test_accepts_emotalk_sidecar_provider_config() -> None:
    settings = AppSettings.model_validate(
        {
            "adapters": {
                "audio2face": {
                    "provider": "emotalk_sidecar",
                    "timeout_seconds": 10,
                }
            },
            "providers": {
                "emotalk_sidecar": {
                    "sidecar_command": ["python", "-m", "bionic_head.emotalk_fake_sidecar"],
                    "sidecar_cwd": "/tmp/bionic-sidecar",
                    "sidecar_env": {"PYTHONPATH": "src:.", "BIONIC_TEST": "1"},
                    "sample_rate": 16000,
                    "fps": 30,
                    "timeout_seconds": 10.0,
                    "channel_count": 52,
                    "prewarm_on_startup": True,
                    "prewarm_on_session_start": False,
                    "prewarm_required": True,
                    "prewarm_audio_seconds": 1.0,
                    "prewarm_timeout_seconds": 30.0,
                }
            },
        }
    )

    assert settings.adapters.audio2face.provider == "emotalk_sidecar"
    assert settings.providers.emotalk_sidecar.sample_rate == 16000
    assert settings.providers.emotalk_sidecar.fps == 30
    assert settings.providers.emotalk_sidecar.timeout_seconds == pytest.approx(10.0)
    assert settings.providers.emotalk_sidecar.sidecar_cwd == Path("/tmp/bionic-sidecar")
    assert settings.providers.emotalk_sidecar.sidecar_env == {
        "PYTHONPATH": "src:.",
        "BIONIC_TEST": "1",
    }
    assert settings.providers.emotalk_sidecar.prewarm_on_startup is True
    assert settings.providers.emotalk_sidecar.prewarm_on_session_start is False
    assert settings.providers.emotalk_sidecar.prewarm_required is True
    assert settings.providers.emotalk_sidecar.prewarm_audio_seconds == pytest.approx(1.0)
    assert settings.providers.emotalk_sidecar.prewarm_timeout_seconds == pytest.approx(30.0)


def test_accepts_face_stitching_config() -> None:
    settings = AppSettings.model_validate(
        {
            "face_stitching": {
                "enabled": False,
                "overlap_frames": 5,
                "reset_on_new_turn": True,
                "record_boundary_metrics": False,
            }
        }
    )

    assert settings.face_stitching.enabled is False
    assert settings.face_stitching.overlap_frames == 5
    assert settings.face_stitching.reset_on_new_turn is True
    assert settings.face_stitching.record_boundary_metrics is False


def test_accepts_eye_continuity_config() -> None:
    settings = AppSettings.model_validate(
        {
            "eye_continuity": {
                "enabled": True,
                "eye_smooth_channel_indices": [1, 3],
                "blink_enabled": True,
                "blink_channel_indices": [4, 5],
                "overlap_frames": 4,
                "record_boundary_metrics": False,
                "blink_interval_min_sec": 1.5,
                "blink_interval_max_sec": 3.0,
                "blink_duration_frames": 6,
                "blink_strength": 0.8,
                "seed": 123,
                "reset_blink_on_new_turn": True,
            }
        }
    )

    assert settings.eye_continuity.enabled is True
    assert settings.eye_continuity.eye_smooth_channel_indices == [1, 3]
    assert settings.eye_continuity.blink_enabled is True
    assert settings.eye_continuity.blink_channel_indices == [4, 5]
    assert settings.eye_continuity.overlap_frames == 4
    assert settings.eye_continuity.record_boundary_metrics is False
    assert settings.eye_continuity.blink_interval_min_sec == pytest.approx(1.5)
    assert settings.eye_continuity.blink_interval_max_sec == pytest.approx(3.0)
    assert settings.eye_continuity.blink_duration_frames == 6
    assert settings.eye_continuity.blink_strength == pytest.approx(0.8)
    assert settings.eye_continuity.seed == 123
    assert settings.eye_continuity.reset_blink_on_new_turn is True


def test_accepts_history_config() -> None:
    settings = AppSettings.model_validate(
        {
            "history": {
                "enabled": False,
                "max_turn_pairs": 3,
                "max_chars": 512,
            }
        }
    )

    assert settings.history.enabled is False
    assert settings.history.max_turn_pairs == 3
    assert settings.history.max_chars == 512


def test_accepts_expression_config() -> None:
    settings = AppSettings.model_validate(
        {
            "expression": {
                "enabled": True,
                "channel_mapping_path": "config/expression_channels.example.json",
                "max_delta": 0.2,
                "profiles": {
                    "happy": {
                        "mouth_smile_left": 0.15,
                        "mouth_smile_right": 0.15,
                    }
                },
            }
        }
    )

    assert settings.expression.enabled is True
    assert settings.expression.channel_mapping_path == Path("config/expression_channels.example.json")
    assert settings.expression.max_delta == pytest.approx(0.2)
    assert settings.expression.profiles["happy"]["mouth_smile_left"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    "bad_config",
    [
        {"eye_smooth_channel_indices": [-1]},
        {"eye_smooth_channel_indices": [52]},
        {"blink_channel_indices": [-1]},
        {"blink_channel_indices": [52]},
    ],
)
def test_rejects_eye_continuity_channel_indices_outside_morpheus_52_raw(
    bad_config: dict[str, list[int]],
) -> None:
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"eye_continuity": bad_config})


def test_rejects_eye_continuity_blink_interval_max_below_min() -> None:
    with pytest.raises(ValidationError):
        AppSettings.model_validate(
            {
                "eye_continuity": {
                    "blink_interval_min_sec": 4.0,
                    "blink_interval_max_sec": 3.0,
                }
            }
        )
