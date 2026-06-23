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


def test_load_emotalk_example_settings() -> None:
    settings = load_settings(Path("config/emotalk.example.json"))

    assert settings.adapters.audio2face.provider == "emotalk"
    assert settings.adapters.ue5.provider == "morpheus-raw"
    assert settings.providers.ollama.keep_alive == "30m"
    assert settings.providers.ollama.num_ctx == 2048
    assert settings.providers.ollama.num_predict == 96
    assert settings.providers.ollama.temperature == pytest.approx(0.3)
    assert settings.providers.ollama.prewarm is True
    assert settings.providers.emotalk.executable == "/home/user/miniconda3/bin/conda"
    assert settings.providers.emotalk.args == [
        "run",
        "-n",
        "emotalk",
        "python",
        "/home/user/code/EmoTalk_release/scripts/export_blendshape_from_audio.py",
        "--wav_path",
        "{input_path}",
        "--out_path",
        "{output_dir}/emotalk.npy",
        "--device",
        "cpu",
    ]
    assert settings.providers.emotalk.cwd == Path("/home/user/code/EmoTalk_release")
    assert settings.providers.emotalk.output_npy_glob == "*.npy"
    assert settings.providers.emotalk.timeout_seconds == 300
    assert settings.providers.piper.runtime == "python"


def test_rejects_unsupported_sample_width() -> None:
    with pytest.raises(ValidationError):
        AppSettings.model_validate(
            {
                "server": {},
                "stream": {"input_sample_width_bytes": 4},
                "retention": {},
                "limits": {},
                "adapters": {
                    name: {"provider": "mock"}
                    for name in ("asr", "llm", "tts", "audio2face", "ue5")
                },
                "mock": {},
                "providers": {},
                "storage": {},
            }
        )
