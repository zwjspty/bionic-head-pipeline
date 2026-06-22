from pathlib import Path

import pytest
from pydantic import ValidationError

from bionic_head.config import AppSettings, load_settings


def test_load_mock_settings() -> None:
    settings = load_settings(Path("config/mock.json"))
    assert settings.stream.input_sample_rate == 16000
    assert settings.adapters.audio2face.provider == "mock"
    assert settings.server.max_active_sessions == 1


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
                "storage": {},
            }
        )
