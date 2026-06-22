import os
from pathlib import Path

import pytest

from bionic_head.api.app import create_app
from bionic_head.config import AppSettings, load_settings
from bionic_head.core.audio import read_wav_pcm16


@pytest.fixture
def real_settings() -> AppSettings:
    config_path = os.environ.get("BIONIC_CONFIG")
    if not config_path:
        pytest.skip("BIONIC_CONFIG is required")
    settings = load_settings(Path(config_path))
    _require_real_pipeline_config(settings)
    return settings


@pytest.fixture
def chinese_wav() -> Path:
    wav_path = os.environ.get("BIONIC_TEST_WAV")
    if not wav_path:
        pytest.skip("BIONIC_TEST_WAV is required")
    return Path(wav_path)


@pytest.fixture
def chinese_pcm(chinese_wav: Path) -> bytes:
    return read_wav_pcm16(chinese_wav)


@pytest.fixture
def real_app(real_settings: AppSettings, tmp_path: Path):
    settings = real_settings.model_copy(deep=True)
    settings.storage.root = tmp_path / "real-test-data"
    return create_app(settings)


def _require_real_pipeline_config(settings: AppSettings) -> None:
    if settings.adapters.tts.provider == "piper":
        if not settings.providers.piper.executable:
            pytest.skip("settings.providers.piper.executable is required")
        if settings.providers.piper.model_path is None:
            pytest.skip("settings.providers.piper.model_path is required")
    if settings.adapters.audio2face.provider == "morpheus":
        if not settings.providers.morpheus.executable:
            pytest.skip("settings.providers.morpheus.executable is required")
        if any(arg == "" for arg in settings.providers.morpheus.args):
            pytest.skip("settings.providers.morpheus.args must not contain an empty command element")
