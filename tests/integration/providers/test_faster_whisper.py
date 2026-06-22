import os
from pathlib import Path

import pytest

from bionic_head.adapters.faster_whisper import FasterWhisperASRAdapter
from bionic_head.config import load_settings


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_faster_whisper_recognizes_chinese(turn_context) -> None:
    config_path = os.environ.get("BIONIC_CONFIG")
    wav_path = os.environ.get("BIONIC_TEST_WAV")
    if not config_path or not wav_path:
        pytest.skip("BIONIC_CONFIG and BIONIC_TEST_WAV are required")

    settings = load_settings(Path(config_path))
    adapter = FasterWhisperASRAdapter(settings.providers.faster_whisper)

    result = await adapter.transcribe(Path(wav_path), turn_context)

    assert result.text.strip()
    assert result.language == "zh"
