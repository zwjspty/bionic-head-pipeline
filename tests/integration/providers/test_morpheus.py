import os
from pathlib import Path

import pytest

from bionic_head.adapters.morpheus import MorpheusAudio2FaceAdapter
from bionic_head.core.audio import audio_artifact_from_wav
from bionic_head.config import load_settings
from bionic_head.domain.models import Emotion


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_morpheus_produces_52_channels(turn_context) -> None:
    config_path = os.environ.get("BIONIC_CONFIG")
    tts_wav_path = os.environ.get("BIONIC_TEST_TTS_WAV")
    if not config_path or not tts_wav_path:
        pytest.skip("BIONIC_CONFIG and BIONIC_TEST_TTS_WAV are required")

    settings = load_settings(Path(config_path))
    adapter = MorpheusAudio2FaceAdapter.from_settings(
        settings.providers.morpheus,
        grace_seconds=settings.limits.subprocess_terminate_grace_seconds,
    )
    audio = audio_artifact_from_wav(Path(tts_wav_path))

    face = await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert face.channel_count == 52
    assert face.frame_count > 0
