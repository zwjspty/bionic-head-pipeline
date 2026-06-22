import os
from pathlib import Path

import pytest

from bionic_head.adapters.piper import PiperTTSAdapter
from bionic_head.config import load_settings
from bionic_head.domain.models import Emotion


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_piper_generates_playable_wav(turn_context) -> None:
    config_path = os.environ.get("BIONIC_CONFIG")
    if not config_path:
        pytest.skip("BIONIC_CONFIG is required")

    settings = load_settings(Path(config_path))
    adapter = PiperTTSAdapter.from_settings(
        settings.providers.piper,
        grace_seconds=settings.limits.subprocess_terminate_grace_seconds,
    )

    audio = await adapter.synthesize("你好，这是语音测试。", Emotion.FRIENDLY, 0.8, turn_context)

    assert audio.path.stat().st_size > 44
    assert audio.duration_seconds > 0
