import os
from pathlib import Path

import pytest

from bionic_head.adapters.ollama import OllamaLLMAdapter
from bionic_head.config import load_settings


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_ollama_streams_valid_result(turn_context) -> None:
    config_path = os.environ.get("BIONIC_CONFIG")
    if not config_path:
        pytest.skip("BIONIC_CONFIG is required")

    settings = load_settings(Path(config_path))
    adapter = OllamaLLMAdapter(settings.providers.ollama)

    events = [
        event async for event in adapter.chat_stream("请简单打个招呼", [], turn_context)
    ]

    assert events[-1].result is not None
    assert events[-1].result.reply
    assert 0.0 <= events[-1].result.intensity <= 1.0
