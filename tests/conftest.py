from array import array
from pathlib import Path
from uuid import uuid4
import wave

import pytest

from bionic_head.core.cancellation import CancellationToken
from bionic_head.domain.models import TurnContext


def _write_pcm_wav(path: Path, samples: array) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    return path


@pytest.fixture
def speech_wav(tmp_path: Path) -> Path:
    return _write_pcm_wav(tmp_path / "speech.wav", array("h", [2000, -2000] * 1600))


@pytest.fixture
def silence_wav(tmp_path: Path) -> Path:
    return _write_pcm_wav(tmp_path / "silence.wav", array("h", [0] * 3200))


@pytest.fixture
def speech_pcm() -> bytes:
    return array("h", [2000, -2000] * 800).tobytes()


@pytest.fixture
def turn_context(tmp_path: Path) -> TurnContext:
    artifact_dir = tmp_path / "turn"
    artifact_dir.mkdir()
    return TurnContext(
        session_id=uuid4(),
        turn_id=uuid4(),
        artifact_dir=artifact_dir,
        cancellation=CancellationToken(),
    )


@pytest.fixture
def mock_settings():
    from bionic_head.config import load_settings

    return load_settings(Path("config/mock.json"))


@pytest.fixture
def mock_registry(mock_settings):
    from bionic_head.adapters.registry import build_registry

    return build_registry(mock_settings)


@pytest.fixture
def artifact_store(tmp_path: Path):
    from bionic_head.core.artifacts import ArtifactStore

    return ArtifactStore(tmp_path / "data")


@pytest.fixture
def offline_orchestrator(mock_settings, mock_registry, artifact_store):
    from bionic_head.orchestrators.offline import OfflineOrchestrator

    async def always_current(_session_id, _turn_id, callback):
        callback()
        return True

    return OfflineOrchestrator(
        settings=mock_settings,
        registry=mock_registry,
        store=artifact_store,
        commit_if_current=always_current,
    )
