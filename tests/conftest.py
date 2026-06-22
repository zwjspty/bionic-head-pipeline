from array import array
from pathlib import Path
from uuid import uuid4
import wave

import pytest

from bionic_head.core.cancellation import CancellationToken
from bionic_head.core.state import TurnHandle
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


@pytest.fixture
def app(mock_settings, tmp_path: Path):
    from bionic_head.api.app import create_app

    settings = mock_settings.model_copy(deep=True)
    settings.storage.root = tmp_path / "api-data"
    return create_app(settings)


class StreamHarness:
    def __init__(self, *, settings, registry, store, input_wav: Path) -> None:
        self.settings = settings
        self.registry = registry
        self.store = store
        self.input_wav = input_wav
        self.turn = TurnHandle(session_id=uuid4(), turn_id=uuid4())
        self.json_envelopes = []
        self.binary_frames: list[bytes] = []
        self._send_lock = None

    @property
    def json_types(self) -> list[str]:
        return [envelope.type.value for envelope in self.json_envelopes]

    @property
    def terminal_types(self) -> list[str]:
        return [
            event_type
            for event_type in self.json_types
            if event_type in {"server.pipeline.done", "server.pipeline.error", "server.turn.cancelled"}
        ]

    async def emit_json(self, envelope) -> None:
        async with self._lock():
            self.json_envelopes.append(envelope)

    async def emit_binary_pair(self, envelope, binary: bytes) -> None:
        async with self._lock():
            self.json_envelopes.append(envelope)
            self.binary_frames.append(binary)

    async def run(self) -> None:
        from bionic_head.orchestrators.stream import StreamOrchestrator

        orchestrator = StreamOrchestrator(
            settings=self.settings,
            registry=self.registry,
            store=self.store,
        )
        await orchestrator.run(
            self.input_wav,
            self.turn,
            self.emit_json,
            self.emit_binary_pair,
        )

    def _lock(self):
        import asyncio

        if self._send_lock is None:
            self._send_lock = asyncio.Lock()
        return self._send_lock


@pytest.fixture
def stream_harness_factory(mock_settings, mock_registry, artifact_store, speech_wav):
    def build(*, settings=mock_settings, registry=mock_registry, input_wav=speech_wav):
        return StreamHarness(
            settings=settings,
            registry=registry,
            store=artifact_store,
            input_wav=input_wav,
        )

    return build


@pytest.fixture
def stream_harness(stream_harness_factory):
    return stream_harness_factory()
