# P0 Mock Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully tested, all-Mock FastAPI service that supports the offline audio pipeline, the pseudo-streaming WebSocket pipeline, diagnostics, timelines, artifacts, latest publication, cancellation, and stale-turn suppression.

**Architecture:** Use a modular monolith under `src/bionic_head`. HTTP and WebSocket routes delegate to orchestrators; orchestrators depend only on adapter protocols and domain models. All P0 providers are deterministic Mocks, while provider selection, process boundaries, and configuration shapes remain compatible with the later real-provider plan.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Pydantic v2, asyncio, NumPy, pytest, pytest-asyncio, HTTPX, Starlette TestClient.

## Global Constraints

- Runtime target is Python 3.11 on Linux.
- Configuration is JSON; real paths must never be hard-coded in Python modules.
- The service is local/LAN only; do not add authentication, HTTPS, Docker, Redis, a database, or a message broker.
- Default tests must use only Mock providers and must not require GPU, Conda, Ollama, Piper, Morpheus, or external audio files.
- Input stream audio is signed 16-bit little-endian PCM, mono, 16000 Hz, in 20–100 ms chunks.
- `max_active_sessions = 1` and `morpheus_max_concurrency = 1`.
- All externally visible stream events carry `protocol`, `type`, `event_id`, `session_id`, `turn_id`, `sequence`, `timestamp`, and `payload`.
- A turn emits exactly one terminal event: `server.pipeline.done`, `server.pipeline.error`, or `server.turn.cancelled`.
- Cancelled, failed, or stale turns never publish `data/latest/*`.
- Face output is exactly 52 channels at 30 fps and is named `morpheus_52_raw`, not ARKit or MetaHuman.
- Implement every task with TDD and commit after its complete test cycle.

---

## File Map

```text
pyproject.toml                         Package metadata, dependencies, pytest configuration
.gitignore                             Ignore virtualenv, caches, runtime data, local config
config/mock.json                       Default runnable all-Mock configuration
config/real.example.json               Shape of later real-provider configuration
src/bionic_head/config.py              Pydantic settings and JSON loader
src/bionic_head/domain/models.py       Shared domain/result models
src/bionic_head/domain/errors.py       Error codes and safe pipeline exception
src/bionic_head/adapters/protocols.py  Adapter contracts
src/bionic_head/adapters/registry.py   Provider construction and lookup
src/bionic_head/adapters/mock.py       Configurable Mock providers
src/bionic_head/core/audio.py          WAV/PCM validation, RMS, conversion
src/bionic_head/core/cancellation.py   Cooperative cancellation token
src/bionic_head/core/timeline.py       UTC and monotonic timing
src/bionic_head/core/artifacts.py      Run directories and atomic JSON publication
src/bionic_head/core/state.py          Turn state machine and session admission
src/bionic_head/core/sentences.py      Streaming sentence segmentation
src/bionic_head/core/ue5.py            52-channel payload validation and chunking
src/bionic_head/orchestrators/offline.py  Offline full-chain coordinator
src/bionic_head/orchestrators/stream.py   Pseudo-streaming coordinator
src/bionic_head/protocol/events.py     WebSocket envelopes and event builders
src/bionic_head/protocol/connection.py JSON/binary pairing and stream controller
src/bionic_head/api/app.py             FastAPI application factory
src/bionic_head/api/dependencies.py    Application container assembly
src/bionic_head/api/routes/health.py   Health and diagnostics routes
src/bionic_head/api/routes/pipeline.py Offline and latest routes
src/bionic_head/api/routes/stream.py   WebSocket route
tests/                                 Unit and integration tests mirroring the package
```

### Task 1: Package Skeleton and Test Harness

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `config/mock.json`
- Create: `config/real.example.json`
- Create: `src/bionic_head/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_package.py`

**Interfaces:**
- Produces: installable package `bionic-head`, pytest marker `integration`, and `config/mock.json` as the default development configuration.

- [ ] **Step 1: Write the failing package test**

```python
# tests/test_package.py
def test_package_exposes_version() -> None:
    import bionic_head

    assert bionic_head.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `python3.11 -m venv .venv`

Expected: `.venv/bin/python` exists.

Run: `.venv/bin/python -m pip install --upgrade pip pytest`

Expected: pip and pytest install into `.venv`.

Run: `.venv/bin/python -m pytest tests/test_package.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'bionic_head'`.

- [ ] **Step 3: Add package metadata and minimal package**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "bionic-head"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "fastapi>=0.115,<1",
  "uvicorn>=0.34,<1",
  "pydantic>=2.10,<3",
  "numpy>=2.1,<3",
  "python-multipart>=0.0.20,<1",
]

[project.optional-dependencies]
dev = [
  "httpx>=0.28,<1",
  "pytest>=8.3,<9",
  "pytest-asyncio>=0.25,<1",
  "pytest-timeout>=2.3,<3",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
  "integration: requires real external providers",
]
```

```python
# src/bionic_head/__init__.py
__version__ = "0.1.0"
```

```gitignore
# .gitignore
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
*.egg-info/
config/local.json
data/runs/
data/latest/
```

Create `config/mock.json` with the exact defaults from the design:

```json
{
  "server": {"host": "127.0.0.1", "port": 8000, "max_active_sessions": 1},
  "stream": {
    "silence_timeout_ms": 1000,
    "max_turn_duration_seconds": 30,
    "silence_rms_threshold": 0.01,
    "input_sample_rate": 16000,
    "input_channels": 1,
    "input_sample_width_bytes": 2,
    "sentence_max_chars": 80,
    "sentence_max_wait_ms": 500
  },
  "retention": {"max_runs": 100},
  "limits": {"morpheus_max_concurrency": 1, "subprocess_terminate_grace_seconds": 2},
  "adapters": {
    "asr": {"provider": "mock", "timeout_seconds": 5},
    "llm": {"provider": "mock", "timeout_seconds": 5},
    "tts": {"provider": "mock", "timeout_seconds": 5},
    "audio2face": {"provider": "mock", "timeout_seconds": 5},
    "ue5": {"provider": "mock", "timeout_seconds": 5}
  },
  "mock": {
    "latency_ms": {"asr": 0, "llm_first_token": 0, "llm_token": 0, "tts": 0, "face": 0},
    "fail_stage": null,
    "timeout_stage": null,
    "asr_text": "你好",
    "reply": "你好！很高兴见到你。",
    "emotion": "friendly",
    "intensity": 0.8
  },
  "storage": {"root": "data"}
}
```

Create `config/real.example.json` with the same common sections and provider names `faster-whisper`, `ollama`, `piper`, `morpheus`, and `morpheus-raw`; use empty strings for currently unknown executable/model/command values.

- [ ] **Step 4: Install the editable package and run the test**

Run: `.venv/bin/python -m pip install -e '.[dev]'`

Expected: installation succeeds.

Run: `.venv/bin/python -m pytest tests/test_package.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore config src/bionic_head/__init__.py tests
git commit -m "build: scaffold bionic head service"
```

### Task 2: Configuration, Domain Models, and Error Contract

**Files:**
- Create: `src/bionic_head/config.py`
- Create: `src/bionic_head/domain/__init__.py`
- Create: `src/bionic_head/domain/models.py`
- Create: `src/bionic_head/domain/errors.py`
- Create: `src/bionic_head/core/__init__.py`
- Create: `src/bionic_head/core/cancellation.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_domain.py`

**Interfaces:**
- Produces: `AppSettings`, `load_settings(path: Path)`, `TurnContext`, adapter result models, `ErrorCode`, and `PipelineException`.
- Consumes: `config/mock.json`.

- [ ] **Step 1: Write failing configuration and model tests**

```python
# tests/unit/test_config.py
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
        AppSettings.model_validate({
            "server": {},
            "stream": {"input_sample_width_bytes": 4},
            "retention": {},
            "limits": {},
            "adapters": {
                name: {"provider": "mock"}
                for name in ("asr", "llm", "tts", "audio2face", "ue5")
            },
            "mock": {},
            "storage": {}
        })
```

```python
# tests/unit/test_domain.py
from pathlib import Path
from uuid import uuid4

from bionic_head.core.cancellation import CancellationToken
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import Emotion, TurnContext


def test_turn_context_keeps_identity_and_artifact_dir(tmp_path: Path) -> None:
    context = TurnContext(
        session_id=uuid4(),
        turn_id=uuid4(),
        artifact_dir=tmp_path,
        cancellation=CancellationToken(),
    )
    assert context.artifact_dir == tmp_path


def test_pipeline_exception_has_safe_shape() -> None:
    error = PipelineException(
        code=ErrorCode.PROVIDER_FAILED,
        stage="tts",
        provider="mock",
        retryable=False,
        message="TTS failed",
    )
    assert error.to_detail()["code"] == "provider_failed"
    assert Emotion.FRIENDLY.value == "friendly"
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `.venv/bin/python -m pytest tests/unit/test_config.py tests/unit/test_domain.py -v`

Expected: FAIL because `bionic_head.config`, `domain`, and `core.cancellation` do not exist.

- [ ] **Step 3: Implement settings, cancellation token, models, and errors**

Define these exact public types:

```python
# src/bionic_head/core/cancellation.py
import asyncio


class CancellationToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError
```

```python
# src/bionic_head/domain/errors.py
from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INVALID_AUDIO_FORMAT = "invalid_audio_format"
    NO_SPEECH_DETECTED = "no_speech_detected"
    SESSION_LIMIT_REACHED = "session_limit_reached"
    PROTOCOL_VIOLATION = "protocol_violation"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_FAILED = "provider_failed"
    OUTPUT_VALIDATION_FAILED = "output_validation_failed"
    TURN_CANCELLED = "turn_cancelled"
    INTERNAL_ERROR = "internal_error"


class PipelineException(Exception):
    def __init__(
        self,
        *,
        code: ErrorCode,
        stage: str,
        provider: str | None,
        retryable: bool,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.provider = provider
        self.retryable = retryable
        self.safe_message = message

    def to_detail(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "stage": self.stage,
            "provider": self.provider,
            "retryable": self.retryable,
            "message": self.safe_message,
        }
```

In `domain/models.py`, define these exact fields:

```python
class Emotion(StrEnum):
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    THINKING = "thinking"
    CALM = "calm"


class AudioStats(BaseModel):
    sample_rate: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    duration_seconds: float
    rms: float
    peak: float


class AudioFormat(BaseModel):
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    sample_width_bytes: Literal[2] = 2


class ASRResult(BaseModel):
    text: str
    language: str
    confidence: float | None = None
    audio: AudioStats


class LLMResult(BaseModel):
    reply: str
    emotion: Emotion
    intensity: float = Field(ge=0.0, le=1.0)


class LLMEvent(BaseModel):
    kind: Literal["token", "final"]
    text: str = ""
    result: LLMResult | None = None


class AudioArtifact(BaseModel):
    path: Path
    sample_rate: int
    channels: int
    sample_width_bytes: int
    duration_seconds: float
    byte_length: int


class FaceArtifact(BaseModel):
    path: Path | None = None
    frames: list[list[float]]
    fps: int = 30
    channel_count: int = 52
    frame_count: int
    auxiliary_paths: list[Path] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)


class UE5Frame(BaseModel):
    frame_index: int
    time_seconds: float
    weights: list[float]


class UE5Payload(BaseModel):
    protocol: Literal["bionic-head-ue5-v1"] = "bionic-head-ue5-v1"
    format: Literal["morpheus_52_raw"] = "morpheus_52_raw"
    fps: int = 30
    channel_count: Literal[52] = 52
    channels: list[str]
    frame_count: int
    frames: list[UE5Frame]


class DiagnosticResult(BaseModel):
    adapter: str
    provider: str
    available: bool
    latency_ms: float
    message: str


class PipelineResult(BaseModel):
    session_id: UUID
    turn_id: UUID
    asr: ASRResult
    llm: LLMResult
    audio: AudioArtifact
    face: FaceArtifact
    ue5: UE5Payload
    timeline: dict[str, object]


@dataclass(frozen=True)
class TurnContext:
    session_id: UUID
    turn_id: UUID
    artifact_dir: Path
    cancellation: CancellationToken
```

Validate every face/frame weight list has length 52 and finite numeric values. Require `frame_count == len(frames)`, `channels` length 52, and timezone-aware datetimes wherever timestamps appear.

In `config.py`, define nested Pydantic settings models and:

```python
def load_settings(path: Path) -> AppSettings:
    return AppSettings.model_validate_json(path.read_text(encoding="utf-8"))
```

Set all defaults exactly as `config/mock.json`; validate sample rate `16000`, channels `1`, sample width `2`, positive timeouts, and max sessions/concurrency equal to or above one.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_config.py tests/unit/test_domain.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bionic_head/config.py src/bionic_head/core src/bionic_head/domain tests/unit
git commit -m "feat: define configuration and domain contracts"
```

### Task 3: Audio Validation, Timeline, and Artifact Storage

**Files:**
- Create: `src/bionic_head/core/audio.py`
- Create: `src/bionic_head/core/timeline.py`
- Create: `src/bionic_head/core/artifacts.py`
- Create: `tests/unit/test_audio.py`
- Create: `tests/unit/test_timeline.py`
- Create: `tests/unit/test_artifacts.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces:
  - `inspect_wav(path: Path, expected: AudioFormat) -> AudioStats`
  - `audio_artifact_from_wav(path: Path) -> AudioArtifact`
  - `pcm16le_to_wav(pcm: bytes, path: Path, sample_rate: int = 16000) -> AudioArtifact`
  - `read_wav_pcm16(path: Path) -> bytes`
  - `Timeline.stage(name, provider)`
  - `ArtifactStore.create_turn(session_id, turn_id) -> Path`
  - `await ArtifactStore.publish_latest(pipeline, ue5, commit_if_current)`.

- [ ] **Step 1: Write failing utility tests**

```python
# tests/unit/test_audio.py
from array import array
from pathlib import Path

from bionic_head.core.audio import pcm16le_to_wav, inspect_wav


def test_pcm_round_trip_and_rms(tmp_path: Path) -> None:
    samples = array("h", [1000, -1000] * 1600)
    artifact = pcm16le_to_wav(samples.tobytes(), tmp_path / "input.wav")
    stats = inspect_wav(artifact.path)
    assert stats.sample_rate == 16000
    assert stats.channels == 1
    assert stats.rms > 0
```

```python
# tests/unit/test_artifacts.py
from uuid import uuid4
import pytest

from bionic_head.core.artifacts import ArtifactStore


@pytest.mark.asyncio
async def test_stale_turn_cannot_publish_latest(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    async def reject(_action) -> bool:
        return False
    await store.publish_latest(
        pipeline={"turn": "old"},
        ue5={"frames": []},
        commit_if_current=reject,
    )
    assert not (tmp_path / "latest/latest_pipeline.json").exists()
```

```python
# tests/unit/test_timeline.py
from bionic_head.core.timeline import Timeline


def test_timeline_records_completed_stage() -> None:
    timeline = Timeline()
    with timeline.stage("asr", "mock"):
        pass
    item = timeline.snapshot()["stages"][0]
    assert item["status"] == "completed"
    assert item["duration_ms"] >= 0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_audio.py tests/unit/test_timeline.py tests/unit/test_artifacts.py -v`

Expected: FAIL because utility modules do not exist.

- [ ] **Step 3: Implement audio, timing, and atomic publication**

In `audio.py`, use `wave` and `array("h")`; reject malformed WAV, wrong channels/sample width/sample rate, odd PCM byte length, and empty audio with `PipelineException(ErrorCode.INVALID_AUDIO_FORMAT, stage="audio")`. `audio_artifact_from_wav` converts validated WAV metadata into `AudioArtifact`. `read_wav_pcm16` first calls the same validation path and then returns frame bytes. Compute normalized RMS as:

```python
normalized_rms = (sum(sample * sample for sample in samples) / len(samples)) ** 0.5 / 32768.0
peak = max(abs(sample) for sample in samples) / 32768.0
```

In `timeline.py`, use `datetime.now(timezone.utc)` for timestamps and `time.monotonic_ns()` for duration. A failed context records `status="failed"` and the `PipelineException.code` when available. Provide `mark(name: str)`, `metric(name: str, start_mark: str, end_mark: str)`, `snapshot()`, and `write(path)`.

In `artifacts.py`, create:

```python
class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs = root / "runs"
        self.latest = root / "latest"

    def create_turn(self, session_id: UUID, turn_id: UUID) -> Path:
        turn_dir = self.runs / str(session_id) / str(turn_id)
        for relative in ("tts", "face", "ue5"):
            (turn_dir / relative).mkdir(parents=True, exist_ok=True)
        return turn_dir

    def write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._atomic_write(path, encoded)

    async def publish_latest(
        self,
        *,
        pipeline: object,
        ue5: object,
        commit_if_current: Callable[[Callable[[], None]], Awaitable[bool]],
    ) -> bool:
        self.latest.mkdir(parents=True, exist_ok=True)
        pipeline_path = self.latest / "latest_pipeline.json"
        ue5_path = self.latest / "latest_ue5_blendshape.json"
        pipeline_tmp = self._write_temp(pipeline_path, pipeline)
        ue5_tmp = self._write_temp(ue5_path, ue5)
        def commit() -> None:
            os.replace(pipeline_tmp, pipeline_path)
            os.replace(ue5_tmp, ue5_path)
        published = await commit_if_current(commit)
        if not published:
            pipeline_tmp.unlink(missing_ok=True)
            ue5_tmp.unlink(missing_ok=True)
        return published
```

Use `tempfile.NamedTemporaryFile(dir=target.parent, delete=False)` in `_write_temp`. The supplied `commit_if_current` must hold the same turn lock used by cancellation while it checks currentness and invokes `commit`; this prevents a turn becoming stale between the two `os.replace` calls.

- [ ] **Step 4: Add shared audio and turn fixtures**

Append fixtures with these exact responsibilities to `tests/conftest.py`:

```python
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
```

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_audio.py tests/unit/test_timeline.py tests/unit/test_artifacts.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/bionic_head/core tests/conftest.py tests/unit
git commit -m "feat: add audio timeline and artifact infrastructure"
```

### Task 4: Adapter Protocols, Mock Providers, Registry, and UE5 Formatting

**Files:**
- Create: `src/bionic_head/adapters/__init__.py`
- Create: `src/bionic_head/adapters/protocols.py`
- Create: `src/bionic_head/adapters/mock.py`
- Create: `src/bionic_head/adapters/registry.py`
- Create: `src/bionic_head/core/ue5.py`
- Create: `tests/unit/test_mock_adapters.py`
- Create: `tests/unit/test_registry.py`
- Create: `tests/unit/test_ue5.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces: `AdapterRegistry`, `build_registry(settings)`, five adapter protocols, Mock implementations, `build_ue5_payload`, and `chunk_ue5_frames`.
- Consumes: domain models, `AppSettings`, `TurnContext`.

- [ ] **Step 1: Write failing adapter and formatter tests**

```python
# tests/unit/test_mock_adapters.py
import pytest

from pathlib import Path

from bionic_head.adapters.registry import build_registry
from bionic_head.config import load_settings


@pytest.mark.asyncio
async def test_mock_chain_returns_deterministic_results(turn_context) -> None:
    registry = build_registry(load_settings(Path("config/mock.json")))
    asr = await registry.asr.transcribe(turn_context.artifact_dir / "input.wav", turn_context)
    llm = await registry.llm.chat(asr.text, [], turn_context)
    assert asr.text == "你好"
    assert llm.emotion.value == "friendly"
```

```python
# tests/unit/test_ue5.py
from bionic_head.core.ue5 import build_ue5_payload, chunk_ue5_frames


def test_formats_and_chunks_52_channel_frames() -> None:
    payload = build_ue5_payload([[0.0] * 52 for _ in range(31)], fps=30)
    chunks = list(chunk_ue5_frames(payload, chunk_size=30, chunk_id="s0"))
    assert payload.channels[0] == "morpheus_00"
    assert [chunk["frame_count"] for chunk in chunks] == [30, 1]
    assert chunks[-1]["is_last"] is True
```

- [ ] **Step 2: Run tests and verify missing interfaces**

Run: `.venv/bin/python -m pytest tests/unit/test_mock_adapters.py tests/unit/test_registry.py tests/unit/test_ue5.py -v`

Expected: FAIL because adapter modules and UE5 formatter do not exist.

- [ ] **Step 3: Define protocols and deterministic Mocks**

Define runtime-checkable protocols with these exact async signatures:

```python
class ASRAdapter(Protocol):
    name: str
    async def transcribe(self, audio_path: Path, context: TurnContext) -> ASRResult:
        raise NotImplementedError
    async def diagnostics(self) -> DiagnosticResult:
        raise NotImplementedError
    async def cancel(self, turn_id: UUID) -> None:
        raise NotImplementedError

class LLMAdapter(Protocol):
    name: str
    async def chat(self, text: str, history: list[dict[str, str]], context: TurnContext) -> LLMResult:
        raise NotImplementedError
    def chat_stream(self, text: str, history: list[dict[str, str]], context: TurnContext) -> AsyncIterator[LLMEvent]:
        raise NotImplementedError
    async def diagnostics(self) -> DiagnosticResult:
        raise NotImplementedError
    async def cancel(self, turn_id: UUID) -> None:
        raise NotImplementedError

class TTSAdapter(Protocol):
    name: str
    async def synthesize(
        self,
        text: str,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> AudioArtifact:
        raise NotImplementedError
    async def diagnostics(self) -> DiagnosticResult:
        raise NotImplementedError
    async def cancel(self, turn_id: UUID) -> None:
        raise NotImplementedError

class Audio2FaceAdapter(Protocol):
    name: str
    async def drive(
        self,
        audio: AudioArtifact,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> FaceArtifact:
        raise NotImplementedError
    async def diagnostics(self) -> DiagnosticResult:
        raise NotImplementedError
    async def cancel(self, turn_id: UUID) -> None:
        raise NotImplementedError

class UE5Adapter(Protocol):
    name: str
    async def format(self, face: FaceArtifact, context: TurnContext) -> UE5Payload:
        raise NotImplementedError
    async def diagnostics(self) -> DiagnosticResult:
        raise NotImplementedError
    async def cancel(self, turn_id: UUID) -> None:
        raise NotImplementedError
```

Mock behavior:

- `_before(stage, latency_ms, context)` checks cancellation, raises configured failure, sleeps configured latency, then raises timeout by sleeping longer than the adapter timeout when configured.
- Mock TTS writes a valid mono 16 kHz WAV containing 250 ms of a low-amplitude sine wave.
- Mock face output duration matches the WAV duration, with `round(duration * 30)` frames and 52 deterministic weights.
- Mock LLM streaming emits one Unicode character per token and then a final event carrying emotion/intensity.

Use `asyncio.wait_for` in a small registry wrapper so every adapter call maps timeout to `provider_timeout` and unexpected errors to `provider_failed`.

- [ ] **Step 4: Implement registry and UE5 validation**

`AdapterRegistry` is a dataclass with `asr`, `llm`, `tts`, `audio2face`, and `ue5`. `build_registry(settings)` accepts only provider `mock` in P0; unknown providers raise `PipelineException(code=PROVIDER_UNAVAILABLE, stage="startup")`.

`build_ue5_payload` validates exactly 52 finite weights per frame and creates channel names `morpheus_00` through `morpheus_51`. `chunk_ue5_frames` rebases neither frame index nor time; it emits at most 30 frames per chunk.

- [ ] **Step 5: Add the registry fixture**

Append:

```python
@pytest.fixture
def mock_settings():
    return load_settings(Path("config/mock.json"))


@pytest.fixture
def mock_registry(mock_settings):
    return build_registry(mock_settings)
```

Mock classes expose per-method call counters so short-circuit and cancellation tests can assert that downstream providers were not invoked.

- [ ] **Step 6: Run focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_mock_adapters.py tests/unit/test_registry.py tests/unit/test_ue5.py -v`

Expected: PASS, including configured failure, timeout, cancellation, 52-channel validation, and 31-frame split tests.

- [ ] **Step 7: Commit**

```bash
git add src/bionic_head/adapters src/bionic_head/core/ue5.py tests/conftest.py tests/unit
git commit -m "feat: add mock provider registry and ue5 formatter"
```

### Task 5: Offline Orchestrator

**Files:**
- Create: `src/bionic_head/orchestrators/__init__.py`
- Create: `src/bionic_head/orchestrators/offline.py`
- Create: `tests/unit/test_offline_orchestrator.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: `AdapterRegistry`, `ArtifactStore`, valid WAV path, `TurnContext`.
- Produces: `OfflineOrchestrator.run(input_path, context) -> PipelineResult`.

- [ ] **Step 1: Write failing happy-path and silence tests**

```python
# tests/unit/test_offline_orchestrator.py
import pytest

from bionic_head.domain.errors import ErrorCode, PipelineException


@pytest.mark.asyncio
async def test_offline_pipeline_writes_all_artifacts(offline_orchestrator, speech_wav, turn_context) -> None:
    result = await offline_orchestrator.run(speech_wav, turn_context)
    assert result.asr.text == "你好"
    assert result.face.channel_count == 52
    assert (turn_context.artifact_dir / "timeline.json").exists()
    assert (turn_context.artifact_dir / "ue5/result.json").exists()


@pytest.mark.asyncio
async def test_silence_stops_before_llm(offline_orchestrator, silence_wav, turn_context, mock_registry) -> None:
    with pytest.raises(PipelineException) as raised:
        await offline_orchestrator.run(silence_wav, turn_context)
    assert raised.value.code is ErrorCode.NO_SPEECH_DETECTED
    assert mock_registry.llm.call_count == 0
```

- [ ] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_offline_orchestrator.py -v`

Expected: FAIL because `OfflineOrchestrator` does not exist.

- [ ] **Step 3: Implement the exact stage order**

Implement:

```python
class OfflineOrchestrator:
    async def run(self, input_path: Path, context: TurnContext) -> PipelineResult:
        timeline = Timeline()
        copied = context.artifact_dir / "input.wav"
        shutil.copy2(input_path, copied)
        stats = inspect_wav(copied)
        if stats.rms <= self.settings.stream.silence_rms_threshold:
            raise PipelineException(
                code=ErrorCode.NO_SPEECH_DETECTED,
                stage="audio",
                provider=None,
                retryable=True,
                message="No speech detected",
            )
        with timeline.stage("asr", self.registry.asr.name):
            asr = await self.registry.asr.transcribe(copied, context)
        with timeline.stage("llm", self.registry.llm.name):
            llm = await self.registry.llm.chat(asr.text, [], context)
        with timeline.stage("tts", self.registry.tts.name):
            audio = await self.registry.tts.synthesize(
                llm.reply, llm.emotion, llm.intensity, context
            )
        with timeline.stage("audio2face", self.registry.audio2face.name):
            face = await self.registry.audio2face.drive(
                audio, llm.emotion, llm.intensity, context
            )
        with timeline.stage("ue5", self.registry.ue5.name):
            ue5 = await self.registry.ue5.format(face, context)
        snapshot = timeline.snapshot()
        result = PipelineResult(
            session_id=context.session_id,
            turn_id=context.turn_id,
            asr=asr,
            llm=llm,
            audio=audio,
            face=face,
            ue5=ue5,
            timeline=snapshot,
        )
        self.store.write_json(context.artifact_dir / "asr.json", asr.model_dump(mode="json"))
        self.store.write_json(context.artifact_dir / "llm.json", llm.model_dump(mode="json"))
        self.store.write_json(context.artifact_dir / "ue5/result.json", ue5.model_dump(mode="json"))
        self.store.write_json(context.artifact_dir / "timeline.json", snapshot)
        await self.store.publish_latest(
            pipeline=result.model_dump(mode="json"),
            ue5=ue5.model_dump(mode="json"),
            commit_if_current=lambda callback: self.commit_if_current(
                context.session_id,
                context.turn_id,
                callback,
            ),
        )
        return result
```

Complete the method by writing `asr.json`, `llm.json`, TTS/face/UE5 outputs, and `timeline.json`. Publish latest only through an injected `commit_if_current(session_id, turn_id, callback)` guard. On every exception, write timeline before re-raising. On cancellation, mark the turn cancelled and do not publish.

- [ ] **Step 4: Add the offline orchestrator fixture**

Append:

```python
@pytest.fixture
def artifact_store(tmp_path):
    return ArtifactStore(tmp_path / "data")


@pytest.fixture
def offline_orchestrator(mock_settings, mock_registry, artifact_store):
    async def always_current(_session_id, _turn_id, callback):
        callback()
        return True
    return OfflineOrchestrator(
        settings=mock_settings,
        registry=mock_registry,
        store=artifact_store,
        commit_if_current=always_current,
    )
```

Reset Mock call counters before every test so test order cannot affect assertions.

- [ ] **Step 5: Run orchestrator tests**

Run: `.venv/bin/python -m pytest tests/unit/test_offline_orchestrator.py -v`

Expected: PASS for success, silence short-circuit, provider failure, timeout, and stale-turn latest suppression.

- [ ] **Step 6: Commit**

```bash
git add src/bionic_head/orchestrators tests/conftest.py tests/unit/test_offline_orchestrator.py
git commit -m "feat: implement offline mock pipeline"
```

### Task 6: FastAPI Container, Health, Diagnostics, Offline, and Latest Routes

**Files:**
- Create: `src/bionic_head/api/__init__.py`
- Create: `src/bionic_head/api/app.py`
- Create: `src/bionic_head/api/dependencies.py`
- Create: `src/bionic_head/api/routes/__init__.py`
- Create: `src/bionic_head/api/routes/health.py`
- Create: `src/bionic_head/api/routes/pipeline.py`
- Create: `tests/integration/test_http_api.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces:
  - `create_app(settings: AppSettings | None = None) -> FastAPI`
  - `GET /health`
  - `GET /diagnostics`
  - `GET /diagnostics/{adapter}`
  - `POST /pipeline/audio`
  - `GET /pipeline/latest`
  - `GET /ue5/latest`.

- [ ] **Step 1: Write failing HTTP integration tests**

```python
# tests/integration/test_http_api.py
from fastapi.testclient import TestClient


def test_health_is_independent_of_provider_status(app) -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_offline_endpoint_and_latest(app, speech_wav) -> None:
    client = TestClient(app)
    with speech_wav.open("rb") as handle:
        response = client.post(
            "/pipeline/audio",
            files={"audio": ("speech.wav", handle, "audio/wav")},
        )
    assert response.status_code == 200
    assert response.json()["face"]["channel_count"] == 52
    assert client.get("/pipeline/latest").status_code == 200
    assert client.get("/ue5/latest").json()["format"] == "morpheus_52_raw"
```

- [ ] **Step 2: Run tests and verify missing app**

Run: `.venv/bin/python -m pytest tests/integration/test_http_api.py -v`

Expected: FAIL because the application factory does not exist.

- [ ] **Step 3: Implement the application container and exception mapping**

Create an `AppContainer` dataclass containing settings, registry, store, `SessionManager(max_active_sessions=1)`, and orchestrator factories. Store it in `app.state.container`.

`create_app(settings=None)` loads:

```python
config_path = Path(os.environ.get("BIONIC_CONFIG", "config/mock.json"))
resolved_settings = settings or load_settings(config_path)
```

Map errors:

```python
ERROR_STATUS = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.INVALID_AUDIO_FORMAT: 415,
    ErrorCode.NO_SPEECH_DETECTED: 422,
    ErrorCode.SESSION_LIMIT_REACHED: 429,
    ErrorCode.PROVIDER_UNAVAILABLE: 503,
    ErrorCode.PROVIDER_TIMEOUT: 504,
    ErrorCode.PROVIDER_FAILED: 502,
    ErrorCode.OUTPUT_VALIDATION_FAILED: 502,
}
```

The handler returns `{"error": exc.to_detail()}` and never returns `repr(exc)`, command arrays, stack traces, or local provider paths.

`POST /pipeline/audio` copies the upload to the new turn directory, invokes the offline orchestrator, and returns `PipelineResult.model_dump(mode="json")`. Missing latest files return 404 with `invalid_request`.

- [ ] **Step 4: Implement diagnostics**

`GET /diagnostics` concurrently awaits all five adapter diagnostics with `asyncio.gather`; the route itself returns 200 even if a provider is unavailable. `GET /diagnostics/{adapter}` rejects names outside `asr/llm/tts/audio2face/ue5` with 404.

- [ ] **Step 5: Add the isolated application fixture**

Append:

```python
@pytest.fixture
def app(mock_settings, tmp_path):
    settings = mock_settings.model_copy(deep=True)
    settings.storage.root = tmp_path / "api-data"
    return create_app(settings)
```

- [ ] **Step 6: Run HTTP tests**

Run: `.venv/bin/python -m pytest tests/integration/test_http_api.py -v`

Expected: PASS for health, diagnostics, offline success, silence 422, provider timeout 504, latest 404-before-run, and no latest overwrite after failure.

- [ ] **Step 7: Commit**

```bash
git add src/bionic_head/api tests/conftest.py tests/integration/test_http_api.py
git commit -m "feat: expose offline pipeline and diagnostics api"
```

### Task 7: WebSocket Event Models, Turn State Machine, and Session Admission

**Files:**
- Create: `src/bionic_head/protocol/__init__.py`
- Create: `src/bionic_head/protocol/events.py`
- Create: `src/bionic_head/core/state.py`
- Create: `tests/unit/test_events.py`
- Create: `tests/unit/test_state.py`

**Interfaces:**
- Produces: `EventEnvelope`, client payload models, `EventFactory`, `TurnState`, `TurnStateMachine`, `SessionManager`, and `TurnHandle`.

- [ ] **Step 1: Write failing event and state tests**

```python
# tests/unit/test_events.py
from uuid import uuid4

from bionic_head.protocol.events import EventFactory


def test_event_sequence_is_monotonic() -> None:
    factory = EventFactory(session_id=uuid4())
    turn_id = uuid4()
    first = factory.server("server.state", turn_id, {"state": "IDLE"})
    second = factory.server("server.pong", turn_id, {})
    assert first.protocol == "bionic-head-stream-v1"
    assert (first.sequence, second.sequence) == (1, 2)
```

```python
# tests/unit/test_state.py
import pytest

from bionic_head.core.state import TurnState, TurnStateMachine


def test_normal_and_cancel_transitions() -> None:
    machine = TurnStateMachine()
    machine.transition(TurnState.LISTENING)
    machine.transition(TurnState.THINKING)
    machine.transition(TurnState.CANCELLING)
    machine.transition(TurnState.IDLE)
    assert machine.state is TurnState.IDLE


def test_illegal_transition_is_rejected() -> None:
    machine = TurnStateMachine()
    with pytest.raises(ValueError):
        machine.transition(TurnState.SPEAKING)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_events.py tests/unit/test_state.py -v`

Expected: FAIL because protocol and state modules do not exist.

- [ ] **Step 3: Implement envelopes and strict payload validation**

Use this exact envelope shape:

```python
class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: Literal["bionic-head-stream-v1"]
    type: EventType
    event_id: UUID
    session_id: UUID
    turn_id: UUID | None
    sequence: int = Field(ge=1)
    timestamp: datetime
    payload: dict[str, object]
```

`turn_id` is null only for session-level events such as `client.session.start`, `server.session.ready`, and ping/pong before a turn begins. The client generates `session_id`, each new `turn_id`, event IDs, timestamps, and a client-direction sequence. The server validates that client sequence increments by one. The server maintains a separate server-direction sequence.

`EventEnvelope.type` is a string enum covering every client and server event in the design. `EventFactory.server()` creates UUID event IDs and timezone-aware UTC timestamps. Allocate server sequence values with one `itertools.count(start=1)` owned by the connection; the controller's send lock serializes event construction and transmission, so concurrent senders cannot interleave or duplicate sequence values.

Define client payload models for session start, audio start, audio chunk metadata (`byte_length`, `duration_ms`), audio end, cancel, and ping. Reject unknown payload fields with `ConfigDict(extra="forbid")`.

- [ ] **Step 4: Implement state and session admission**

Use:

```python
class TurnState(StrEnum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    CANCELLING = "CANCELLING"
    ERROR = "ERROR"
```

`TurnHandle` owns session ID, turn ID, `CancellationToken`, active task, terminal event flag, and one async lock.

```python
async def emit_if_current(self, operation: Callable[[], Awaitable[None]]) -> bool:
    async with self._lock:
        if not self.current or self.cancellation.cancelled:
            return False
        await operation()
        return True

async def commit_if_current(self, callback: Callable[[], None]) -> bool:
    async with self._lock:
        if not self.current or self.cancellation.cancelled:
            return False
        callback()
        return True
```

`emit_terminal_once(type)` uses the same lock and atomically returns whether the caller won terminal publication. Cancellation acquires the lock before invalidating the handle, so any event that completed before cancellation is ordered before `server.turn.cancelled`, and no old event can begin after cancellation.

`SessionManager.admit(session_id)` is an async context manager protected by a lock; when one session is active, a second admission raises `session_limit_reached`. Disconnect always releases admission.

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_events.py tests/unit/test_state.py -v`

Expected: PASS, including concurrent sequence allocation, invalid payloads, illegal transitions, one-session limit, and terminal-event uniqueness.

- [ ] **Step 6: Commit**

```bash
git add src/bionic_head/protocol src/bionic_head/core/state.py tests/unit
git commit -m "feat: define websocket protocol and turn state"
```

### Task 8: Sentence Segmentation and Mock Stream Orchestrator

**Files:**
- Create: `src/bionic_head/core/sentences.py`
- Create: `src/bionic_head/orchestrators/stream.py`
- Create: `tests/unit/test_sentences.py`
- Create: `tests/unit/test_stream_orchestrator.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces:
  - `SentenceBuffer.push(token: str) -> list[str]`
  - `SentenceBuffer.flush() -> str | None`
  - `StreamOrchestrator.run(input_wav, turn, emit_json, emit_binary_pair)`.
- Consumes: registry, timeline, artifact store, event factory, and current-turn guard.

- [ ] **Step 1: Write failing segmentation tests**

```python
# tests/unit/test_sentences.py
from bionic_head.core.sentences import SentenceBuffer


def test_punctuation_and_max_chars_emit_segments() -> None:
    buffer = SentenceBuffer(max_chars=4)
    assert buffer.push("你好。") == ["你好。"]
    assert buffer.push("12345") == ["1234"]
    assert buffer.flush() == "5"
```

- [ ] **Step 2: Write the failing stream-order test**

```python
# tests/unit/test_stream_orchestrator.py
import pytest


@pytest.mark.asyncio
async def test_stream_emits_audio_before_face_then_segment_ready(stream_harness) -> None:
    await stream_harness.run()
    types = stream_harness.json_types
    assert types.index("server.asr.final") < types.index("server.llm.token")
    assert types.index("server.tts.audio") < types.index("server.face.frames")
    assert types.index("server.face.frames") < types.index("server.segment.ready")
    assert types[-1] == "server.pipeline.done"
    assert len(stream_harness.binary_frames) >= 1
```

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_sentences.py tests/unit/test_stream_orchestrator.py -v`

Expected: FAIL because the sentence buffer and stream orchestrator do not exist.

- [ ] **Step 4: Implement segmentation**

`SentenceBuffer` must:

- Emit through and including `。！？!?；;\n`.
- Emit exactly `max_chars` characters if no punctuation is present.
- Keep residual text.
- Strip only surrounding whitespace, never internal punctuation.
- Return no empty segments.

The orchestrator handles the 500 ms idle rule by wrapping `anext(llm_iterator)` in `asyncio.wait_for(anext(llm_iterator), sentence_max_wait_ms / 1000)`. On timeout, it flushes a non-empty buffer and continues waiting unless the LLM iterator is exhausted.

- [ ] **Step 5: Implement sequential segment processing**

For each segment:

1. Emit every `server.llm.token`.
2. Emit `server.llm.chunk` with `chunk_id`.
3. Run TTS.
4. Call `emit_binary_pair(server.tts.audio metadata, wav_bytes)` so metadata and exactly one WAV binary frame are sent under one send lock.
5. Run Audio2Face.
6. Emit `server.face.frames`.
7. Format UE5 and emit chunks of at most 30 frames.
8. Emit `server.segment.ready`.

Mark timeline points `audio_end`, `asr_final`, `llm_first_token`, `first_tts_ready`, `first_face_ready`, and `first_segment_ready` exactly once. Before every adapter call and artifact index write, call `turn.cancellation.raise_if_cancelled()` and verify `turn.is_current()`. Send every nonterminal JSON/binary operation through `await turn.emit_if_current(operation)`, and publish latest through `await turn.commit_if_current(commit_callback)`.

Every server event payload repeats `session_id` and `turn_id` from the envelope. On the first `server.segment.ready`, transition THINKING to SPEAKING. After the unique terminal event, the connection controller returns the state to IDLE.

Use a single `try/except/finally`:

- `CancelledError` emits `server.turn.cancelled` once.
- `PipelineException` emits `server.pipeline.error` once with safe detail.
- Other exceptions are logged server-side and mapped to `internal_error`.
- Success emits `server.pipeline.done` once.
- `timeline.json` is always written.

- [ ] **Step 6: Add a reusable stream harness**

In `tests/conftest.py`, add a `StreamHarness` that creates a current `TurnHandle`, records JSON envelopes in order, records binary frames separately, and invokes `StreamOrchestrator.run`. Its `emit_json` and `emit_binary_pair` callbacks must use one async lock, matching production send serialization; the pair callback appends metadata and binary without releasing the lock.

Expose:

```python
@pytest.fixture
def stream_harness(mock_settings, mock_registry, artifact_store, speech_wav):
    return StreamHarness(
        settings=mock_settings,
        registry=mock_registry,
        store=artifact_store,
        input_wav=speech_wav,
    )
```

- [ ] **Step 7: Run focused stream tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sentences.py tests/unit/test_stream_orchestrator.py -v`

Expected: PASS for punctuation, max characters, idle flush, event order, binary pairing, provider failure, timeout, cancellation, stale output suppression, frame chunking, and terminal uniqueness.

- [ ] **Step 8: Commit**

```bash
git add src/bionic_head/core/sentences.py src/bionic_head/orchestrators/stream.py tests/conftest.py tests/unit
git commit -m "feat: implement mock pseudo streaming orchestrator"
```

### Task 9: WebSocket Connection Controller and Route

**Files:**
- Create: `src/bionic_head/protocol/connection.py`
- Create: `src/bionic_head/api/routes/stream.py`
- Modify: `src/bionic_head/api/app.py`
- Modify: `src/bionic_head/api/dependencies.py`
- Create: `tests/integration/test_websocket_api.py`

**Interfaces:**
- Produces: `WS /pipeline/stream`, strict JSON/binary pairing, audio watchdog, explicit cancel, and barge-in.
- Consumes: `StreamOrchestrator`, `SessionManager`, event models, audio helpers.

- [ ] **Step 1: Write the failing normal-flow WebSocket test**

```python
# tests/integration/test_websocket_api.py
import itertools
import json
from uuid import uuid4
from fastapi.testclient import TestClient


def test_websocket_mock_turn_reaches_done(app, speech_pcm) -> None:
    session_id = uuid4()
    turn_id = uuid4()
    sequence = itertools.count(1)
    with TestClient(app).websocket_connect("/pipeline/stream") as ws:
        ws.send_json(client_event("client.session.start", session_id, None, next(sequence), {}))
        ready = ws.receive_json()
        ws.send_json(client_event("client.audio.start", session_id, turn_id, next(sequence), {}))
        ws.send_json(client_event(
            "client.audio.chunk",
            session_id,
            turn_id,
            next(sequence),
            {"byte_length": len(speech_pcm), "duration_ms": 100},
        ))
        ws.send_bytes(speech_pcm)
        ws.send_json(client_event("client.audio.end", session_id, turn_id, next(sequence), {}))
        events = receive_until_terminal(ws)
    assert ready["type"] == "server.session.ready"
    assert events[-1]["type"] == "server.pipeline.done"
```

- [ ] **Step 2: Add failing protocol and cancellation tests**

Define `client_event(event_type, session_id, turn_id, sequence, payload)` and `receive_until_terminal(ws)` in the test module. `client_event` must populate all envelope fields with a UTC timestamp and a fresh event UUID. `receive_until_terminal` must consume a binary frame immediately after any `server.tts.audio` event, store binaries separately, and stop only at one of the three terminal event types.

Add exact cases:

- chunk metadata followed by JSON instead of binary → `server.pipeline.error` with `protocol_violation`;
- binary without pending metadata → protocol violation;
- binary byte length differs → protocol violation;
- second `client.audio.start` while THINKING cancels old turn and creates a new turn;
- explicit cancel emits `server.turn.cancelled`;
- stale delayed Mock output never appears after cancellation;
- a second simultaneous session receives `session_limit_reached`;
- sequence values are strictly increasing.

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/integration/test_websocket_api.py -v`

Expected: FAIL because the WebSocket route/controller does not exist.

- [ ] **Step 4: Implement the connection controller**

`StreamConnection` owns:

```python
self.websocket
self.session_id
self.event_factory
self.state_machine
self.current_turn
self.pending_binary_metadata
self.pcm_buffer
self.last_non_silent_monotonic
self.turn_started_monotonic
self.watchdog_task
```

Rules:

- Require `client.session.start` before audio.
- Require the same session ID on every client event and the current turn ID on every turn event.
- Require client sequence to increment by exactly one.
- On `client.audio.chunk`, store exactly one pending metadata object.
- The next frame must be binary with matching length.
- Validate chunk duration is 20–100 ms.
- Append PCM only while LISTENING.
- Update `last_non_silent_monotonic` when normalized chunk RMS is above threshold.
- The watchdog finalizes after 1000 ms continuous silence or 30 seconds total turn duration.
- `client.audio.end` finalizes immediately.
- Finalization writes `input.wav`, transitions to THINKING, and launches one stream task.
- New `client.audio.start` in THINKING/SPEAKING cancels the old turn before creating the new one.
- Disconnect cancels the active turn and releases session admission.

The route must use a single reader loop and a send lock so a JSON metadata frame and its immediately following binary frame cannot be interleaved with another sender.

- [ ] **Step 5: Run WebSocket integration tests**

Run: `.venv/bin/python -m pytest tests/integration/test_websocket_api.py -v`

Expected: PASS for normal flow, pairing violations, explicit end, watchdog end, max duration, cancel, barge-in, stale suppression, session limit, and sequence.

- [ ] **Step 6: Run the entire P0 suite**

Run: `.venv/bin/python -m pytest -m 'not integration' -v`

Expected: all unit and Mock HTTP/WebSocket integration tests PASS. Here `integration` means real-provider tests only; local API tests must not carry that marker.

- [ ] **Step 7: Commit**

```bash
git add src/bionic_head/protocol/connection.py src/bionic_head/api tests/integration
git commit -m "feat: expose cancellable websocket pipeline"
```

### Task 10: Operational Documentation and P0 Acceptance

**Files:**
- Create: `README.md`
- Create: `docs/protocols/bionic-head-stream-v1.md`
- Create: `docs/protocols/bionic-head-ue5-v1.md`
- Create: `docs/operations/mock-development.md`
- Create: `tests/acceptance/test_p0_acceptance.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces: exact local setup/start/test instructions and executable P0 acceptance coverage.

- [ ] **Step 1: Write the failing acceptance test**

```python
# tests/acceptance/test_p0_acceptance.py
def test_p0_acceptance(app, speech_wav, websocket_turn):
    offline = post_audio(app, speech_wav)
    assert offline.status_code == 200
    assert offline.json()["face"]["channel_count"] == 52

    events = websocket_turn(app)
    assert terminal_types(events) == ["server.pipeline.done"]
    assert strictly_increasing_sequences(events)
```

- [ ] **Step 2: Run acceptance test**

Run: `.venv/bin/python -m pytest tests/acceptance/test_p0_acceptance.py -v`

Expected: FAIL until helper fixtures and final response fields match the documented contract.

- [ ] **Step 3: Complete docs and acceptance fixtures**

Implement `post_audio`, `terminal_types`, `strictly_increasing_sequences`, and `websocket_turn` as shared test helpers in `tests/conftest.py`; they must call only public HTTP/WebSocket interfaces and must not reach into `app.state`.

README must include:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/uvicorn bionic_head.api.app:create_app --factory --host 127.0.0.1 --port 8000
.venv/bin/python -m pytest -m 'not integration'
```

Protocol docs must list every event, required payload fields, JSON/binary pairing, terminal uniqueness, cancellation behavior, PCM format, WAV behavior, and UE5 52-channel schema. `mock-development.md` must show how to configure delay, failure, timeout, and fixed output.

- [ ] **Step 4: Run acceptance and full tests**

Run: `.venv/bin/python -m pytest -m 'not integration' -v`

Expected: PASS.

Run: `.venv/bin/python -m pytest --collect-only -q`

Expected: collection succeeds with no unknown markers or import errors.

- [ ] **Step 5: Manually smoke-start the app**

Run: `timeout 5s .venv/bin/uvicorn bionic_head.api.app:create_app --factory --host 127.0.0.1 --port 8000`

Expected: Uvicorn reports startup complete; timeout stops it after five seconds without traceback.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/protocols docs/operations tests/conftest.py tests/acceptance
git commit -m "docs: document and verify p0 mock service"
```

## P0 Completion Gate

Before starting P1, verify:

```bash
git status --short
.venv/bin/python -m pytest -m 'not integration' -v
.venv/bin/python -m pytest --collect-only -q
```

Expected:

- clean worktree;
- all Mock unit, HTTP, WebSocket, and acceptance tests pass;
- real-provider tests are collected but skipped/deselected by default;
- `POST /pipeline/audio` and `WS /pipeline/stream` both work with `config/mock.json`;
- cancelled/stale turns never overwrite latest;
- no P1 provider package is required to import or start the P0 service.
