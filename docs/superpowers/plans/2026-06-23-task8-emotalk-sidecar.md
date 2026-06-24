# Task 8 EmoTalk Sidecar Implementation Plan

> **Superseded, 2026-06-24:** This early Task 8 plan described an HTTP sidecar/provider name `emotalk-sidecar`. The implemented production path now uses provider `emotalk_sidecar`, `sidecar_command`, and the stdin/stdout binary protocol `emotalk-sidecar-v1` in `src/bionic_head/sidecar_protocol.py`. Use `docs/status/2026-06-24-current-state.md` plus the Task 10.5/Task 11 plans as the current source of truth.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an `emotalk-sidecar` Audio2Face provider and a Python-stdlib EmoTalk sidecar process so EmoTalk loads once and each stream segment sends a local binary inference request.

**Architecture:** The main FastAPI app keeps its current public HTTP/WebSocket protocols and adds a new Audio2Face adapter that calls a localhost sidecar over binary-framed HTTP. The sidecar runs inside the existing `emotalk` Conda env, loads EmoTalk once at startup, accepts WAV bytes, returns raw float32 `[N, 52]` frames, and reports fine-grained timings. The existing `emotalk` subprocess provider remains available as fallback.

**Tech Stack:** Python 3.11 main service, Python 3.8-compatible sidecar script, FastAPI/Uvicorn main app, Pydantic v2 config, asyncio/httpx adapter, NumPy, PyTorch inside sidecar, pytest.

## Global Constraints

- Work on branch `task8-emotalk-sidecar`.
- Keep existing public `/pipeline/audio`, `/pipeline/stream`, `/ue5/latest`, and WebSocket event type names unchanged.
- Keep existing `audio2face.provider = "emotalk"` command provider unchanged as fallback.
- Add new provider name exactly `emotalk-sidecar`.
- Use protocol name exactly `bionic-head-emotalk-sidecar-v1`.
- Binary sidecar messages are `uint32 big-endian header_length + UTF-8 JSON header + raw body bytes`.
- First version accepts complete WAV bytes from the main service; raw 16k PCM is a later task.
- Sidecar response body is contiguous row-major `float32[N, 52]`.
- Sidecar script must be Python 3.8 compatible because Conda env `emotalk` is Python 3.8.8.
- `src/bionic_head/sidecar_protocol.py` must not import Pydantic, FastAPI, httpx, torch, or numpy.
- Automated tests must not require real EmoTalk, Conda, GPU, Ollama, Piper, or network access.
- Real acceptance benchmark compares `audio2face.provider = emotalk` against `audio2face.provider = emotalk-sidecar`.

---

## File Structure

```text
src/bionic_head/sidecar_protocol.py
  Python 3.8-compatible helper for binary-framed sidecar messages.

src/bionic_head/config.py
  Adds EmoTalkSidecarSettings and providers.emotalk_sidecar.

src/bionic_head/domain/models.py
  Adds optional provider_timings_ms to FaceArtifact so stream benchmark can see sidecar timings.

src/bionic_head/adapters/emotalk_sidecar.py
  Main app Audio2Face adapter that calls the sidecar and converts response frames into FaceArtifact.

src/bionic_head/adapters/registry.py
  Registers provider name "emotalk-sidecar".

scripts/emotalk_sidecar.py
  Python-stdlib HTTP sidecar server intended to run inside conda env emotalk.

config/emotalk-sidecar.example.json
  Real local config using faster-whisper, Ollama, Piper, emotalk-sidecar, morpheus-raw.

scripts/stream_client.py
  Records optional provider timings from server.face.frames.

scripts/benchmark.py
  Promotes sidecar timing summary into latency report metrics.

docs/operations/real-providers.md
  Adds sidecar startup, health check, benchmark, and fallback instructions.

tests/unit/test_sidecar_protocol.py
tests/unit/test_emotalk_sidecar_adapter.py
tests/unit/test_emotalk_sidecar_script.py
tests/unit/test_config.py
tests/unit/test_real_registry.py
tests/unit/test_stream_client.py
tests/unit/test_benchmark.py
```

---

### Task 1: Binary Sidecar Protocol Helper

**Files:**
- Create: `src/bionic_head/sidecar_protocol.py`
- Create: `tests/unit/test_sidecar_protocol.py`

**Interfaces:**
- Produces:
  - `PROTOCOL_VERSION: str`
  - `encode_message(header: dict[str, object], body: bytes) -> bytes`
  - `decode_message(payload: bytes) -> tuple[dict[str, object], bytes]`
  - `read_message(stream: BinaryIO) -> tuple[dict[str, object], bytes]`
  - `write_message(stream: BinaryIO, header: dict[str, object], body: bytes) -> None`
- Consumes: only Python standard library.

- [ ] **Step 1: Write failing protocol tests**

Create `tests/unit/test_sidecar_protocol.py`:

```python
from __future__ import annotations

import io
import struct

import pytest

from bionic_head.sidecar_protocol import (
    PROTOCOL_VERSION,
    decode_message,
    encode_message,
    read_message,
    write_message,
)


def test_encode_decode_round_trip() -> None:
    header = {
        "protocol": PROTOCOL_VERSION,
        "ok": True,
        "shape": [2, 52],
        "timings_ms": {"total": 12.5},
    }
    body = b"abc123"

    payload = encode_message(header, body)
    decoded_header, decoded_body = decode_message(payload)

    assert decoded_header == header
    assert decoded_body == body


def test_message_prefix_is_big_endian_header_length() -> None:
    header = {"protocol": PROTOCOL_VERSION}
    payload = encode_message(header, b"body")

    header_length = struct.unpack(">I", payload[:4])[0]

    assert header_length == len(payload) - 4 - len(b"body")


def test_stream_helpers_round_trip() -> None:
    stream = io.BytesIO()
    header = {"protocol": PROTOCOL_VERSION, "request_id": "req-1"}

    write_message(stream, header, b"frames")
    stream.seek(0)

    assert read_message(stream) == (header, b"frames")


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00\x00\x00",
        b"\x00\x00\x00\x10{}",
        b"\x00\x00\x00\x02[]",
        b"\x00\x00\x00\x08not-json",
    ],
)
def test_decode_rejects_malformed_messages(payload: bytes) -> None:
    with pytest.raises(ValueError):
        decode_message(payload)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_sidecar_protocol.py -q
```

Expected: FAIL because `bionic_head.sidecar_protocol` does not exist.

- [ ] **Step 3: Implement protocol helper**

Create `src/bionic_head/sidecar_protocol.py`:

```python
from __future__ import annotations

import json
import struct
from typing import BinaryIO, Tuple


PROTOCOL_VERSION = "bionic-head-emotalk-sidecar-v1"
_HEADER_PREFIX_SIZE = 4


def encode_message(header: dict[str, object], body: bytes) -> bytes:
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > 0xFFFFFFFF:
        raise ValueError("sidecar header is too large")
    return struct.pack(">I", len(header_bytes)) + header_bytes + body


def decode_message(payload: bytes) -> Tuple[dict[str, object], bytes]:
    if len(payload) < _HEADER_PREFIX_SIZE:
        raise ValueError("sidecar message is missing header length")
    header_length = struct.unpack(">I", payload[:_HEADER_PREFIX_SIZE])[0]
    header_start = _HEADER_PREFIX_SIZE
    header_end = header_start + header_length
    if header_length <= 0 or len(payload) < header_end:
        raise ValueError("sidecar message header is incomplete")
    try:
        decoded = json.loads(payload[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("sidecar message header is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("sidecar message header must be a JSON object")
    return decoded, payload[header_end:]


def read_message(stream: BinaryIO) -> Tuple[dict[str, object], bytes]:
    prefix = stream.read(_HEADER_PREFIX_SIZE)
    if len(prefix) != _HEADER_PREFIX_SIZE:
        raise ValueError("sidecar stream ended before header length")
    header_length = struct.unpack(">I", prefix)[0]
    if header_length <= 0:
        raise ValueError("sidecar stream header length must be positive")
    header_bytes = stream.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError("sidecar stream ended before header")
    try:
        decoded = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("sidecar stream header is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("sidecar stream header must be a JSON object")
    body = stream.read()
    return decoded, body


def write_message(stream: BinaryIO, header: dict[str, object], body: bytes) -> None:
    stream.write(encode_message(header, body))
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_sidecar_protocol.py -q
```

Expected: all tests in `tests/unit/test_sidecar_protocol.py` pass.

- [ ] **Step 5: Commit**

```bash
git add src/bionic_head/sidecar_protocol.py tests/unit/test_sidecar_protocol.py
git commit -m "feat: add emotalk sidecar binary protocol"
```

---

### Task 2: Config, Domain Model, and Registry Plumbing

**Files:**
- Modify: `src/bionic_head/config.py`
- Modify: `src/bionic_head/domain/models.py`
- Modify: `src/bionic_head/adapters/registry.py`
- Create: `config/emotalk-sidecar.example.json`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_real_registry.py`

**Interfaces:**
- Consumes:
  - `EmoTalkSidecarAudio2FaceAdapter.from_settings(settings: EmoTalkSidecarSettings)`.
- Produces:
  - `EmoTalkSidecarSettings`
  - `ProvidersSettings.emotalk_sidecar`
  - `FaceArtifact.provider_timings_ms`
  - registry support for provider `"emotalk-sidecar"`.

- [ ] **Step 1: Write failing config and registry tests**

Append to `tests/unit/test_config.py`:

```python
def test_load_emotalk_sidecar_example_settings() -> None:
    settings = load_settings(Path("config/emotalk-sidecar.example.json"))

    assert settings.adapters.audio2face.provider == "emotalk-sidecar"
    assert settings.adapters.audio2face.timeout_seconds == 120
    assert settings.adapters.ue5.provider == "morpheus-raw"
    assert str(settings.providers.emotalk_sidecar.base_url) == "http://127.0.0.1:8013/"
    assert settings.providers.emotalk_sidecar.timeout_seconds == 120
    assert settings.providers.emotalk_sidecar.output_npy_name == "emotalk.npy"
    assert settings.providers.emotalk_sidecar.fps == 30
    assert settings.providers.emotalk_sidecar.level == 1
    assert settings.providers.emotalk_sidecar.person == 0
```

Append to `tests/unit/test_real_registry.py`:

```python
def test_registry_constructs_emotalk_sidecar_example_without_network() -> None:
    settings = load_settings(Path("config/emotalk-sidecar.example.json"))

    registry = build_registry(settings)

    assert registry.asr.name == "faster-whisper"
    assert registry.llm.name == "ollama"
    assert registry.tts.name == "piper"
    assert registry.audio2face.name == "emotalk-sidecar"
    assert registry.ue5.name == "morpheus-raw"
```

Append to `tests/unit/test_domain.py`:

```python
def test_face_artifact_accepts_provider_timings() -> None:
    face = FaceArtifact(
        frames=[[0.0] * 52],
        frame_count=1,
        provider_timings_ms={"predict": 12.5, "total": 13.0},
    )

    assert face.provider_timings_ms["predict"] == 12.5
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_config.py::test_load_emotalk_sidecar_example_settings \
  tests/unit/test_real_registry.py::test_registry_constructs_emotalk_sidecar_example_without_network \
  tests/unit/test_domain.py::test_face_artifact_accepts_provider_timings \
  -q
```

Expected: FAIL because config field, config file, registry provider, and `FaceArtifact.provider_timings_ms` do not exist.

- [ ] **Step 3: Add settings and domain field**

Modify `src/bionic_head/config.py`:

```python
class EmoTalkSidecarSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: AnyHttpUrl = "http://127.0.0.1:8013"
    timeout_seconds: float = Field(default=120.0, gt=0)
    output_npy_name: str = "emotalk.npy"
    fps: int = Field(default=30, ge=1)
    level: int = Field(default=1, ge=0)
    person: int = Field(default=0, ge=0)
```

Add to `ProvidersSettings`:

```python
emotalk_sidecar: EmoTalkSidecarSettings = Field(default_factory=EmoTalkSidecarSettings)
```

Modify `src/bionic_head/domain/models.py` in `FaceArtifact`:

```python
provider_timings_ms: dict[str, float] = Field(default_factory=dict)
```

- [ ] **Step 4: Add example config**

Create `config/emotalk-sidecar.example.json` by copying `config/emotalk.example.json`, then change only:

```json
"audio2face": {"provider": "emotalk-sidecar", "timeout_seconds": 120}
```

and add under `"providers"`:

```json
"emotalk_sidecar": {
  "base_url": "http://127.0.0.1:8013",
  "timeout_seconds": 120,
  "output_npy_name": "emotalk.npy",
  "fps": 30,
  "level": 1,
  "person": 0
}
```

Keep the old `"emotalk"` provider section in the file so users can switch back without rebuilding local paths.

- [ ] **Step 5: Register provider**

Modify `_build_audio2face` in `src/bionic_head/adapters/registry.py`:

```python
if settings.adapters.audio2face.provider == "emotalk-sidecar":
    from bionic_head.adapters.emotalk_sidecar import EmoTalkSidecarAudio2FaceAdapter

    return EmoTalkSidecarAudio2FaceAdapter.from_settings(settings.providers.emotalk_sidecar)
```

- [ ] **Step 6: Verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_config.py::test_load_emotalk_sidecar_example_settings \
  tests/unit/test_real_registry.py::test_registry_constructs_emotalk_sidecar_example_without_network \
  tests/unit/test_domain.py::test_face_artifact_accepts_provider_timings \
  -q
```

Expected: the three tests pass after Task 3 creates the adapter module. If Task 3 has not run yet, the registry test still fails with `ModuleNotFoundError`; commit Task 2 after adding the minimal adapter shell below.

Create a minimal shell `src/bionic_head/adapters/emotalk_sidecar.py` only if needed for Task 2 tests:

```python
from __future__ import annotations

from bionic_head.config import EmoTalkSidecarSettings


class EmoTalkSidecarAudio2FaceAdapter:
    name = "emotalk-sidecar"

    def __init__(self, settings: EmoTalkSidecarSettings) -> None:
        self.settings = settings
        self.call_count = 0

    @classmethod
    def from_settings(
        cls,
        settings: EmoTalkSidecarSettings,
    ) -> "EmoTalkSidecarAudio2FaceAdapter":
        return cls(settings)
```

- [ ] **Step 7: Commit**

```bash
git add src/bionic_head/config.py src/bionic_head/domain/models.py src/bionic_head/adapters/registry.py src/bionic_head/adapters/emotalk_sidecar.py config/emotalk-sidecar.example.json tests/unit/test_config.py tests/unit/test_real_registry.py tests/unit/test_domain.py
git commit -m "feat: register emotalk sidecar provider"
```

---

### Task 3: Main App EmoTalk Sidecar Adapter

**Files:**
- Modify: `src/bionic_head/adapters/emotalk_sidecar.py`
- Create: `tests/unit/test_emotalk_sidecar_adapter.py`

**Interfaces:**
- Consumes:
  - `EmoTalkSidecarSettings`
  - `encode_message`, `decode_message`, `PROTOCOL_VERSION`
  - `AudioArtifact`, `FaceArtifact`, `TurnContext`
- Produces:
  - Fully functional `EmoTalkSidecarAudio2FaceAdapter`
  - `diagnostics()` health check
  - sidecar inference response validation.

- [ ] **Step 1: Write failing success-path adapter test**

Create `tests/unit/test_emotalk_sidecar_adapter.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from bionic_head.adapters.emotalk_sidecar import EmoTalkSidecarAudio2FaceAdapter
from bionic_head.config import EmoTalkSidecarSettings
from bionic_head.core.audio import audio_artifact_from_wav
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import Emotion
from bionic_head.sidecar_protocol import PROTOCOL_VERSION, encode_message


def _settings() -> EmoTalkSidecarSettings:
    return EmoTalkSidecarSettings(
        base_url="http://sidecar.local",
        timeout_seconds=2,
        output_npy_name="emotalk.npy",
        fps=30,
        level=1,
        person=0,
    )


def _adapter(transport: httpx.AsyncBaseTransport) -> EmoTalkSidecarAudio2FaceAdapter:
    return EmoTalkSidecarAudio2FaceAdapter(_settings(), transport=transport)


def _sidecar_response(array: np.ndarray, *, timings: dict[str, float] | None = None) -> httpx.Response:
    header = {
        "protocol": PROTOCOL_VERSION,
        "ok": True,
        "request_id": "req",
        "dtype": "float32",
        "shape": list(array.shape),
        "fps": 30,
        "channel_count": 52,
        "timings_ms": timings or {"predict": 12.5, "total": 13.0},
    }
    return httpx.Response(200, content=encode_message(header, array.astype(np.float32).tobytes()))


@pytest.mark.asyncio
async def test_drive_posts_wav_and_writes_face_artifact(speech_wav: Path, turn_context) -> None:
    array = np.ones((3, 52), dtype=np.float32) * 0.25
    seen_request = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_request["method"] = request.method
        seen_request["path"] = request.url.path
        seen_request["body_prefix"] = request.content[:4]
        return _sidecar_response(array)

    adapter = _adapter(httpx.MockTransport(handler))
    audio = audio_artifact_from_wav(speech_wav)

    face = await adapter.drive(audio, Emotion.FRIENDLY, 0.8, turn_context)

    assert seen_request["method"] == "POST"
    assert seen_request["path"] == "/infer"
    assert len(seen_request["body_prefix"]) == 4
    assert face.frame_count == 3
    assert face.channel_count == 52
    assert face.frames[0][0] == pytest.approx(0.25)
    assert face.provider_timings_ms == {"predict": 12.5, "total": 13.0}
    assert face.path is not None
    assert face.path.name == "emotalk.npy"
    assert face.path.parent.name == "emotalk_sidecar_0001"
    assert np.load(face.path).shape == (3, 52)
    meta = json.loads((face.path.parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["provider"] == "emotalk-sidecar"
    assert meta["timings_ms"]["predict"] == pytest.approx(12.5)
```

- [ ] **Step 2: Run success test to verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_emotalk_sidecar_adapter.py::test_drive_posts_wav_and_writes_face_artifact -q
```

Expected: FAIL because the adapter shell has no `drive` implementation.

- [ ] **Step 3: Implement success path**

Replace `src/bionic_head/adapters/emotalk_sidecar.py` with a complete adapter:

```python
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from time import perf_counter
from uuid import UUID

import numpy as np

from bionic_head.config import EmoTalkSidecarSettings
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import AudioArtifact, DiagnosticResult, Emotion, FaceArtifact, TurnContext
from bionic_head.sidecar_protocol import PROTOCOL_VERSION, decode_message, encode_message

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


class EmoTalkSidecarAudio2FaceAdapter:
    name = "emotalk-sidecar"

    def __init__(
        self,
        settings: EmoTalkSidecarSettings,
        transport: object | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport
        self.call_count = 0

    @classmethod
    def from_settings(
        cls,
        settings: EmoTalkSidecarSettings,
    ) -> "EmoTalkSidecarAudio2FaceAdapter":
        return cls(settings)

    async def drive(
        self,
        audio: AudioArtifact,
        emotion: Emotion,
        intensity: float,
        context: TurnContext,
    ) -> FaceArtifact:
        if httpx is None:
            raise self._error(ErrorCode.PROVIDER_UNAVAILABLE, "httpx is not installed", retryable=False)
        context.cancellation.raise_if_cancelled()
        if not audio.path.exists() or audio.path.stat().st_size <= 0:
            raise self._error(ErrorCode.OUTPUT_VALIDATION_FAILED, "Input WAV is missing or empty", retryable=False)

        self.call_count += 1
        output_dir = context.artifact_dir / "face" / f"emotalk_sidecar_{self.call_count:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        request_id = f"{context.turn_id}:{self.call_count}"
        request_body = audio.path.read_bytes()
        request_header: dict[str, object] = {
            "protocol": PROTOCOL_VERSION,
            "request_id": request_id,
            "session_id": str(context.session_id),
            "turn_id": str(context.turn_id),
            "input_format": "wav",
            "byte_length": len(request_body),
            "fps": self.settings.fps,
            "level": self.settings.level,
            "person": self.settings.person,
            "emotion": emotion.value,
            "intensity": float(intensity),
        }
        payload = encode_message(request_header, request_body)

        try:
            async with self._client() as client:
                response = await client.post(
                    "/infer",
                    content=payload,
                    headers={"content-type": "application/octet-stream"},
                )
        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException as exc:
            raise self._error(ErrorCode.PROVIDER_TIMEOUT, "EmoTalk sidecar request timed out", retryable=True) from exc
        except httpx.RequestError as exc:
            raise self._error(ErrorCode.PROVIDER_UNAVAILABLE, "EmoTalk sidecar is unreachable", retryable=True) from exc

        context.cancellation.raise_if_cancelled()
        if response.status_code >= 400:
            raise self._error(ErrorCode.PROVIDER_FAILED, "EmoTalk sidecar returned an HTTP error", retryable=True)
        try:
            header, body = decode_message(response.content)
        except ValueError as exc:
            raise self._error(ErrorCode.OUTPUT_VALIDATION_FAILED, "EmoTalk sidecar returned an invalid message", retryable=False) from exc

        array, fps, timings = self._parse_success_response(header, body)
        npy_path = output_dir / self.settings.output_npy_name
        np.save(npy_path, array)
        meta_path = output_dir / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "provider": self.name,
                    "protocol": PROTOCOL_VERSION,
                    "fps": fps,
                    "shape": list(array.shape),
                    "timings_ms": timings,
                    "sidecar_header": header,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return FaceArtifact(
            path=npy_path,
            frames=array.astype(float).tolist(),
            fps=fps,
            channel_count=52,
            frame_count=int(array.shape[0]),
            auxiliary_paths=[meta_path],
            provider_timings_ms=timings,
        )

    def _client(self) -> "httpx.AsyncClient":
        if httpx is None:
            raise self._error(ErrorCode.PROVIDER_UNAVAILABLE, "httpx is not installed", retryable=False)
        return httpx.AsyncClient(
            base_url=str(self.settings.base_url),
            timeout=self.settings.timeout_seconds,
            transport=self._transport,
        )

    def _parse_success_response(
        self,
        header: dict[str, object],
        body: bytes,
    ) -> tuple[np.ndarray, int, dict[str, float]]:
        if header.get("protocol") != PROTOCOL_VERSION:
            raise self._error(ErrorCode.OUTPUT_VALIDATION_FAILED, "EmoTalk sidecar protocol mismatch", retryable=False)
        if header.get("ok") is not True:
            message = header.get("message")
            raise self._error(ErrorCode.PROVIDER_FAILED, str(message or "EmoTalk sidecar inference failed"), retryable=True)
        if header.get("dtype") != "float32":
            raise self._error(ErrorCode.OUTPUT_VALIDATION_FAILED, "EmoTalk sidecar dtype must be float32", retryable=False)
        shape = header.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not isinstance(shape[0], int)
            or not isinstance(shape[1], int)
            or shape[0] <= 0
            or shape[1] != 52
        ):
            raise self._error(ErrorCode.OUTPUT_VALIDATION_FAILED, "EmoTalk sidecar shape must be [N, 52]", retryable=False)
        expected_bytes = int(shape[0]) * int(shape[1]) * 4
        if len(body) != expected_bytes:
            raise self._error(ErrorCode.OUTPUT_VALIDATION_FAILED, "EmoTalk sidecar body length does not match shape", retryable=False)
        array = np.frombuffer(body, dtype=np.float32).reshape((int(shape[0]), int(shape[1]))).copy()
        if not np.isfinite(array).all():
            raise self._error(ErrorCode.OUTPUT_VALIDATION_FAILED, "EmoTalk sidecar frames must be finite", retryable=False)
        fps_value = header.get("fps", self.settings.fps)
        fps = int(fps_value) if isinstance(fps_value, (int, float)) and math.isfinite(float(fps_value)) and fps_value > 0 else self.settings.fps
        timings = self._parse_timings(header.get("timings_ms"))
        return array, fps, timings

    def _parse_timings(self, payload: object) -> dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        timings: dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, (int, float)) and math.isfinite(float(value)):
                timings[key] = float(value)
        return timings

    async def diagnostics(self) -> DiagnosticResult:
        started = perf_counter()
        if httpx is None:
            return self._diagnostic(False, started, "httpx is not installed")
        try:
            async with self._client() as client:
                response = await client.get("/health")
            if response.status_code >= 400:
                return self._diagnostic(False, started, "EmoTalk sidecar health endpoint returned an error")
            payload = response.json()
        except httpx.TimeoutException:
            return self._diagnostic(False, started, "EmoTalk sidecar diagnostics timed out")
        except httpx.RequestError:
            return self._diagnostic(False, started, "EmoTalk sidecar is unreachable")
        except ValueError:
            return self._diagnostic(False, started, "EmoTalk sidecar health response is invalid")
        if not isinstance(payload, dict) or payload.get("ok") is not True or payload.get("loaded") is not True:
            return self._diagnostic(False, started, "EmoTalk sidecar is not loaded")
        return self._diagnostic(True, started, "EmoTalk sidecar ready")

    def _diagnostic(self, available: bool, started: float, message: str) -> DiagnosticResult:
        return DiagnosticResult(
            adapter="audio2face",
            provider=self.name,
            available=available,
            latency_ms=(perf_counter() - started) * 1000.0,
            message=message,
        )

    async def cancel(self, turn_id: UUID) -> None:
        await asyncio.sleep(0)

    def _error(self, code: ErrorCode, message: str, *, retryable: bool) -> PipelineException:
        return PipelineException(
            code=code,
            stage="audio2face",
            provider=self.name,
            retryable=retryable,
            message=message,
        )
```

- [ ] **Step 4: Verify success path GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_emotalk_sidecar_adapter.py::test_drive_posts_wav_and_writes_face_artifact -q
```

Expected: success-path test passes.

- [ ] **Step 5: Add failing validation and diagnostics tests**

Append to `tests/unit/test_emotalk_sidecar_adapter.py`:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("header_patch", "body_factory", "expected_message"),
    [
        ({"dtype": "float64"}, lambda array: array.astype(np.float32).tobytes(), "dtype"),
        ({"shape": [3, 51]}, lambda array: array.astype(np.float32).tobytes(), "shape"),
        ({"shape": [3, 52]}, lambda array: array.astype(np.float32).tobytes()[:-4], "body length"),
    ],
)
async def test_drive_rejects_invalid_sidecar_output(
    speech_wav: Path,
    turn_context,
    header_patch: dict[str, object],
    body_factory,
    expected_message: str,
) -> None:
    array = np.ones((3, 52), dtype=np.float32)

    async def handler(request: httpx.Request) -> httpx.Response:
        header = {
            "protocol": PROTOCOL_VERSION,
            "ok": True,
            "dtype": "float32",
            "shape": [3, 52],
            "fps": 30,
            "channel_count": 52,
        }
        header.update(header_patch)
        return httpx.Response(200, content=encode_message(header, body_factory(array)))

    with pytest.raises(PipelineException) as raised:
        await _adapter(httpx.MockTransport(handler)).drive(
            audio_artifact_from_wav(speech_wav),
            Emotion.FRIENDLY,
            0.8,
            turn_context,
        )

    assert raised.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED
    assert expected_message in raised.value.safe_message


@pytest.mark.asyncio
async def test_drive_rejects_nan_frames(speech_wav: Path, turn_context) -> None:
    array = np.ones((3, 52), dtype=np.float32)
    array[0, 0] = np.nan

    async def handler(request: httpx.Request) -> httpx.Response:
        return _sidecar_response(array)

    with pytest.raises(PipelineException) as raised:
        await _adapter(httpx.MockTransport(handler)).drive(
            audio_artifact_from_wav(speech_wav),
            Emotion.FRIENDLY,
            0.8,
            turn_context,
        )

    assert raised.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED
    assert "finite" in raised.value.safe_message


@pytest.mark.asyncio
async def test_drive_maps_sidecar_failure_header(speech_wav: Path, turn_context) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=encode_message(
                {
                    "protocol": PROTOCOL_VERSION,
                    "ok": False,
                    "message": "model exploded",
                },
                b"",
            ),
        )

    with pytest.raises(PipelineException) as raised:
        await _adapter(httpx.MockTransport(handler)).drive(
            audio_artifact_from_wav(speech_wav),
            Emotion.FRIENDLY,
            0.8,
            turn_context,
        )

    assert raised.value.code is ErrorCode.PROVIDER_FAILED
    assert raised.value.retryable is True


@pytest.mark.asyncio
async def test_diagnostics_reports_ready() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"ok": True, "loaded": True})

    result = await _adapter(httpx.MockTransport(handler)).diagnostics()

    assert result.adapter == "audio2face"
    assert result.provider == "emotalk-sidecar"
    assert result.available is True
```

- [ ] **Step 6: Verify full adapter test module GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_emotalk_sidecar_adapter.py -q
```

Expected: all adapter tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/bionic_head/adapters/emotalk_sidecar.py tests/unit/test_emotalk_sidecar_adapter.py
git commit -m "feat: add emotalk sidecar adapter"
```

---

### Task 4: Standard-Library EmoTalk Sidecar Script

**Files:**
- Create: `scripts/emotalk_sidecar.py`
- Create: `tests/unit/test_emotalk_sidecar_script.py`

**Interfaces:**
- Consumes:
  - `bionic_head.sidecar_protocol`
  - existing `/home/user/code/EmoTalk_release/model.py` runtime only when started for real.
- Produces:
  - CLI parser
  - `/health` JSON endpoint
  - `/infer` binary endpoint
  - lazy runtime class that loads EmoTalk once.

- [ ] **Step 1: Write failing tests for parser, health payload, and fake inference**

Create `tests/unit/test_emotalk_sidecar_script.py`:

```python
from __future__ import annotations

import json

import numpy as np

from bionic_head.sidecar_protocol import PROTOCOL_VERSION, decode_message, encode_message
from scripts.emotalk_sidecar import (
    FakeRuntime,
    build_health_payload,
    build_parser,
    handle_infer_payload,
)


def test_parser_defaults_match_local_sidecar_contract() -> None:
    args = build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 8013
    assert str(args.emotalk_root) == "/home/user/code/EmoTalk_release"
    assert args.device == "cpu"
    assert args.torch_threads == 4
    assert args.torch_interop_threads == 1


def test_health_payload_reports_loaded_runtime() -> None:
    payload = build_health_payload(
        loaded=True,
        device="cpu",
        model_path="/models/EmoTalk.pth",
        torch_threads=4,
        torch_interop_threads=1,
    )

    assert payload["ok"] is True
    assert payload["loaded"] is True
    assert payload["provider"] == "emotalk-sidecar"
    assert payload["device"] == "cpu"


def test_handle_infer_payload_returns_binary_float32_frames() -> None:
    runtime = FakeRuntime(np.ones((2, 52), dtype=np.float32) * 0.5)
    request = encode_message(
        {
            "protocol": PROTOCOL_VERSION,
            "request_id": "req-1",
            "input_format": "wav",
            "byte_length": 4,
            "fps": 30,
            "level": 1,
            "person": 0,
        },
        b"RIFF",
    )

    response = handle_infer_payload(runtime, request)
    header, body = decode_message(response)
    frames = np.frombuffer(body, dtype=np.float32).reshape((2, 52))

    assert header["ok"] is True
    assert header["protocol"] == PROTOCOL_VERSION
    assert header["shape"] == [2, 52]
    assert header["dtype"] == "float32"
    assert header["timings_ms"]["total"] >= 0
    assert frames[0, 0] == 0.5


def test_handle_infer_payload_returns_failure_message_for_bad_protocol() -> None:
    runtime = FakeRuntime(np.zeros((1, 52), dtype=np.float32))
    response = handle_infer_payload(runtime, encode_message({"protocol": "bad"}, b""))

    header, body = decode_message(response)

    assert header["ok"] is False
    assert header["error_code"] == "invalid_request"
    assert body == b""
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_emotalk_sidecar_script.py -q
```

Expected: FAIL because `scripts/emotalk_sidecar.py` does not exist.

- [ ] **Step 3: Implement testable sidecar shell**

Create `scripts/emotalk_sidecar.py` with these sections:

```python
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter

import numpy as np

from bionic_head.sidecar_protocol import PROTOCOL_VERSION, decode_message, encode_message


DEFAULT_EMOTALK_ROOT = Path("/home/user/code/EmoTalk_release")


class FakeRuntime:
    def __init__(self, frames: np.ndarray) -> None:
        self.frames = frames.astype(np.float32)
        self.loaded = True
        self.device = "fake"
        self.model_path = "fake"
        self.torch_threads = 1
        self.torch_interop_threads = 1

    def infer(self, wav_bytes: bytes, *, fps: int, level: int, person: int) -> tuple[np.ndarray, dict[str, float]]:
        return self.frames, {"decode": 0.0, "resample": 0.0, "tensor": 0.0, "predict": 0.0, "serialize": 0.0, "total": 0.0}
```

Then add:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a persistent EmoTalk sidecar server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--emotalk-root", type=Path, default=DEFAULT_EMOTALK_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_EMOTALK_ROOT / "pretrain_model" / "EmoTalk.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    return parser


def build_health_payload(
    *,
    loaded: bool,
    device: str,
    model_path: str,
    torch_threads: int,
    torch_interop_threads: int,
) -> dict[str, object]:
    return {
        "ok": True,
        "loaded": loaded,
        "provider": "emotalk-sidecar",
        "device": device,
        "model_path": model_path,
        "torch_threads": torch_threads,
        "torch_interop_threads": torch_interop_threads,
    }
```

Then add a real runtime class that imports torch/librosa/model only inside `__init__`:

```python
class EmoTalkRuntime:
    def __init__(
        self,
        *,
        emotalk_root: Path,
        model_path: Path,
        device: str,
        torch_threads: int,
        torch_interop_threads: int,
        warmup_seconds: float,
    ) -> None:
        self.loaded = False
        self.device = device
        self.model_path = str(model_path)
        self.torch_threads = torch_threads
        self.torch_interop_threads = torch_interop_threads
        self._lock = threading.Lock()

        import torch

        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(torch_interop_threads)
        if str(emotalk_root) not in sys.path:
            sys.path.insert(0, str(emotalk_root))
        from model import EmoTalk

        class Args:
            pass

        args = Args()
        args.device = device
        args.bs_dim = 52
        args.feature_dim = 832
        args.period = 30
        args.max_seq_len = 5000
        args.batch_size = 1
        state = torch.load(str(model_path), map_location=torch.device(device))
        self._torch = torch
        self._model = EmoTalk(args)
        self._model.load_state_dict(state, strict=False)
        self._model = self._model.to(device)
        self._model.eval()
        self.loaded = True
        if warmup_seconds > 0:
            samples = np.zeros(int(16000 * warmup_seconds), dtype=np.float32)
            self._predict_samples(samples, fps=30, level=1, person=0)

    def infer(self, wav_bytes: bytes, *, fps: int, level: int, person: int) -> tuple[np.ndarray, dict[str, float]]:
        with self._lock:
            return self._infer_locked(wav_bytes, fps=fps, level=level, person=person)
```

Add helper methods:

```python
    def _infer_locked(self, wav_bytes: bytes, *, fps: int, level: int, person: int) -> tuple[np.ndarray, dict[str, float]]:
        started = perf_counter()
        decoded_started = perf_counter()
        samples, sample_rate = decode_wav_mono_float32(wav_bytes)
        decode_ms = (perf_counter() - decoded_started) * 1000.0
        resample_started = perf_counter()
        if sample_rate != 16000:
            samples = resample_to_16k(samples, sample_rate)
        resample_ms = (perf_counter() - resample_started) * 1000.0
        tensor_started = perf_counter()
        tensor = self._torch.FloatTensor(samples).unsqueeze(0).to(self.device)
        level_tensor = self._torch.tensor([level]).to(self.device)
        person_tensor = self._torch.tensor([person]).to(self.device)
        tensor_ms = (perf_counter() - tensor_started) * 1000.0
        predict_started = perf_counter()
        with self._torch.inference_mode():
            prediction = self._model.predict(tensor, level_tensor, person_tensor)
        predict_ms = (perf_counter() - predict_started) * 1000.0
        serialize_started = perf_counter()
        frames = prediction.squeeze(0).detach().cpu().numpy().astype(np.float32)
        validate_frames(frames)
        frames = np.ascontiguousarray(frames)
        serialize_ms = (perf_counter() - serialize_started) * 1000.0
        return frames, {
            "decode": decode_ms,
            "resample": resample_ms,
            "tensor": tensor_ms,
            "predict": predict_ms,
            "serialize": serialize_ms,
            "total": (perf_counter() - started) * 1000.0,
        }

    def _predict_samples(self, samples: np.ndarray, *, fps: int, level: int, person: int) -> np.ndarray:
        tensor = self._torch.FloatTensor(samples).unsqueeze(0).to(self.device)
        level_tensor = self._torch.tensor([level]).to(self.device)
        person_tensor = self._torch.tensor([person]).to(self.device)
        with self._torch.inference_mode():
            prediction = self._model.predict(tensor, level_tensor, person_tensor)
        frames = prediction.squeeze(0).detach().cpu().numpy().astype(np.float32)
        validate_frames(frames)
        return frames
```

Add pure helpers:

```python
def decode_wav_mono_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)
    if channels != 1:
        raise ValueError("input WAV must be mono")
    if sample_width != 2:
        raise ValueError("input WAV must be signed 16-bit PCM")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sample_rate


def resample_to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    return librosa.resample(samples, orig_sr=sample_rate, target_sr=16000).astype(np.float32)


def validate_frames(frames: np.ndarray) -> None:
    if frames.ndim != 2 or frames.shape[1] != 52 or frames.shape[0] <= 0:
        raise ValueError("EmoTalk output must have shape [N, 52]")
    if not np.isfinite(frames).all():
        raise ValueError("EmoTalk output must be finite")
```

Add inference payload handler:

```python
def handle_infer_payload(runtime, payload: bytes) -> bytes:
    request_id = None
    try:
        header, body = decode_message(payload)
        request_id = str(header.get("request_id", ""))
        if header.get("protocol") != PROTOCOL_VERSION:
            raise ValueError("protocol mismatch")
        if header.get("input_format") != "wav":
            raise ValueError("input_format must be wav")
        fps = int(header.get("fps", 30))
        level = int(header.get("level", 1))
        person = int(header.get("person", 0))
        frames, timings = runtime.infer(body, fps=fps, level=level, person=person)
        response_header = {
            "protocol": PROTOCOL_VERSION,
            "ok": True,
            "request_id": request_id,
            "dtype": "float32",
            "shape": list(frames.shape),
            "fps": fps,
            "channel_count": 52,
            "timings_ms": timings,
        }
        return encode_message(response_header, frames.astype(np.float32).tobytes())
    except Exception as exc:
        return encode_message(
            {
                "protocol": PROTOCOL_VERSION,
                "ok": False,
                "request_id": request_id or "",
                "error_code": "invalid_request",
                "message": str(exc),
            },
            b"",
        )
```

Add HTTP server:

```python
def make_handler(runtime):
    class EmoTalkSidecarHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            payload = build_health_payload(
                loaded=bool(runtime.loaded),
                device=str(runtime.device),
                model_path=str(runtime.model_path),
                torch_threads=int(runtime.torch_threads),
                torch_interop_threads=int(runtime.torch_interop_threads),
            )
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/infer":
                self.send_error(404)
                return
            length = int(self.headers.get("content-length", "0"))
            payload = self.rfile.read(length)
            body = handle_infer_payload(runtime, payload)
            self.send_response(200)
            self.send_header("content-type", "application/octet-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            print("sidecar:", format % args, file=sys.stderr)

    return EmoTalkSidecarHandler


def main() -> None:
    args = build_parser().parse_args()
    runtime = EmoTalkRuntime(
        emotalk_root=args.emotalk_root,
        model_path=args.model_path,
        device=args.device,
        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
        warmup_seconds=args.warmup_seconds,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"EmoTalk sidecar listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify sidecar script unit tests GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_emotalk_sidecar_script.py -q
```

Expected: all tests pass without loading real EmoTalk.

- [ ] **Step 5: Verify script help works in both environments**

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/emotalk_sidecar.py --help
PYTHONPATH=src /home/user/miniconda3/bin/conda run -n emotalk python scripts/emotalk_sidecar.py --help
```

Expected: both commands print CLI help and exit 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/emotalk_sidecar.py tests/unit/test_emotalk_sidecar_script.py
git commit -m "feat: add emotalk sidecar server"
```

---

### Task 5: Expose Sidecar Timings in Stream Client and Benchmark

**Files:**
- Modify: `src/bionic_head/orchestrators/stream.py`
- Modify: `scripts/stream_client.py`
- Modify: `scripts/benchmark.py`
- Modify: `tests/unit/test_stream_orchestrator.py`
- Create or modify: `tests/unit/test_stream_client.py`
- Modify: `tests/unit/test_benchmark.py`

**Interfaces:**
- Consumes:
  - `FaceArtifact.provider_timings_ms`
- Produces:
  - Optional `provider_timings_ms` in `server.face.frames` payload when adapter supplies timings.
  - `summary["face_provider_timings_ms"]` in stream client.
  - benchmark metrics named `sidecar_predict_ms` and `sidecar_total_ms` when available.

- [ ] **Step 1: Write failing stream orchestrator test**

Append to `tests/unit/test_stream_orchestrator.py`:

```python
@pytest.mark.asyncio
async def test_stream_face_frames_include_provider_timings_when_available(
    stream_harness_factory,
    mock_registry,
) -> None:
    class _TimedAudio2FaceAdapter:
        name = "timed-face"
        call_count = 0

        async def drive(self, audio, emotion, intensity, context):
            return FaceArtifact(
                frames=[[0.1] * 52],
                frame_count=1,
                provider_timings_ms={"predict": 11.0, "total": 12.0},
            )

        async def diagnostics(self):
            raise AssertionError("not used")

        async def cancel(self, turn_id):
            return None

    registry = replace(mock_registry, audio2face=_TimedAudio2FaceAdapter())
    harness = stream_harness_factory(registry=registry)

    await harness.run()

    face_events = [
        envelope for envelope in harness.json_envelopes
        if envelope.type.value == "server.face.frames"
    ]
    assert face_events
    assert face_events[0].payload["provider_timings_ms"] == {"predict": 11.0, "total": 12.0}
```

If `replace` is not imported in the test file, add:

```python
from dataclasses import replace
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_orchestrator.py::test_stream_face_frames_include_provider_timings_when_available -q
```

Expected: FAIL because `server.face.frames` payload lacks `provider_timings_ms`.

- [ ] **Step 3: Add optional timings to stream event**

Modify `_process_face_segment` in `src/bionic_head/orchestrators/stream.py`:

```python
payload = {
    "chunk_id": chunk_id,
    "fps": face.fps,
    "frame_count": face.frame_count,
    "frames": face.frames,
}
if face.provider_timings_ms:
    payload["provider_timings_ms"] = face.provider_timings_ms
await self._emit_server_json(
    turn,
    emit_json,
    factory,
    EventType.SERVER_FACE_FRAMES,
    payload,
)
```

- [ ] **Step 4: Write failing client and benchmark tests**

Create `tests/unit/test_stream_client.py` if missing:

```python
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from scripts.stream_client import ClientReceiver


def _envelope(event_type: str, sequence: int, session_id, turn_id, payload: dict[str, object]) -> dict[str, object]:
    return {
        "protocol": "bionic-head-stream-v1",
        "type": event_type,
        "event_id": str(uuid4()),
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "sequence": sequence,
        "timestamp": "2026-06-23T00:00:00+00:00",
        "payload": payload,
    }


def test_client_summary_records_first_face_provider_timings(tmp_path: Path) -> None:
    session_id = uuid4()
    turn_id = uuid4()
    receiver = ClientReceiver(tmp_path, session_id=session_id, turn_id=turn_id, clock=lambda: 1.0)

    receiver.accept_json(
        _envelope(
            "server.face.frames",
            1,
            session_id,
            turn_id,
            {
                "chunk_id": "chunk-0001",
                "fps": 30,
                "frame_count": 1,
                "frames": [[0.0] * 52],
                "provider_timings_ms": {"predict": 11.0, "total": 12.0},
            },
        )
    )

    assert receiver.summary["face_provider_timings_ms"] == {"predict": 11.0, "total": 12.0}
```

Append to `tests/unit/test_benchmark.py`:

```python
def test_stream_metrics_include_sidecar_timings_when_available() -> None:
    metrics = stream_metrics_from_summary(
        {
            "face_provider_timings_ms": {
                "predict": 11.0,
                "total": 12.0,
            }
        },
        wall_ms=1000.0,
    )

    assert metrics["sidecar_predict_ms"] == 11.0
    assert metrics["sidecar_total_ms"] == 12.0
```

- [ ] **Step 5: Run client and benchmark tests to verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_stream_client.py::test_client_summary_records_first_face_provider_timings \
  tests/unit/test_benchmark.py::test_stream_metrics_include_sidecar_timings_when_available \
  -q
```

Expected: FAIL because the summary and metrics do not record sidecar timings yet.

- [ ] **Step 6: Implement summary and benchmark extraction**

Modify `scripts/stream_client.py` summary initialization:

```python
"face_provider_timings_ms": None,
```

Add in `accept_json`:

```python
elif event_type == "server.face.frames":
    self._accept_face_frames(payload)
```

Add method:

```python
def _accept_face_frames(self, payload: dict[str, object]) -> None:
    if self.summary["face_provider_timings_ms"] is not None:
        return
    timings = payload.get("provider_timings_ms")
    if not isinstance(timings, dict):
        return
    cleaned: dict[str, float] = {}
    for key, value in timings.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            cleaned[key] = float(value)
    if cleaned:
        self.summary["face_provider_timings_ms"] = cleaned
```

Modify `scripts/benchmark.py` in `stream_metrics_from_summary`:

```python
timings = summary.get("face_provider_timings_ms")
if isinstance(timings, dict):
    predict = _float_or_none(timings.get("predict"))
    total = _float_or_none(timings.get("total"))
    if predict is not None:
        metrics["sidecar_predict_ms"] = predict
    if total is not None:
        metrics["sidecar_total_ms"] = total
```

- [ ] **Step 7: Verify timing tests GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_stream_orchestrator.py::test_stream_face_frames_include_provider_timings_when_available \
  tests/unit/test_stream_client.py::test_client_summary_records_first_face_provider_timings \
  tests/unit/test_benchmark.py::test_stream_metrics_include_sidecar_timings_when_available \
  -q
```

Expected: all timing tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/bionic_head/orchestrators/stream.py scripts/stream_client.py scripts/benchmark.py tests/unit/test_stream_orchestrator.py tests/unit/test_stream_client.py tests/unit/test_benchmark.py
git commit -m "feat: expose sidecar timings in stream benchmark"
```

---

### Task 6: Operations Documentation and Smoke Commands

**Files:**
- Modify: `docs/operations/real-providers.md`
- Modify: `tests/unit/test_benchmark.py` if help command coverage needs updates

**Interfaces:**
- Consumes:
  - `scripts/emotalk_sidecar.py`
  - `config/emotalk-sidecar.example.json`
  - existing stream client and benchmark scripts.
- Produces:
  - User-facing commands to start sidecar, start main service, run diagnostics, stream client, grey-head render, and benchmark.

- [ ] **Step 1: Add documentation section**

Add to `docs/operations/real-providers.md` after the “Local EmoTalk Audio2Face option” section:

```markdown
## EmoTalk sidecar Audio2Face option

Use `audio2face.provider = "emotalk-sidecar"` when the local machine has the
EmoTalk Conda env and you want to avoid reloading EmoTalk for every TTS chunk.
The old `audio2face.provider = "emotalk"` command provider remains the fallback.

Start the sidecar:

```bash
PYTHONPATH=src /home/user/miniconda3/bin/conda run -n emotalk python scripts/emotalk_sidecar.py \
  --host 127.0.0.1 \
  --port 8013 \
  --emotalk-root /home/user/code/EmoTalk_release \
  --model-path /home/user/code/EmoTalk_release/pretrain_model/EmoTalk.pth \
  --device cpu \
  --torch-threads 4 \
  --torch-interop-threads 1 \
  --warmup-seconds 1.0
```

Check health:

```bash
curl -s http://127.0.0.1:8013/health | python -m json.tool
```

Start the main service:

```bash
PYTHONPATH=src BIONIC_CONFIG=config/emotalk-sidecar.example.json \
  .venv/bin/uvicorn bionic_head.api.app:create_app \
  --factory --host 127.0.0.1 --port 8005
```

Run the stream client:

```bash
rm -rf /tmp/bionic-sidecar-stream
PYTHONPATH=src .venv/bin/python scripts/stream_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-long-question-16k.wav \
  --output-dir /tmp/bionic-sidecar-stream \
  --chunk-ms 40
cat /tmp/bionic-sidecar-stream/summary.json
```

Run benchmark:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark.py \
  --mode stream \
  --ws-url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-long-question-16k.wav \
  --runs 3 \
  --output /tmp/bionic-latency-emotalk-sidecar-stream.json
```

For acceptance evidence use `--runs 10`.
```

- [ ] **Step 2: Verify docs contain exact commands**

Run:

```bash
rg -n "emotalk-sidecar|scripts/emotalk_sidecar.py|8013|bionic-latency-emotalk-sidecar" docs/operations/real-providers.md
```

Expected: all four patterns appear.

- [ ] **Step 3: Verify test suite docs-related smoke coverage still passes**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_benchmark.py::test_benchmark_script_help_runs_when_executed_by_path -q
```

Expected: benchmark script help test passes.

- [ ] **Step 4: Commit**

```bash
git add docs/operations/real-providers.md
git commit -m "docs: document emotalk sidecar runbook"
```

---

### Task 7: Full Verification and Real Benchmark

**Files:**
- No required source edits.
- Optional: add a benchmark note to `docs/operations/real-providers.md` only if real timings reveal a necessary operational caveat.

**Interfaces:**
- Consumes all prior tasks.
- Produces final verification evidence and benchmark comparison.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_sidecar_protocol.py \
  tests/unit/test_emotalk_sidecar_adapter.py \
  tests/unit/test_emotalk_sidecar_script.py \
  tests/unit/test_config.py::test_load_emotalk_sidecar_example_settings \
  tests/unit/test_real_registry.py::test_registry_constructs_emotalk_sidecar_example_without_network \
  tests/unit/test_stream_client.py \
  tests/unit/test_benchmark.py \
  -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run full automated test suite**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: full suite passes with the same known integration skips as before.

- [ ] **Step 3: Verify sidecar help inside Conda env**

Run:

```bash
PYTHONPATH=src /home/user/miniconda3/bin/conda run -n emotalk python scripts/emotalk_sidecar.py --help
```

Expected: exits 0 and prints `Run a persistent EmoTalk sidecar server`.

- [ ] **Step 4: Start real sidecar**

Run in terminal A:

```bash
cd /home/user/code/端到端
PYTHONPATH=src /home/user/miniconda3/bin/conda run -n emotalk python scripts/emotalk_sidecar.py \
  --host 127.0.0.1 \
  --port 8013 \
  --emotalk-root /home/user/code/EmoTalk_release \
  --model-path /home/user/code/EmoTalk_release/pretrain_model/EmoTalk.pth \
  --device cpu \
  --torch-threads 4 \
  --torch-interop-threads 1 \
  --warmup-seconds 1.0
```

Expected: after model load, prints `EmoTalk sidecar listening on http://127.0.0.1:8013`.

- [ ] **Step 5: Check sidecar health**

Run in terminal B:

```bash
curl -s http://127.0.0.1:8013/health | python -m json.tool
```

Expected:

```json
{
  "ok": true,
  "loaded": true,
  "provider": "emotalk-sidecar"
}
```

The actual response also includes device, model path, and torch thread fields.

- [ ] **Step 6: Start main service with sidecar config**

Run in terminal B after health passes:

```bash
cd /home/user/code/端到端
source .venv/bin/activate
PYTHONPATH=src BIONIC_CONFIG=config/emotalk-sidecar.example.json \
  .venv/bin/uvicorn bionic_head.api.app:create_app \
  --factory --host 127.0.0.1 --port 8005
```

Expected: Uvicorn running on `http://127.0.0.1:8005`.

- [ ] **Step 7: Confirm diagnostics**

Run in terminal C:

```bash
curl -s http://127.0.0.1:8005/diagnostics | python -m json.tool
```

Expected: `audio2face.provider` is `emotalk-sidecar` and `available` is true.

- [ ] **Step 8: Run one real stream pass**

Run:

```bash
rm -rf /tmp/bionic-sidecar-stream
PYTHONPATH=src .venv/bin/python scripts/stream_client.py \
  --url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-long-question-16k.wav \
  --output-dir /tmp/bionic-sidecar-stream \
  --chunk-ms 40
cat /tmp/bionic-sidecar-stream/summary.json
```

Expected:

```json
{
  "terminal_event": "server.pipeline.done",
  "tts_chunks": 1,
  "ue5_chunks": 1,
  "face_provider_timings_ms": {
    "predict": 0.0,
    "total": 0.0
  }
}
```

Timing values are real non-negative floats, not zeros.

- [ ] **Step 9: Run sidecar benchmark**

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark.py \
  --mode stream \
  --ws-url ws://127.0.0.1:8005/pipeline/stream \
  --wav /tmp/bionic-long-question-16k.wav \
  --runs 3 \
  --output /tmp/bionic-latency-emotalk-sidecar-stream.json
cat /tmp/bionic-latency-emotalk-sidecar-stream.json
```

Expected: report includes successful runs and metrics containing `sidecar_predict_ms` and `sidecar_total_ms`.

- [ ] **Step 10: Compare against Task 7 baseline**

Run:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from pathlib import Path

paths = {
    "task7": Path("/tmp/bionic-latency-task7-stream.json"),
    "sidecar": Path("/tmp/bionic-latency-emotalk-sidecar-stream.json"),
}
for name, path in paths.items():
    report = json.loads(path.read_text(encoding="utf-8"))
    print(name, path)
    for metric in [
        "face_first_chunk_ms",
        "e2e_first_visible_face_ms",
        "total_turn_duration_ms",
        "sidecar_predict_ms",
        "sidecar_total_ms",
    ]:
        value = report.get("metrics", {}).get(metric, {})
        if value:
            print(" ", metric, value)
PY
```

Expected: sidecar report has lower Face latency than Task 7 if repeated model loading was dominant. If Face latency remains near 8–9 seconds, `sidecar_predict_ms` and `sidecar_total_ms` show whether the cost is true model inference.

- [ ] **Step 11: Render grey-head preview for a sidecar run**

Find latest sidecar artifacts:

```bash
find data/runs -path '*face/emotalk_sidecar_*/emotalk.npy' -printf '%T@ %p\n' | sort -n | tail -1
find data/runs -path '*tts/*.wav' -printf '%T@ %p\n' | sort -n | tail -1
```

Render:

```bash
PYTHONPATH=src .venv/bin/python scripts/render_emotalk_grey_head.py \
  --face-npy /path/from/find/emotalk.npy \
  --audio-wav /path/from/find/piper.wav \
  --output /tmp/bionic-sidecar-grey-head.mp4 \
  --name bionic-sidecar \
  --work-dir /tmp/bionic-sidecar-grey-render
```

Expected: `/tmp/bionic-sidecar-grey-head.mp4` exists and plays if local video playback is available.

- [ ] **Step 12: Commit any benchmark note**

If real benchmark reveals a required operational setting such as different `--torch-threads`, update `docs/operations/real-providers.md` with the measured recommendation and commit:

```bash
git add docs/operations/real-providers.md
git commit -m "docs: record emotalk sidecar benchmark note"
```

If no doc update is needed, do not create an empty commit.

---

## Plan Self-Review

- Spec coverage: Tasks 1–4 implement protocol, config, adapter, and sidecar; Task 5 exposes timings; Task 6 documents operations; Task 7 performs full verification and benchmark.
- Public protocols: existing event names and API routes remain unchanged; only optional `provider_timings_ms` is added to `server.face.frames`.
- Fallback: old `emotalk` command provider is not modified.
- Environment: sidecar script is Python 3.8 compatible and uses standard-library HTTP.
- Automated tests: all fake; no real EmoTalk or network required.
- Real acceptance: explicit sidecar health, diagnostics, stream client, benchmark, and grey-head render commands are included.
