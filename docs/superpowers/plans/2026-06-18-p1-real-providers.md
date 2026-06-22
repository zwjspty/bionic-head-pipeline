# P1 Real Providers and Deployment Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-shaped faster-whisper, Ollama, Piper, Morpheus, and raw UE5 providers to the tested P0 core, then validate them with an external WebSocket client, integration smoke tests, and latency reports.

**Architecture:** Keep all orchestration and protocols unchanged. Each real provider is an adapter behind the same P0 contracts; external commands run through one cancellable subprocess utility, optional Python dependencies load lazily, and all real tests use the `integration` marker so the Mock suite remains hermetic.

**Tech Stack:** Python 3.11, existing P0 stack, HTTPX, faster-whisper as an optional dependency, asyncio subprocesses, NumPy, websockets client library, pytest integration markers.

## Global Constraints

- Complete and verify the P0 plan before beginning this plan.
- Do not change public HTTP routes, WebSocket event names, JSON/binary pairing, domain result types, or UE5 channel semantics.
- Real provider imports must be lazy; absent provider packages cannot break Mock startup or normal tests.
- Piper and Morpheus commands are argument arrays, never interpolated shell strings and never run with `shell=True`.
- External process cancellation sends terminate, waits 2 seconds, then sends kill.
- Real-provider tests use `@pytest.mark.integration` and are excluded from default test runs.
- Diagnostics must not run expensive full inference.
- Detailed commands, stderr, stack traces, and sensitive local paths remain server-side; API errors are safe.
- Real provider order is Ollama, faster-whisper, Piper, Morpheus, then end-to-end validation.
- Implement every task with TDD and commit after its complete test cycle.

---

## File Map

```text
pyproject.toml                              Optional real-provider dependencies and client extra
config/real.example.json                    Complete provider configuration schema
src/bionic_head/core/process.py             Cancellable subprocess runner
src/bionic_head/adapters/ollama.py           HTTP streaming LLM provider
src/bionic_head/adapters/faster_whisper.py   Lazy-loaded ASR provider
src/bionic_head/adapters/piper.py            CLI TTS provider
src/bionic_head/adapters/morpheus.py         Conda/CLI Audio2Face provider
src/bionic_head/adapters/morpheus_raw.py     Real raw UE5 formatter provider
src/bionic_head/adapters/registry.py         Real-provider registration
src/bionic_head/evaluation/latency.py        P50/P90 aggregation
scripts/stream_client.py                     Protocol-aware WebSocket client
scripts/benchmark.py                         Repeated real-environment benchmark runner
tests/integration/providers/                 Provider contract smoke tests
tests/integration/test_real_pipeline.py      Real HTTP/WS smoke tests
docs/operations/real-providers.md            Server configuration and validation guide
```

### Task 1: Optional Dependencies, Real Configuration, and Cancellable Process Runner

**Files:**
- Modify: `pyproject.toml`
- Modify: `config/real.example.json`
- Modify: `src/bionic_head/config.py`
- Create: `src/bionic_head/core/process.py`
- Create: `tests/unit/test_process.py`
- Modify: `tests/unit/test_config.py`

**Interfaces:**
- Produces:
  - optional dependency groups `asr` and `client`;
  - `CommandSettings`;
  - `run_command(args, cwd, stdin, timeout, cancellation, grace_seconds) -> CompletedCommand`.
- Consumes: P0 cancellation token and error model.

- [x] **Step 1: Write failing process-runner tests**

```python
# tests/unit/test_process.py
import asyncio
import sys
import pytest

from bionic_head.core.cancellation import CancellationToken
from bionic_head.core.process import run_command
from bionic_head.domain.errors import ErrorCode, PipelineException


@pytest.mark.asyncio
async def test_command_captures_stdout_and_stderr(tmp_path) -> None:
    result = await run_command(
        args=[sys.executable, "-c", "import sys; print('ok'); print('note', file=sys.stderr)"],
        cwd=tmp_path,
        stdin=None,
        timeout_seconds=2,
        cancellation=CancellationToken(),
        grace_seconds=0.1,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == b"ok"
    assert result.stderr.strip() == b"note"


@pytest.mark.asyncio
async def test_timeout_maps_to_provider_timeout(tmp_path) -> None:
    with pytest.raises(PipelineException) as raised:
        await run_command(
            args=[sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            stdin=None,
            timeout_seconds=0.05,
            cancellation=CancellationToken(),
            grace_seconds=0.05,
        )
    assert raised.value.code is ErrorCode.PROVIDER_TIMEOUT
```

- [x] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_process.py tests/unit/test_config.py -v`

Expected: FAIL because the process runner and real command settings do not exist.

- [x] **Step 3: Add optional dependencies and exact config fields**

Add:

```toml
[project.optional-dependencies]
dev = [
  "httpx>=0.28,<1",
  "pytest>=8.3,<9",
  "pytest-asyncio>=0.25,<1",
  "pytest-timeout>=2.3,<3",
]
asr = ["faster-whisper>=1.1,<2"]
llm = ["httpx>=0.28,<1"]
client = ["websockets>=15,<16"]
all = [
  "faster-whisper>=1.1,<2",
  "httpx>=0.28,<1",
  "websockets>=15,<16",
]
```

Extend settings with:

```python
class CommandSettings(BaseModel):
    executable: str = ""
    args: list[str] = Field(default_factory=list)
    cwd: Path | None = None
    timeout_seconds: float = 120.0


class FasterWhisperSettings(BaseModel):
    model: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "zh"


class OllamaSettings(BaseModel):
    base_url: AnyHttpUrl = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b"
    timeout_seconds: float = 120.0


class PiperSettings(CommandSettings):
    model_path: Path | None = None


class MorpheusSettings(CommandSettings):
    output_npy_glob: str = "*.npy"
    output_json_glob: str = "*.json"
```

`config/real.example.json` must include:

```json
{
  "providers": {
    "faster_whisper": {"model": "base", "device": "cpu", "compute_type": "int8", "language": "zh"},
    "ollama": {"base_url": "http://127.0.0.1:11434", "model": "qwen2.5:3b", "timeout_seconds": 120},
    "piper": {
      "executable": "",
      "args": ["--model", "{model_path}", "--output_file", "{output_path}"],
      "model_path": "",
      "cwd": null,
      "timeout_seconds": 120
    },
    "morpheus": {
      "executable": "conda",
      "args": ["run", "-n", "lyyMor", "", "--input", "{input_path}", "--output-dir", "{output_dir}"],
      "cwd": "/home/hailab/liuyiyu/head-project/Morpheus-Software",
      "output_npy_glob": "*.npy",
      "output_json_glob": "*.json",
      "timeout_seconds": 300
    }
  }
}
```

The empty Morpheus command element is deliberately invalid until the deployment command is confirmed; diagnostics must report it as unavailable.

- [x] **Step 4: Implement process lifecycle**

`run_command` uses `asyncio.create_subprocess_exec(*args, cwd=cwd, stdin=PIPE if stdin else DEVNULL, stdout=PIPE, stderr=PIPE, start_new_session=True)`. Race `process.communicate(stdin)` against `cancellation.wait()` and timeout.

On cancel:

```python
process.terminate()
try:
    await asyncio.wait_for(process.wait(), grace_seconds)
except TimeoutError:
    process.kill()
    await process.wait()
raise asyncio.CancelledError
```

On nonzero exit, raise `provider_failed` with a safe message. Keep decoded stderr available only on `CompletedCommand.debug_stderr` for logging; never put it in `PipelineException.safe_message`.

- [x] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_process.py tests/unit/test_config.py -v`

Expected: PASS for success, nonzero exit, timeout, cooperative cancel, kill-after-grace, missing executable, and argument-array validation.

- [x] **Step 6: Commit**

```bash
git add pyproject.toml config/real.example.json src/bionic_head/config.py src/bionic_head/core/process.py tests/unit
git commit -m "feat: add real provider process infrastructure"
```

### Task 2: Ollama LLM Provider

**Files:**
- Create: `src/bionic_head/adapters/ollama.py`
- Modify: `src/bionic_head/adapters/registry.py`
- Create: `tests/unit/test_ollama_adapter.py`
- Create: `tests/integration/providers/test_ollama.py`

**Interfaces:**
- Produces: `OllamaLLMAdapter` implementing P0 `LLMAdapter`.
- Consumes: `/api/chat` streaming NDJSON and target model `qwen2.5:3b`.

- [x] **Step 1: Write failing unit tests with HTTPX MockTransport**

```python
# tests/unit/test_ollama_adapter.py
import json
import httpx
import pytest

from bionic_head.adapters.ollama import OllamaLLMAdapter


@pytest.mark.asyncio
async def test_streams_tokens_and_parses_final_emotion(turn_context) -> None:
    lines = [
        {"message": {"content": '{"reply":"你好'}}, "done": False},
        {"message": {"content": '！","emotion":"friendly","intensity":0.8}'}, "done": False},
        {"done": True},
    ]
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=b"\n".join(json.dumps(line, ensure_ascii=False).encode() for line in lines),
        )
    )
    adapter = OllamaLLMAdapter(
        settings=OllamaSettings(
            base_url="http://ollama.test:11434",
            model="qwen2.5:3b",
            timeout_seconds=2,
        ),
        transport=transport,
    )
    events = [event async for event in adapter.chat_stream("你好", [], turn_context)]
    assert "".join(event.text for event in events if event.kind == "token")
    assert events[-1].result.emotion.value == "friendly"
```

- [x] **Step 2: Run the unit test and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_ollama_adapter.py -v`

Expected: FAIL because `OllamaLLMAdapter` does not exist.

- [x] **Step 3: Implement structured prompting and robust parsing**

Send `POST /api/chat`:

```json
{
  "model": "qwen2.5:3b",
  "stream": true,
  "format": "json",
  "messages": [
    {
      "role": "system",
      "content": "Return one JSON object with reply, emotion, intensity. emotion must be one of neutral,friendly,happy,sad,angry,surprised,thinking,calm. intensity must be 0.0 to 1.0."
    },
    {"role": "user", "content": "<recognized text>"}
  ]
}
```

Accumulate content for final JSON parsing. To preserve early speech, emit token events from the `reply` JSON string only after the parser has seen the opening `"reply":"` sequence, unescaping JSON string content incrementally. If incremental parsing becomes invalid, continue accumulating and emit no unsafe partial reply; at stream end parse the full object. Reject missing fields, unknown emotion, or invalid intensity with `output_validation_failed`.

`chat()` consumes `chat_stream()` and returns the final `LLMResult`. `cancel()` closes active HTTP response streams associated with the turn ID.

Diagnostics:

1. `GET /api/tags`;
2. available false if unreachable;
3. available false if `qwen2.5:3b` is absent;
4. never run generation.

- [x] **Step 4: Add integration smoke test**

```python
# tests/integration/providers/test_ollama.py
import os
from pathlib import Path
import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_ollama_streams_valid_result(turn_context) -> None:
    config_path = os.environ.get("BIONIC_CONFIG")
    if not config_path:
        pytest.skip("BIONIC_CONFIG is required")
    settings = load_settings(Path(config_path))
    adapter = OllamaLLMAdapter(settings.providers.ollama)
    events = [event async for event in adapter.chat_stream("请简单打个招呼", [], turn_context)]
    result = events[-1].result
    assert result.reply
    assert 0.0 <= result.intensity <= 1.0
```

- [x] **Step 5: Run Mock and optional real tests**

Run: `.venv/bin/python -m pytest tests/unit/test_ollama_adapter.py -v`

Expected: PASS for streaming, malformed NDJSON, HTTP failure, timeout, cancellation, invalid JSON output, and diagnostics.

When Ollama is available, run:

`BIONIC_CONFIG=config/local.json .venv/bin/python -m pytest tests/integration/providers/test_ollama.py -m integration -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/bionic_head/adapters/ollama.py src/bionic_head/adapters/registry.py tests
git commit -m "feat: add ollama streaming provider"
```

### Task 3: faster-whisper ASR Provider

**Files:**
- Create: `src/bionic_head/adapters/faster_whisper.py`
- Modify: `src/bionic_head/adapters/registry.py`
- Create: `tests/unit/test_faster_whisper_adapter.py`
- Create: `tests/integration/providers/test_faster_whisper.py`

**Interfaces:**
- Produces: `FasterWhisperASRAdapter` implementing P0 `ASRAdapter`.
- Consumes: valid WAV and lazy `WhisperModel`.

- [x] **Step 1: Write failing tests with a fake model**

```python
# tests/unit/test_faster_whisper_adapter.py
from types import SimpleNamespace
import pytest

from bionic_head.adapters.faster_whisper import FasterWhisperASRAdapter


class FakeModel:
    def transcribe(self, path, language, vad_filter):
        return [SimpleNamespace(text=" 你好 "), SimpleNamespace(text=" 世界 ")], SimpleNamespace(language="zh")


@pytest.mark.asyncio
async def test_transcribe_normalizes_segments(turn_context, speech_wav) -> None:
    adapter = FasterWhisperASRAdapter(
        settings=FasterWhisperSettings(
            model="base",
            device="cpu",
            compute_type="int8",
            language="zh",
        ),
        model_factory=lambda **_: FakeModel(),
    )
    result = await adapter.transcribe(speech_wav, turn_context)
    assert result.text == "你好 世界"
    assert result.language == "zh"
```

- [x] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_faster_whisper_adapter.py -v`

Expected: FAIL because the adapter does not exist.

- [x] **Step 3: Implement lazy model loading and thread offload**

Import `faster_whisper` only inside the default model factory. Cache one model per adapter instance under an async lock. Run model construction and `.transcribe()` through `asyncio.to_thread`.

Use:

```python
segments, info = model.transcribe(
    str(audio_path),
    language=self.settings.language,
    vad_filter=True,
)
text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
```

Empty text raises `no_speech_detected`. Cancellation after the worker starts cannot stop CTranslate2 reliably; mark the turn cancelled and discard the late result through the existing turn guard.

Diagnostics checks:

- `importlib.util.find_spec("faster_whisper")`;
- configured model/device/compute type;
- no model download or inference.

- [x] **Step 4: Add real smoke test**

```python
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
```

- [x] **Step 5: Run unit and optional real tests**

Run: `.venv/bin/python -m pytest tests/unit/test_faster_whisper_adapter.py -v`

Expected: PASS for lazy loading, normalization, empty speech, worker failure, cancellation-discard, and diagnostics.

When model access is available:

`BIONIC_CONFIG=config/local.json .venv/bin/python -m pytest tests/integration/providers/test_faster_whisper.py -m integration -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/bionic_head/adapters/faster_whisper.py src/bionic_head/adapters/registry.py tests
git commit -m "feat: add faster whisper provider"
```

### Task 4: Piper TTS Provider

**Files:**
- Create: `src/bionic_head/adapters/piper.py`
- Modify: `src/bionic_head/adapters/registry.py`
- Create: `tests/unit/test_piper_adapter.py`
- Create: `tests/integration/providers/test_piper.py`

**Interfaces:**
- Produces: `PiperTTSAdapter` implementing P0 `TTSAdapter`.
- Consumes: configurable executable, model path, argument template, and `run_command`.

- [x] **Step 1: Write failing argument and output tests**

```python
# tests/unit/test_piper_adapter.py
import sys
import pytest

from bionic_head.adapters.piper import PiperTTSAdapter


@pytest.mark.asyncio
async def test_piper_writes_and_validates_wav(fake_piper_script, turn_context) -> None:
    adapter = PiperTTSAdapter(
        executable=sys.executable,
        args=[str(fake_piper_script), "{output_path}"],
        model_path="/models/voice.onnx",
        timeout_seconds=2,
        grace_seconds=0.1,
    )
    result = await adapter.synthesize("你好", Emotion.FRIENDLY, 0.8, turn_context)
    assert result.path.exists()
    assert result.sample_rate == 16000
```

Define `fake_piper_script` in the test module. It is a temporary Python program that accepts the output path as its final argument and writes a valid mono, PCM16, 16 kHz WAV; separate test variants exit nonzero, sleep past timeout, and write invalid bytes.

- [x] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_piper_adapter.py -v`

Expected: FAIL because the adapter does not exist.

- [x] **Step 3: Implement safe command rendering**

Allow only these template variables:

```text
{model_path}
{output_path}
{text}
```

Reject unknown template variables at configuration validation. Prefer text on stdin when `{text}` is absent; when present, insert it as one argument, never through a shell.

Create output under `turn_context.artifact_dir / "tts" / f"{chunk_id}.wav"`. After success, call `inspect_wav`; reject missing, empty, malformed, stereo, or unexpected sample-width output with `output_validation_failed`. Preserve Piper's actual sample rate in `AudioArtifact`; do not silently resample.

Diagnostics verifies executable with `shutil.which` or an executable absolute path, model file existence, writable output root, and valid template variables. It does not synthesize.

- [x] **Step 4: Add real smoke test**

```python
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
```

- [x] **Step 5: Run unit and optional real tests**

Run: `.venv/bin/python -m pytest tests/unit/test_piper_adapter.py -v`

Expected: PASS for stdin mode, text-argument mode, timeout, cancel/kill, nonzero exit, invalid WAV, missing model, and diagnostics.

After deployment paths are configured:

`BIONIC_CONFIG=config/local.json .venv/bin/python -m pytest tests/integration/providers/test_piper.py -m integration -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/bionic_head/adapters/piper.py src/bionic_head/adapters/registry.py tests
git commit -m "feat: add piper tts provider"
```

### Task 5: Morpheus Audio2Face and Raw UE5 Providers

**Files:**
- Create: `src/bionic_head/adapters/morpheus.py`
- Create: `src/bionic_head/adapters/morpheus_raw.py`
- Modify: `src/bionic_head/adapters/registry.py`
- Create: `tests/unit/test_morpheus_adapter.py`
- Create: `tests/integration/providers/test_morpheus.py`

**Interfaces:**
- Produces:
  - `MorpheusAudio2FaceAdapter`;
  - `MorpheusRawUE5Adapter`;
  - validated `FaceArtifact` and `UE5Payload`.
- Consumes: command template, output globs, NumPy `.npy`, optional JSON artifacts.

- [x] **Step 1: Write failing output-discovery tests**

```python
# tests/unit/test_morpheus_adapter.py
import sys
import pytest


@pytest.mark.asyncio
async def test_morpheus_loads_n_by_52_output(fake_morpheus_script, mock_audio, turn_context) -> None:
    adapter = MorpheusAudio2FaceAdapter(
        executable=sys.executable,
        args=[str(fake_morpheus_script), "{input_path}", "{output_dir}"],
        output_npy_glob="*.npy",
        output_json_glob="*.json",
        timeout_seconds=2,
        grace_seconds=0.1,
    )
    face = await adapter.drive(mock_audio, Emotion.FRIENDLY, 0.8, turn_context)
    assert face.frame_count > 0
    assert face.channel_count == 52
```

Define `fake_morpheus_script` in the test module. It accepts input and output directory arguments and writes one finite NumPy array shaped `[6, 52]`; separate variants write no file, multiple files, `[6, 51]`, NaN values, sleep past timeout, and ignore terminate long enough to exercise kill. Define `mock_audio` from a valid WAV using `audio_artifact_from_wav`.

- [x] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_morpheus_adapter.py -v`

Expected: FAIL because Morpheus adapters do not exist.

- [x] **Step 3: Implement command and output validation**

Allowed template variables:

```text
{input_path}
{output_dir}
{emotion}
{intensity}
```

Create a unique output directory per segment. After process success:

1. Find exactly one `.npy` matching `output_npy_glob`; zero or multiple matches are errors.
2. Load with `numpy.load(npy_path, allow_pickle=False)`.
3. Require `ndim == 2`, `shape[1] == 52`, at least one frame, finite values only.
4. Use fps 30 unless an emitted JSON file contains a valid explicit fps.
5. Copy or reference auxiliary JSON artifacts in `FaceArtifact`.
6. Validate audio duration and frame duration differ by no more than `max(0.1 seconds, one frame)`; otherwise add a quality warning but do not fail P1.

Use an `asyncio.Semaphore(1)` shared by all Morpheus adapter calls. Cancellation follows the shared process runner.

Diagnostics checks Conda executable, `lyyMor` environment presence through `conda env list --json`, project directory, non-empty command argument, output glob syntax, and writable artifact root. It does not run inference.

`MorpheusRawUE5Adapter` delegates to the P0 `build_ue5_payload` and retains `protocol=bionic-head-ue5-v1`, `format=morpheus_52_raw`.

- [x] **Step 4: Add real smoke test**

```python
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
```

- [x] **Step 5: Run unit and optional real tests**

Run: `.venv/bin/python -m pytest tests/unit/test_morpheus_adapter.py -v`

Expected: PASS for command rendering, semaphore serialization, missing/multiple output, wrong shape, NaN output, timeout, cancel/kill, diagnostics, and raw UE5 formatting.

After the real command is confirmed:

`BIONIC_CONFIG=config/local.json .venv/bin/python -m pytest tests/integration/providers/test_morpheus.py -m integration -v`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/bionic_head/adapters/morpheus.py src/bionic_head/adapters/morpheus_raw.py src/bionic_head/adapters/registry.py tests
git commit -m "feat: add morpheus and raw ue5 providers"
```

### Task 6: Mixed-Provider Registry and Real End-to-End Smoke Tests

**Files:**
- Modify: `src/bionic_head/adapters/registry.py`
- Modify: `src/bionic_head/api/dependencies.py`
- Create: `tests/unit/test_real_registry.py`
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_real_pipeline.py`
- Create: `tests/fixtures/README.md`

**Interfaces:**
- Produces: per-adapter provider mixing and full real HTTP/WS smoke coverage.
- Consumes: all real providers from Tasks 2–5.

- [x] **Step 1: Write failing mixed-registry test**

```python
# tests/unit/test_real_registry.py
def test_registry_allows_mock_asr_and_real_ollama() -> None:
    settings = load_settings(Path("config/mock.json")).model_copy(deep=True)
    settings.adapters.llm.provider = "ollama"
    settings.adapters.ue5.provider = "morpheus-raw"
    registry = build_registry(settings)
    assert registry.asr.name == "mock"
    assert registry.llm.name == "ollama"
    assert registry.ue5.name == "morpheus-raw"
```

- [x] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_real_registry.py -v`

Expected: FAIL until registry dispatch covers all real names.

- [x] **Step 3: Complete registry dispatch and startup behavior**

Provider map:

```python
ASR_PROVIDERS = {"mock": build_mock_asr, "faster-whisper": build_faster_whisper}
LLM_PROVIDERS = {"mock": build_mock_llm, "ollama": build_ollama}
TTS_PROVIDERS = {"mock": build_mock_tts, "piper": build_piper}
A2F_PROVIDERS = {"mock": build_mock_a2f, "morpheus": build_morpheus}
UE5_PROVIDERS = {"mock": build_mock_ue5, "morpheus-raw": build_morpheus_raw}
```

Construction must not run diagnostics or expensive model loading. Unknown provider names fail startup with a clear configuration error. Known but unconfigured real providers construct successfully and report unavailable through diagnostics; invoking them raises `provider_unavailable`.

- [x] **Step 4: Add real end-to-end tests**

Create integration fixtures:

```python
# tests/integration/conftest.py
@pytest.fixture
def real_settings():
    config_path = os.environ.get("BIONIC_CONFIG")
    if not config_path:
        pytest.skip("BIONIC_CONFIG is required")
    return load_settings(Path(config_path))


@pytest.fixture
def chinese_wav():
    wav_path = os.environ.get("BIONIC_TEST_WAV")
    if not wav_path:
        pytest.skip("BIONIC_TEST_WAV is required")
    return Path(wav_path)


@pytest.fixture
def chinese_pcm(chinese_wav):
    return read_wav_pcm16(chinese_wav)


@pytest.fixture
def real_app(real_settings, tmp_path):
    settings = real_settings.model_copy(deep=True)
    settings.storage.root = tmp_path / "real-test-data"
    return create_app(settings)
```

`tests/integration/test_real_pipeline.py` contains:

```python
@pytest.mark.integration
def test_real_offline_pipeline(real_app, chinese_wav):
    response = post_audio(real_app, chinese_wav)
    assert response.status_code == 200
    body = response.json()
    assert body["asr"]["text"]
    assert body["llm"]["reply"]
    assert body["audio"]["duration_seconds"] > 0
    assert body["face"]["channel_count"] == 52


@pytest.mark.integration
def test_real_pseudo_streaming_pipeline(real_app, chinese_pcm):
    events, binaries = run_ws_turn(real_app, chinese_pcm)
    assert terminal_types(events) == ["server.pipeline.done"]
    assert any(event["type"] == "server.segment.ready" for event in events)
    assert binaries
```

Define `post_audio` and `run_ws_turn` in the same test module. They must use only the public API, send full `bionic-head-stream-v1` envelopes, consume each WAV binary immediately after `server.tts.audio`, and stop at exactly one terminal event.

Test setup:

- `BIONIC_CONFIG` must point to `config/local.json`;
- `BIONIC_TEST_WAV` must point to the Chinese WAV;
- `BIONIC_TEST_TTS_WAV` must point to a valid Chinese response WAV for the isolated Morpheus test;
- each test skips with an explicit reason naming its missing environment variable;
- fixture README documents the required Morpheus `[N,52]` and UE5 JSON reference samples.

- [x] **Step 5: Run all hermetic tests**

Run: `.venv/bin/python -m pytest -m 'not integration' -v`

Expected: PASS without faster-whisper installed and without external services.

- [x] **Step 6: Run staged real validation**

Run each in this order:

```bash
BIONIC_CONFIG=config/local.json .venv/bin/python -m pytest tests/integration/providers/test_ollama.py -m integration -v
BIONIC_CONFIG=config/local.json BIONIC_TEST_WAV=/path/to/chinese.wav .venv/bin/python -m pytest tests/integration/providers/test_faster_whisper.py -m integration -v
BIONIC_CONFIG=config/local.json .venv/bin/python -m pytest tests/integration/providers/test_piper.py -m integration -v
BIONIC_CONFIG=config/local.json .venv/bin/python -m pytest tests/integration/providers/test_morpheus.py -m integration -v
BIONIC_CONFIG=config/local.json BIONIC_TEST_WAV=/path/to/chinese.wav .venv/bin/python -m pytest tests/integration/test_real_pipeline.py -m integration -v
```

Expected: each layer passes before running the next. If Piper or Morpheus command data remains unavailable, its test must SKIP with the exact missing configuration field named, not fail the Mock suite.

- [x] **Step 7: Commit**

```bash
git add src/bionic_head/adapters/registry.py src/bionic_head/api/dependencies.py tests
git commit -m "test: validate mixed and real provider pipelines"
```

### Task 7: Protocol-Aware WebSocket Test Client

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/stream_client.py`
- Create: `scripts/__init__.py`
- Create: `tests/unit/test_stream_client.py`
- Create: `docs/operations/stream-client.md`

**Interfaces:**
- Produces: CLI client that sends PCM/WAV input, receives paired WAV binaries and UE5 frame chunks, saves outputs, validates sequence, and supports cancel.

- [x] **Step 1: Write failing client-state tests**

```python
# tests/unit/test_stream_client.py
from scripts.stream_client import ClientReceiver


def test_receiver_pairs_tts_metadata_with_next_binary(tmp_path) -> None:
    receiver = ClientReceiver(tmp_path)
    receiver.accept_json(server_event(
        event_type="server.tts.audio",
        sequence=1,
        payload={"chunk_id": "0", "byte_length": 4, "format": "wav"},
    ))
    receiver.accept_binary(b"RIFF")
    assert (tmp_path / "tts/0.wav").read_bytes() == b"RIFF"


def test_cancel_clears_pending_playback(tmp_path) -> None:
    receiver = ClientReceiver(tmp_path)
    receiver.pending_segments["0"] = object()
    receiver.accept_json(server_event(
        event_type="server.turn.cancelled",
        sequence=1,
        payload={},
    ))
    assert receiver.pending_segments == {}
```

- [x] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_stream_client.py -v`

Expected: FAIL because the client does not exist.

- [x] **Step 3: Implement CLI and protocol receiver**

Define `server_event` in the test module to populate the full protocol envelope with fixed session/turn UUIDs, a fresh event UUID, and a timezone-aware UTC timestamp.

CLI:

```text
.venv/bin/python scripts/stream_client.py \
  --url ws://127.0.0.1:8000/pipeline/stream \
  --wav /path/to/input.wav \
  --output-dir client-output \
  --chunk-ms 40
```

Validate source WAV is mono, PCM16, 16 kHz. Split it into exact chunk duration, send metadata then binary, then `client.audio.end`.

The sender generates one session UUID, one turn UUID, fresh event UUIDs, UTC timestamps, and a client-direction sequence beginning at one. Every JSON message includes the full P0 envelope; only the PCM frame is binary.

Receiver requirements:

- sequence must increase by one;
- protocol, session ID, and turn ID must match the active connection/turn;
- after `server.tts.audio`, next server frame must be binary of declared length;
- save `tts/{chunk_id}.wav`;
- append `server.ue5.frames` by frame offset and reject overlap/gaps;
- write `events.jsonl`, `ue5/{chunk_id}.json`, and `summary.json`;
- print state, ASR final, LLM chunks, terminal event, and measured latency;
- Ctrl-C sends `client.turn.cancel` once before closing;
- `server.turn.cancelled` clears pending audio/frame queues.

- [x] **Step 4: Run unit tests and a Mock live smoke test**

Run: `.venv/bin/python -m pytest tests/unit/test_stream_client.py -v`

Expected: PASS.

With Mock server running:

```bash
.venv/bin/uvicorn bionic_head.api.app:create_app --factory --host 127.0.0.1 --port 8000
.venv/bin/python scripts/stream_client.py --url ws://127.0.0.1:8000/pipeline/stream --wav tests/fixtures/generated-speech.wav --output-dir /tmp/bionic-client
```

Expected: terminal event is `server.pipeline.done`; WAV and UE5 files are saved.

- [x] **Step 5: Commit**

```bash
git add pyproject.toml scripts/__init__.py scripts/stream_client.py tests/unit/test_stream_client.py docs/operations/stream-client.md
git commit -m "feat: add websocket validation client"
```

### Task 8: Latency Benchmarking, Report, and Deployment Runbook

**Files:**
- Create: `src/bionic_head/evaluation/__init__.py`
- Create: `src/bionic_head/evaluation/latency.py`
- Create: `scripts/benchmark.py`
- Create: `tests/unit/test_latency_report.py`
- Create: `docs/operations/real-providers.md`
- Modify: `README.md`

**Interfaces:**
- Produces: repeated benchmark runner and `latency_report.json` with count, failures, P50, and P90.
- Consumes: `timeline.json` metrics from successful turns.

- [ ] **Step 1: Write failing percentile test**

```python
# tests/unit/test_latency_report.py
from bionic_head.evaluation.latency import summarize


def test_summary_reports_nearest_rank_p50_p90() -> None:
    report = summarize([100, 200, 300, 400, 500])
    assert report == {"count": 5, "p50_ms": 300, "p90_ms": 500, "min_ms": 100, "max_ms": 500}
```

- [ ] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_latency_report.py -v`

Expected: FAIL because evaluation code does not exist.

- [ ] **Step 3: Implement deterministic report generation**

Use nearest-rank percentile:

```python
def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]
```

Report metrics:

```text
audio_end_to_asr_final_ms
audio_end_to_llm_first_token_ms
audio_end_to_first_tts_ready_ms
audio_end_to_first_face_ready_ms
audio_end_to_first_segment_ready_ms
total_turn_duration_ms
```

Include run count, success count, failure count, failure codes, provider names, timestamp, and source WAV path. Never mix failed-run missing metrics into percentile arrays.

- [ ] **Step 4: Implement benchmark CLI**

CLI:

```text
.venv/bin/python scripts/benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --ws-url ws://127.0.0.1:8000/pipeline/stream \
  --wav /path/to/chinese.wav \
  --runs 10 \
  --output latency_report.json
```

Run offline and stream modes separately via `--mode offline|stream`. For stream mode, reuse the tested client sender/receiver. Require at least 10 runs for a report used as acceptance evidence.

- [ ] **Step 5: Write the real-provider runbook**

Document:

1. install `.venv/bin/python -m pip install -e '.[dev,all]'`;
2. copy `config/real.example.json` to ignored `config/local.json`;
3. confirm Ollama model;
4. confirm faster-whisper model access;
5. fill Piper executable/model/arguments;
6. fill the empty Morpheus command element and output globs;
7. run `/diagnostics`;
8. execute staged provider tests;
9. execute full real pipeline tests;
10. run at least 10 benchmark turns;
11. verify `[N,52]`, WAV playback, latest files, and client output.

Explicitly state that unknown Piper/Morpheus command data is a deployment blocker for their real smoke tests, not a blocker for the completed Mock service.

- [ ] **Step 6: Run complete verification**

Run:

```bash
.venv/bin/python -m pytest -m 'not integration' -v
.venv/bin/python -m pytest --collect-only -q
.venv/bin/python scripts/benchmark.py --help
.venv/bin/python scripts/stream_client.py --help
```

Expected: all hermetic tests pass, collection succeeds, and both CLIs show usage without importing faster-whisper.

On the deployment server, run:

```bash
BIONIC_CONFIG=config/local.json BIONIC_TEST_WAV=/path/to/chinese.wav .venv/bin/python -m pytest -m integration -v
.venv/bin/python scripts/benchmark.py --mode stream --ws-url ws://127.0.0.1:8000/pipeline/stream --wav /path/to/chinese.wav --runs 10 --output latency_report.json
```

Expected: real tests pass and report contains P50/P90 for first token, first TTS, first face, first segment, and total duration.

- [ ] **Step 7: Commit**

```bash
git add src/bionic_head/evaluation scripts/benchmark.py tests/unit/test_latency_report.py docs/operations/real-providers.md README.md
git commit -m "feat: add real pipeline benchmarking and runbook"
```

## P1 Completion Gate

Verify locally:

```bash
git status --short
.venv/bin/python -m pytest -m 'not integration' -v
.venv/bin/python -m pytest --collect-only -q
```

Verify on the real deployment server:

```bash
curl -s http://127.0.0.1:8000/diagnostics
BIONIC_CONFIG=config/local.json BIONIC_TEST_WAV=/path/to/chinese.wav .venv/bin/python -m pytest -m integration -v
.venv/bin/python scripts/stream_client.py --url ws://127.0.0.1:8000/pipeline/stream --wav /path/to/chinese.wav --output-dir client-output
.venv/bin/python scripts/benchmark.py --mode stream --ws-url ws://127.0.0.1:8000/pipeline/stream --wav /path/to/chinese.wav --runs 10 --output latency_report.json
```

Expected:

- all diagnostics are available;
- real WAV is recognized;
- Ollama returns reply/emotion/intensity;
- Piper generates playable WAV;
- Morpheus generates finite `[N,52]`;
- client receives ordered WAV and frame chunks;
- latest files belong to the successful current turn;
- cancellation prevents late real-provider output from publishing;
- latency report records P50/P90 without enforcing the 1-second stretch goal.
