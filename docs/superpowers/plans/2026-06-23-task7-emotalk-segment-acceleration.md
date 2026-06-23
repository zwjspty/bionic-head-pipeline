# Task 7 EmoTalk Segment Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/pipeline/stream` produce Face-friendly short segments and prevent slow Audio2Face from blocking later TTS chunks.

**Architecture:** Keep the public WebSocket protocol unchanged. Add configurable minimum sentence length to `SentenceBuffer`, lower default max segment length, split stream segment processing into a fast TTS path plus background Face/UE5 tasks, and wait for all Face tasks before publishing latest/done.

**Tech Stack:** Python 3.11 target / local Python 3.10 compatible tests, FastAPI, asyncio, Pydantic v2, pytest, mock providers.

## Global Constraints

- Do not change event names or binary framing.
- Do not introduce EmoTalk sidecar in this task.
- Preserve `turn_id` / `generation_epoch` stale-drop semantics.
- Use TDD: write each failing test before production code.
- Use `PYTHONPATH=src .venv/bin/python -m pytest ...` for local tests.

---

### Task 1: Add Face-friendly sentence buffering

**Files:**
- Modify: `src/bionic_head/core/sentences.py`
- Modify: `src/bionic_head/config.py`
- Modify: `config/mock.json`
- Modify: `config/real.example.json`
- Modify: `config/emotalk.example.json`
- Test: `tests/unit/test_sentences.py`

**Interfaces:**
- Consumes: `SentenceBuffer.push(token: str) -> list[str]`, `SentenceBuffer.flush() -> str | None`
- Produces: `SentenceBuffer(max_chars: int, min_chars: int = 1)`

- [ ] **Step 1: Write failing sentence-buffer tests**

Create `tests/unit/test_sentences.py`:

```python
from bionic_head.core.sentences import SentenceBuffer


def test_sentence_buffer_waits_for_min_chars_before_punctuation_flush() -> None:
    buffer = SentenceBuffer(max_chars=24, min_chars=8)

    assert buffer.push("你好！") == []
    assert buffer.push("很高兴。") == []
    assert buffer.push("呀。") == ["你好！很高兴。呀。"]


def test_sentence_buffer_forces_flush_at_max_chars_without_punctuation() -> None:
    buffer = SentenceBuffer(max_chars=6, min_chars=4)

    assert buffer.push("一二三") == []
    assert buffer.push("四五六") == ["一二三四五六"]
    assert buffer.flush() is None


def test_sentence_buffer_splits_at_first_eligible_punctuation_inside_token() -> None:
    buffer = SentenceBuffer(max_chars=24, min_chars=8)

    assert buffer.push("你好！很高兴。呀。下一句。") == ["你好！很高兴。呀。"]
    assert buffer.flush() == "下一句。"
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_sentences.py -q
```

Expected: failure because `SentenceBuffer.__init__()` does not accept `min_chars`.

- [ ] **Step 3: Implement minimal buffering change**

Change `SentenceBuffer.__init__` to accept and validate `min_chars`, then only split at punctuation when `punctuation_index + 1 >= min_chars`; keep max-char split unchanged.

- [ ] **Step 4: Add config field**

Add to `StreamSettings`:

```python
sentence_min_chars: int = Field(default=8, ge=1)
sentence_max_chars: int = Field(default=24, ge=1)
```

Update JSON configs with:

```json
"sentence_min_chars": 8,
"sentence_max_chars": 24
```

- [ ] **Step 5: Verify Task 1**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_sentences.py tests/unit/test_config.py -q
```

Expected: all selected tests pass.

---

### Task 2: Decouple TTS chunks from slow Face processing

**Files:**
- Modify: `src/bionic_head/orchestrators/stream.py`
- Test: `tests/unit/test_stream_orchestrator.py`

**Interfaces:**
- Produces: `_process_audio_segment(...) -> tuple[str, AudioArtifact]`
- Produces: `_process_face_segment(chunk_id: str, audio: AudioArtifact, ...) -> None`
- Preserves: `server.tts.audio` binary pair before `server.face.frames`

- [ ] **Step 1: Write failing stream ordering test**

Append to `tests/unit/test_stream_orchestrator.py`:

```python
@pytest.mark.asyncio
async def test_stream_does_not_block_later_tts_on_slow_face(
    mock_settings,
    stream_harness_factory,
) -> None:
    settings = mock_settings.model_copy(deep=True)
    settings.mock.reply = "第一段内容已经足够。第二段内容也足够。"
    settings.mock.latency_ms.face = 100
    settings.stream.sentence_min_chars = 4
    settings.stream.sentence_max_chars = 12
    registry = build_registry(settings)
    harness = stream_harness_factory(settings=settings, registry=registry)

    await harness.run()

    types = harness.json_types
    tts_indexes = [index for index, event_type in enumerate(types) if event_type == "server.tts.audio"]
    first_face_index = types.index("server.face.frames")
    assert len(tts_indexes) >= 2
    assert tts_indexes[1] < first_face_index
```

- [ ] **Step 2: Verify test fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_orchestrator.py::test_stream_does_not_block_later_tts_on_slow_face -q
```

Expected: failure because current `_process_segment()` waits for Face before processing the second TTS segment.

- [ ] **Step 3: Split segment processing**

In `StreamOrchestrator.run`, create `face_tasks: list[asyncio.Task[None]] = []`. Replace each `_process_segment(...)` call with:

```python
chunk_id, audio = await self._process_audio_segment(...)
face_tasks.append(asyncio.create_task(self._process_face_segment(chunk_id, audio, ...)))
```

After residual segment processing and before `_ensure_complete(artifacts)`, wait for:

```python
if face_tasks:
    await asyncio.gather(*face_tasks)
```

In `except` / `finally` paths, cancel unfinished Face tasks before writing final timeline.

- [ ] **Step 4: Preserve event semantics**

Move only Audio2Face, Face, UE5, and `server.segment.ready` into `_process_face_segment`. Keep `server.llm.chunk` and `server.tts.audio` in `_process_audio_segment`.

- [ ] **Step 5: Verify Task 2**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_orchestrator.py -q
```

Expected: stream orchestrator tests pass.

---

### Task 3: Regression verification and documentation

**Files:**
- Modify: `阶段式目标.md`
- Optionally modify: `docs/operations/stream-client.md`

**Interfaces:**
- Preserves command-line usage.
- Produces documented Task 7 current-state note.

- [ ] **Step 1: Update phase goals**

In `阶段式目标.md`, add a Task 7 note under Phase 3 explaining:

```text
当前执行方案 A：短文本分段、TTS 先发、Face/UE5 后台追赶。
方案 B：EmoTalk sidecar 作为后续任务。
方案 C：student FaceDriver 不阻塞当前迭代。
```

- [ ] **Step 2: Run full unit suite**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit -q
```

Expected: all unit tests pass.

- [ ] **Step 3: Run stream client smoke test**

With the mock server already started or by starting it manually, run:

```bash
PYTHONPATH=src BIONIC_CONFIG=config/mock.json .venv/bin/uvicorn bionic_head.api.app:create_app --factory --host 127.0.0.1 --port 8010
```

In another terminal:

```bash
PYTHONPATH=src .venv/bin/python scripts/stream_client.py \
  --url ws://127.0.0.1:8010/pipeline/stream \
  --wav /tmp/bionic-demo-input.wav \
  --output-dir /tmp/bionic-task7-stream \
  --chunk-ms 40
```

Expected: `terminal_event=server.pipeline.done` and summary contains at least one TTS chunk and one UE5 chunk.

- [ ] **Step 4: Commit**

Run:

```bash
git add docs/superpowers/specs/2026-06-23-task7-emotalk-segment-acceleration-design.md \
  docs/superpowers/plans/2026-06-23-task7-emotalk-segment-acceleration.md \
  src/bionic_head/core/sentences.py \
  src/bionic_head/config.py \
  src/bionic_head/orchestrators/stream.py \
  tests/unit/test_sentences.py \
  tests/unit/test_stream_orchestrator.py \
  config/mock.json \
  config/real.example.json \
  config/emotalk.example.json \
  阶段式目标.md
git commit -m "feat: accelerate face segment streaming"
```
