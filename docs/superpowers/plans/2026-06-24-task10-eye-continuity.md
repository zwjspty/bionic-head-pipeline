# Task 10 Eye Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add session-level eye continuity and optional blink scheduling for stream face frames without changing sidecar, audio, provider, or UE5 protocol behavior.

**Architecture:** Add a pure `EyeContinuityProcessor` that owns only eye/blink state and metrics. Configure it through `AppSettings.eye_continuity`, instantiate one processor per stream run, apply it after Task 9 face stitching and before UE5 formatting, then expose metrics through existing stream timing, stream client summaries, and benchmark reports.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, FastAPI/Uvicorn stream path, pure Python list math for `morpheus_52_raw` frame arrays.

## Global Constraints

- Work on branch `task10-eye-continuity`.
- Do not hard-code 52-dimensional eye or blink channel semantics.
- Do not change EmoTalk sidecar, sidecar protocol, real provider loading, or benchmark startup/prewarm behavior.
- Do not change ASR, LLM, TTS, audio overlap, GPU behavior, student FaceDriver, or UE5 event type names.
- Preserve frame count, per-frame timestamps, channel count, and UE5 chunking.
- Default configuration must not modify frames because eye/blink channel lists are empty.
- Real visual eye/blink changes require explicit channel indices in JSON config.
- Automated tests must not require real EmoTalk, Conda, GPU, Ollama, Piper, or network access.
- Existing Task 8 stale-drop and Task 9 crossfade metrics must remain available.

---

## File Structure

```text
src/bionic_head/eye_continuity.py
  New pure module with EyeContinuityProcessor and EyeContinuityMetrics.

src/bionic_head/config.py
  Adds EyeContinuitySettings and AppSettings.eye_continuity.

config/mock.json
config/real.example.json
config/emotalk.example.json
  Add explicit eye_continuity JSON block for strict Pydantic config.

src/bionic_head/orchestrators/stream.py
  Creates per-run processor and applies it after FaceSegmentStitcher.

scripts/benchmark.py
  Promotes eye continuity metrics from stream_client summaries into latency reports.

tests/unit/test_eye_continuity.py
tests/unit/test_config.py
tests/unit/test_stream_orchestrator.py
tests/unit/test_stream_client.py
tests/unit/test_benchmark.py
tests/integration/test_stream_emotalk_sidecar_real.py
  Regression and smoke coverage.
```

---

### Task 1: Pure EyeContinuityProcessor

**Files:**
- Create: `src/bionic_head/eye_continuity.py`
- Create: `tests/unit/test_eye_continuity.py`

**Interfaces:**
- Produces:
  - `EyeContinuityMetrics.to_timing_payload() -> dict[str, bool | float]`
  - `EyeContinuityProcessor(...).process(frames, session_id, turn_id, generation_epoch, segment_index, fps) -> tuple[list[list[float]], EyeContinuityMetrics]`
  - `EyeContinuityProcessor.reset() -> None`
- Consumes: no app config, no provider types, no FastAPI.

- [ ] **Step 1: Write failing processor tests**

Create `tests/unit/test_eye_continuity.py` with tests for no-op defaults, channel-limited smoothing, reset behavior, deterministic blink, frame-count preservation, and disabled mode.

- [ ] **Step 2: Run RED test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_eye_continuity.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'bionic_head.eye_continuity'`.

- [ ] **Step 3: Implement pure module**

Create `src/bionic_head/eye_continuity.py` with:

```python
EyeContinuityMetrics
EyeContinuityProcessor
```

Implementation rules:

- Deep-copy input frames before modification.
- Validate rectangular frames before accessing configured channels.
- Validate configured channel indices are within `[0, channel_count - 1]`.
- Smooth only configured `eye_smooth_channel_indices`.
- Blink only configured `blink_channel_indices`.
- Store previous eye tail for same-session/same-turn/same-generation consecutive smoothing.
- Maintain global frame index across segments for blink scheduling.

- [ ] **Step 4: Run GREEN test**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_eye_continuity.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/bionic_head/eye_continuity.py tests/unit/test_eye_continuity.py
git commit -m "feat: add eye continuity processor"
```

---

### Task 2: Configuration

**Files:**
- Modify: `src/bionic_head/config.py`
- Modify: `config/mock.json`
- Modify: `config/real.example.json`
- Modify: `config/emotalk.example.json`
- Modify: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `EyeContinuityProcessor` constructor arguments.
- Produces: `settings.eye_continuity`.

- [ ] **Step 1: Write failing config tests**

Add tests proving:

- Defaults are safe no-op: enabled true, blink false, channel lists empty.
- Explicit config accepts channel indices and blink parameters.
- Out-of-range channel indices fail validation.
- `blink_interval_max_sec < blink_interval_min_sec` fails validation.

- [ ] **Step 2: Run RED tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_config.py -q
```

Expected: fail because `AppSettings` has no `eye_continuity`.

- [ ] **Step 3: Implement settings and JSON blocks**

Add `EyeContinuitySettings` to `src/bionic_head/config.py`, then add explicit `eye_continuity` blocks to all tracked JSON configs.

- [ ] **Step 4: Run GREEN tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_config.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/bionic_head/config.py config/mock.json config/real.example.json config/emotalk.example.json tests/unit/test_config.py
git commit -m "feat: configure eye continuity"
```

---

### Task 3: Stream Integration

**Files:**
- Modify: `src/bionic_head/orchestrators/stream.py`
- Modify: `tests/unit/test_stream_orchestrator.py`
- Modify: `tests/integration/test_stream_emotalk_sidecar_real.py`

**Interfaces:**
- Consumes: `EyeContinuityProcessor.process(...)`.
- Produces: eye continuity timing in `server.face.frames`, `server.ue5.frames`, and timeline segment payloads.

- [ ] **Step 1: Write failing stream tests**

Add tests proving:

- Default stream timing includes eye metrics but does not apply changes.
- Configured eye channels apply on the second same-turn segment and report `eye_boundary_delta_after <= eye_boundary_delta_before`.
- Existing stale/cancel tests still have no `server.ue5.frames` for stale turns.

- [ ] **Step 2: Run RED stream test**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_orchestrator.py -q
```

Expected: fail because timing has no eye continuity keys.

- [ ] **Step 3: Implement stream integration**

Update `StreamOrchestrator.run(...)` to create one `EyeContinuityProcessor` from settings and pass it into `_process_face_segment(...)`. Apply it immediately after `FaceSegmentStitcher.stitch(...)` and before `server.face.frames` / UE5 formatting. Extend `_StreamSegmentTiming` with eye metric fields and an `apply_eye_continuity(...)` method.

- [ ] **Step 4: Run GREEN stream tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_orchestrator.py tests/integration/test_stream_emotalk_sidecar_real.py -q
```

Expected: pass, with real EmoTalk smoke skipped unless its environment variable is set.

- [ ] **Step 5: Commit**

```bash
git add src/bionic_head/orchestrators/stream.py tests/unit/test_stream_orchestrator.py tests/integration/test_stream_emotalk_sidecar_real.py
git commit -m "feat: apply eye continuity in stream"
```

---

### Task 4: Client and Benchmark Metrics

**Files:**
- Modify: `scripts/benchmark.py`
- Modify: `tests/unit/test_stream_client.py`
- Modify: `tests/unit/test_benchmark.py`

**Interfaces:**
- Consumes: stream timing keys from Task 3.
- Produces: benchmark metrics for eye continuity and blink counts.

- [ ] **Step 1: Write failing client/benchmark tests**

Add tests proving:

- Stream client segment summaries preserve eye timing keys.
- Benchmark extracts first-segment eye config metrics.
- Benchmark counts `eye_continuity_applied`, `eye_continuity_reset`, `blink_applied_count`, `blink_frame_count`, and `blink_reset_count` across all segments.

- [ ] **Step 2: Run RED tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_client.py tests/unit/test_benchmark.py -q
```

Expected: fail because benchmark does not promote eye metrics.

- [ ] **Step 3: Implement metric extraction**

Update `scripts/benchmark.py` to promote the first-segment metrics and aggregate count metrics across all stream segments. Change `scripts/stream_client.py` only if tests show existing merge behavior does not preserve booleans/numbers.

- [ ] **Step 4: Run GREEN tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_stream_client.py tests/unit/test_benchmark.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark.py tests/unit/test_stream_client.py tests/unit/test_benchmark.py
git commit -m "feat: report eye continuity benchmark metrics"
```

---

### Task 5: Full Verification

**Files:**
- All changed files from Tasks 1-4.

**Interfaces:**
- Consumes: full Task 10 implementation.
- Produces: verified Task 10 branch ready for review/merge.

- [ ] **Step 1: Run targeted tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_eye_continuity.py \
  tests/unit/test_config.py \
  tests/unit/test_stream_orchestrator.py \
  tests/unit/test_stream_client.py \
  tests/unit/test_benchmark.py \
  tests/integration/test_stream_emotalk_sidecar_real.py \
  -q
```

Expected: pass, with real EmoTalk smoke skipped unless explicitly enabled.

- [ ] **Step 2: Run full test suite**

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Inspect git status and log**

```bash
git status --short
git log --oneline -5
```

Expected: no uncommitted files after final commit.

- [ ] **Step 4: Commit any final test/doc fixes**

```bash
git add <changed-files>
git commit -m "test: verify eye continuity integration"
```

Use this only if Task 5 produces additional changes.

## Self-Review Checklist

- [ ] Spec requirement “no hard-coded blink channels” maps to config validation and empty defaults.
- [ ] Spec requirement “default no-op” maps to pure processor tests and config tests.
- [ ] Spec requirement “stream timing metrics” maps to stream tests and benchmark tests.
- [ ] Spec requirement “old turn does not leak” remains covered by existing stale/drop tests.
- [ ] No task changes sidecar, protocol, audio overlap, ASR, LLM, TTS, GPU, or student FaceDriver.
