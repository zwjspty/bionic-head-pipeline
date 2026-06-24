# Real provider deployment runbook

Mock remains the default for development and automated tests. Real providers are for deployment validation.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -e '.[dev,all]'
cp config/real.example.json config/local.json
```

Run with:

```bash
BIONIC_CONFIG=config/local.json .venv/bin/uvicorn bionic_head.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

For the current local machine, where EmoTalk is available but the original Morpheus
environment is not, start from the EmoTalk template instead:

```bash
cp config/emotalk.example.json config/local.json
```

## Fill `config/local.json`

1. Confirm Ollama is running and `qwen2.5:3b` is available.
2. Confirm faster-whisper can access the configured `base` model.
3. Fill Piper:
   - `providers.piper.executable`
   - `providers.piper.model_path`
   - `providers.piper.args`
4. Fill Morpheus:
   - replace the empty command element in `providers.morpheus.args`
   - confirm `providers.morpheus.cwd`
   - confirm output globs for `.npy` and `.json`
5. Keep `morpheus_max_concurrency: 1` until safe parallel execution is confirmed.

Unknown Piper or Morpheus command data is a deployment blocker for their real smoke tests, not a blocker for the completed Mock service.

## Local EmoTalk Audio2Face option

`audio2face.provider = "emotalk"` is a real local Audio2Face provider for the
available EmoTalk model. It reuses the same external-command safety behavior as
the Morpheus adapter: argument-array execution, timeout handling, cancellation,
single-output validation, and `[N, 52]` finite blendshape checks.

Use it when this machine has:

- `/home/user/miniconda3/bin/conda`;
- Conda env `emotalk`;
- `/home/user/code/EmoTalk_release/scripts/export_blendshape_from_audio.py`;
- output written as `{output_dir}/emotalk.npy`.

`audio2face.provider = "emotalk_sidecar"` is the preferred low-latency local
option for stream-style validation. It keeps a Python worker alive and sends
raw PCM over the sidecar binary protocol, so the model is loaded once instead
of once per segment.

Use the Conda environment's Python executable directly:

```json
{
  "sidecar_command": [
    "/home/user/miniconda3/envs/emotalk/bin/python",
    "-m",
    "bionic_head.emotalk_sidecar_worker"
  ],
  "sidecar_cwd": "/home/user/code/端到端",
  "sidecar_env": {
    "PYTHONPATH": "src:."
  }
}
```

Do not use `conda run` for the stdin/stdout sidecar protocol. On this machine
it can leave stdout empty and make the worker see EOF before it handles the
request. The sidecar provider reports this as unsupported so deployment config
does not silently fall back to the broken launch mode.

This does not mean the deployment Morpheus provider is complete. Morpheus remains
blocked until the `lyyMor` environment, Morpheus project path, and exact command
format are confirmed on the target machine.

## Validation order

```bash
curl -s http://127.0.0.1:8000/diagnostics
BIONIC_CONFIG=config/local.json .venv/bin/python -m pytest tests/integration/providers/test_ollama.py -m integration -v
BIONIC_CONFIG=config/local.json BIONIC_TEST_WAV=/path/to/chinese.wav .venv/bin/python -m pytest tests/integration/providers/test_faster_whisper.py -m integration -v
BIONIC_CONFIG=config/local.json .venv/bin/python -m pytest tests/integration/providers/test_piper.py -m integration -v
BIONIC_CONFIG=config/local.json BIONIC_TEST_TTS_WAV=/path/to/tts.wav .venv/bin/python -m pytest tests/integration/providers/test_morpheus.py -m integration -v
BIONIC_CONFIG=config/local.json BIONIC_TEST_WAV=/path/to/chinese.wav .venv/bin/python -m pytest tests/integration/test_real_pipeline.py -m integration -v
```

When using `config/emotalk.example.json`, the Morpheus provider smoke test is not
the right isolated check; validate the full pipeline instead and confirm
`/diagnostics` reports `audio2face.provider = emotalk_sidecar`.

For an isolated sidecar smoke test:

```bash
BIONIC_HEAD_RUN_REAL_EMOTALK=1 \
BIONIC_HEAD_REAL_EMOTALK_COMMAND="/home/user/miniconda3/envs/emotalk/bin/python -m bionic_head.emotalk_sidecar_worker" \
PYTHONPATH=src:. \
.venv/bin/python -m pytest tests/integration/test_emotalk_sidecar_worker_real.py -q
```

Then run the protocol client:

```bash
.venv/bin/python scripts/stream_client.py \
  --url ws://127.0.0.1:8000/pipeline/stream \
  --wav /path/to/chinese.wav \
  --output-dir client-output
```

## Grey-head preview before UE5

Before the real UE5 character is connected, use the EmoTalk Blender grey-head
renderer for visual acceptance. Do not use the simplified bar-chart preview for
product demos.

The renderer consumes the same real pipeline artifacts:

- Piper reply WAV;
- EmoTalk/Morpheus-style `[N, 52]` blendshape `.npy`;
- `/home/user/code/EmoTalk_release/render.blend`;
- `/home/user/code/EmoTalk_release/render.py`;
- `/home/user/code/EmoTalk_release/blender/blender`.

Render one pipeline result:

```bash
PYTHONPATH=src .venv/bin/python scripts/render_emotalk_grey_head.py \
  --face-npy /tmp/my-bionic-face.npy \
  --audio-wav /tmp/my-bionic-reply.wav \
  --output /tmp/my-bionic-grey-head.mp4 \
  --name my-bionic \
  --work-dir /tmp/my-bionic-grey-render
```

Play it:

```bash
ffplay /tmp/my-bionic-grey-head.mp4
```

## Benchmark

Use at least 10 turns for acceptance evidence:

```bash
.venv/bin/python scripts/benchmark.py \
  --mode stream \
  --ws-url ws://127.0.0.1:8000/pipeline/stream \
  --wav /path/to/chinese.wav \
  --runs 10 \
  --output latency_report.json
```

Check:

- real WAV is recognized;
- Ollama returns `reply`, `emotion`, and `intensity`;
- Piper produces playable WAV;
- Morpheus produces finite `[N, 52]`;
- UE5 payload uses `protocol=bionic-head-ue5-v1` and `format=morpheus_52_raw`;
- `data/latest/*` belongs to the successful current turn;
- client output contains ordered events, WAV chunks, UE5 JSON chunks, and `summary.json`;
- benchmark report contains P50/P90 latency summaries.

For focused Audio2Face A/B evidence between the old per-request EmoTalk command
and the persistent sidecar:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/benchmark_emotalk_sidecar.py \
  --config config/emotalk.example.json \
  --wav /path/to/tts-or-input.wav \
  --old-runs 1 \
  --sidecar-runs 3 \
  --output data/benchmarks/emotalk_sidecar_latest.json
```
