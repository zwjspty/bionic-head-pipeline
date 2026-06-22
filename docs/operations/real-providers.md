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

## Validation order

```bash
curl -s http://127.0.0.1:8000/diagnostics
BIONIC_CONFIG=config/local.json .venv/bin/python -m pytest tests/integration/providers/test_ollama.py -m integration -v
BIONIC_CONFIG=config/local.json BIONIC_TEST_WAV=/path/to/chinese.wav .venv/bin/python -m pytest tests/integration/providers/test_faster_whisper.py -m integration -v
BIONIC_CONFIG=config/local.json .venv/bin/python -m pytest tests/integration/providers/test_piper.py -m integration -v
BIONIC_CONFIG=config/local.json BIONIC_TEST_TTS_WAV=/path/to/tts.wav .venv/bin/python -m pytest tests/integration/providers/test_morpheus.py -m integration -v
BIONIC_CONFIG=config/local.json BIONIC_TEST_WAV=/path/to/chinese.wav .venv/bin/python -m pytest tests/integration/test_real_pipeline.py -m integration -v
```

Then run the protocol client:

```bash
.venv/bin/python scripts/stream_client.py \
  --url ws://127.0.0.1:8000/pipeline/stream \
  --wav /path/to/chinese.wav \
  --output-dir client-output
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
