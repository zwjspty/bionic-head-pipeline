# Bionic Head P0 Mock Service

本仓库实现首版 P0：全 Mock、可重复测试的端到端语音到 UE5 52 维表情流水线。

当前能力：

- `POST /pipeline/audio`：上传 WAV，跑通 ASR → LLM → TTS → Audio2Face → UE5 JSON。
- `WS /pipeline/stream`：PCM16LE 伪流式输入，输出 LLM token、句子级 WAV、face frames、UE5 frame chunks。
- `GET /health`、`GET /diagnostics`、`GET /pipeline/latest`、`GET /ue5/latest`。
- Mock provider 支持固定输出、失败、超时和延迟配置。
- 默认测试不依赖 GPU、Conda、Ollama、Piper、Morpheus 或真实音频文件。

## 本地启动

```bash
python3.11 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/uvicorn bionic_head.api.app:create_app --factory --host 127.0.0.1 --port 8000
.venv/bin/python -m pytest -m 'not integration'
```

默认配置文件是 `config/mock.json`。可通过环境变量覆盖：

```bash
BIONIC_CONFIG=config/mock.json .venv/bin/uvicorn bionic_head.api.app:create_app --factory
```

## 主要配置

- Mock 默认：`config/mock.json`
- 真实 provider 模板：`config/real.example.json`
- 数据输出：`data/runs/{session_id}/{turn_id}/`
- latest 输出：`data/latest/latest_pipeline.json`、`data/latest/latest_ue5_blendshape.json`

## 协议文档

- WebSocket：`docs/protocols/bionic-head-stream-v1.md`
- UE5 JSON：`docs/protocols/bionic-head-ue5-v1.md`
- Mock 开发：`docs/operations/mock-development.md`

P0 只声明 `morpheus_52_raw`，不声明 ARKit 或 MetaHuman 映射。
