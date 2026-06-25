# Bionic Head Pipeline

中文端到端准实时数字人原型：语音输入 → ASR → LLM → TTS → Audio2Face → `morpheus_52_raw` 52 维表情帧，并通过 HTTP / WebSocket 对外提供可测试、可观测、可打断的实验室级交互链路。

当前项目已经不是单纯概念验证。它已经完成：

- Mock / Real provider 双线架构。
- `POST /pipeline/audio` HTTP 完整链路。
- `WS /pipeline/stream` 句子级增量链路。
- faster-whisper、Ollama qwen2.5:3b、Piper 中文 TTS、EmoTalk sidecar、`morpheus_52_raw` formatter。
- EmoTalk 常驻 sidecar：冷启动移到 startup/prewarm，热路径约 0.37–0.52s。
- stream timing、benchmark、generation_epoch / stale-drop、face crossfade、eye continuity 框架。
- Blender 灰模头预览视频，作为 UE5 完成前的正式视觉验收方式。

当前阶段更准确地称为：

```text
实验室级准实时数字人 MVP
—— 正在从“管线能跑且较快”
进入“真实交互正确性与体验收口”
```

## 当前真实能力与边界

已完成：

- `POST /pipeline/audio`：上传 WAV，跑通 ASR → LLM → TTS → Audio2Face → UE5 JSON。
- `WS /pipeline/stream`：PCM16LE 输入，输出 LLM token、句子级 WAV、face frames、UE5 frame chunks。
- `GET /health`、`GET /diagnostics`、`GET /pipeline/latest`、`GET /ue5/latest`。
- 自动化测试默认使用 mock provider，不依赖 GPU、Conda、Ollama、Piper、Morpheus 或真实音频。
- 真实 stream benchmark 已记录首音频、首可见脸、Face 热路径、stale-drop 等指标。

仍未完成或只是框架：

- 真实 UE5 工程尚未接入；当前只输出 `morpheus_52_raw` 中间格式。
- Eye continuity 框架已完成，但默认 no-op；真实 blink/eye 效果需要确认 52 维通道映射后配置 channel indices。
- Barge-in 是 RMS 实验版；尚未完成真实播放客户端、AEC、真实播放停止延迟验收。
- 多轮对话记忆未实现；当前每个 turn 调 LLM 时 history 仍为空。
- ASR 仍是 endpoint 后整段识别，不是 partial/stable/final 真流式 ASR。
- EmoTalk sidecar 已把 Face 从 8–16s 降到约 0.5s，但仍是最终 face 推理瓶颈之一。

## 关键 benchmark 基线

截至 Task 10 后，真实 stream 基线约为：

```text
10/10 success
tts_first_audio_ms p50        ≈ 583 ms
e2e_first_visible_face_ms p50 ≈ 1062 ms
face_total_ms p50             ≈ 469 ms
face_total_ms p90             ≈ 519 ms
old_turn_face_leak_count      = 0
stale_face_drop_count         = 0
```

解释：

- 管线已经较快：Face 热路径从旧方式 8–16s 降到 0.4–0.5s。
- 但真实音画同步还没完成：如果客户端收到 WAV 就立刻播放，脸通常晚约 0.48s 才可见。
- 下一阶段比继续压模型更重要的是取消正确性、sidecar 通信对齐、真实播放时间轴和多轮记忆。

## 本地启动

```bash
python3.11 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -e '.[dev,client]'
BIONIC_CONFIG=config/mock.json \
  .venv/bin/uvicorn bionic_head.api.app:create_app \
  --factory --host 127.0.0.1 --port 8000
```

运行测试：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

默认配置文件是 `config/mock.json`。可通过环境变量覆盖：

```bash
BIONIC_CONFIG=config/emotalk.example.json \
  .venv/bin/uvicorn bionic_head.api.app:create_app \
  --factory --host 127.0.0.1 --port 8005
```

## 主要配置

- Mock 默认：`config/mock.json`
- 真实 provider 模板：`config/real.example.json`
- EmoTalk sidecar 本机模板：`config/emotalk.example.json`
- 本机私有路径：请放在未提交的 `config/local.json`
- 数据输出：`data/runs/{session_id}/{turn_id}/`
- latest 输出：`data/latest/latest_pipeline.json`、`data/latest/latest_ue5_blendshape.json`

注意：`config/emotalk.example.json` 仍是示例模板，包含本机示例路径。公开部署时应复制到 `config/local.json` 后改成本机真实路径或环境变量注入。

## 文档入口

建议网页端 GPT 先读这些文件：

1. [`项目说明goal.md`](项目说明goal.md)
2. [`阶段式目标.md`](阶段式目标.md)
3. [`docs/status/2026-06-24-current-state.md`](docs/status/2026-06-24-current-state.md)
4. [`docs/protocols/bionic-head-stream-v1.md`](docs/protocols/bionic-head-stream-v1.md)
5. [`docs/protocols/bionic-head-ue5-v1.md`](docs/protocols/bionic-head-ue5-v1.md)
6. [`docs/operations/real-providers.md`](docs/operations/real-providers.md)
7. [`docs/operations/local-demo-client.md`](docs/operations/local-demo-client.md)
8. 最新计划：`docs/superpowers/plans/2026-06-24-task10-eye-continuity.md`

## 推荐下一阶段

下一阶段不是继续盲目压低模型耗时，而是进入：

```text
Task 11: 取消正确性与有序 Face 管线
```

核心目标：

- sidecar request pump / drain-and-discard，避免取消旧 turn 后 stdout 响应错位。
- Face 推理可异步完成，但 stitch / eye continuity / UE5 emit 必须按 `segment_index` 顺序释放。
- 避免在 turn 状态锁内等待网络 I/O；引入 connection 级高优先级出站队列。
- 取消正在推理的旧 turn 后，新 turn 的 Face 请求不能读到旧 response。

后续路线：

```text
Task 12: 真实播放客户端与音画同步
Task 13: session 多轮记忆
Task 14: 确认 52 维通道映射并激活真实 eye/blink/mouth channel groups
Task 15: AEC、VAD 与 partial ASR
Task 16: 真实 UE5 接入
Task 17: Student FaceDriver
```

## 协议声明

当前只声明 `morpheus_52_raw`：

```text
protocol: bionic-head-ue5-v1
format: morpheus_52_raw
fps: 30
channel_count: 52
channels: morpheus_00 ~ morpheus_51
```

目前不声明 ARKit、MetaHuman 或任何真实 UE5 曲线映射。52 维到 UE5 曲线名的映射表是后续单独任务。
