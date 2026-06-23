# Task 7: EmoTalk 分段加速与短音频适配设计

## 背景

真实链路回归 benchmark 已确认：Ollama 预热和 Piper 常驻后，首音频延迟已经明显下降，当前剩余体验瓶颈集中在 Face / Audio2Face / 灰模预览侧。最新 EmoTalk 调研给出的结论是：先执行方案 A，把 EmoTalk 当成可切块的离线 teacher，用 0.8–1.2 秒级小段尽快产出首段 `[N,52]`，不要等 sidecar 或 student FaceDriver 才改善体验。

本设计把调研结论落到当前 FastAPI / WebSocket pipeline 中。目标不是把 EmoTalk 改成真正流式模型，而是降低“第一段脸部帧可见”的等待感，并防止慢速 Audio2Face 阻塞后续 TTS 音频。

## 目标

1. `/pipeline/stream` 的 LLM 文本切分从“较长句子”调整为“适合 Face teacher 的短片段”。
2. 首段文本优先进入 TTS 和 Audio2Face，目标片段约 0.8–1.2 秒语音。
3. TTS 音频发送后，不再同步等待该段 Audio2Face 完成才继续消费 LLM token / 处理后续 TTS。
4. Audio2Face / UE5 仍按 turn_id 和 generation_epoch 做 stale 丢弃，旧 turn 不推送帧、不覆盖 latest。
5. 保持协议兼容：继续使用 `server.tts.audio`、`server.face.frames`、`server.ue5.frames`、`server.segment.ready`。
6. 使用 mock provider 增加可重复自动化测试，证明慢 Face 不会阻塞后续 TTS chunk。

## 非目标

1. 本任务不实现 EmoTalk sidecar / 常驻模型进程。
2. 本任务不训练 student FaceDriver。
3. 本任务不改 UE5 曲线映射。
4. 本任务不把 Blender 灰模渲染放入实时 WebSocket 路径。
5. 本任务不承诺“电话级真全双工”；它只是下一步低延迟体验优化。

## 设计

### 1. 短分段策略

当前 `SentenceBuffer` 只有 `sentence_max_chars`，遇到中文标点会立即 flush。这会产生两个问题：

- 过短片段如“你好！”可能只有约 0.25–0.5 秒，EmoTalk 能出形状但视觉质量和随机 blink 更容易不稳。
- 过长片段如 80 字会推迟首段 Face。

新增 `sentence_min_chars` 配置，保留 `sentence_max_chars`：

```text
sentence_min_chars: 默认 8
sentence_max_chars: 默认从 80 调整为 24
sentence_max_wait_ms: 保持 500
```

规则：

1. 标点只在累计字符数达到 `sentence_min_chars` 后触发分段。
2. 累计达到 `sentence_max_chars` 时强制分段，即使没有标点。
3. `sentence_max_wait_ms` 超时仍 flush 当前缓冲，避免 LLM 停顿导致永远不发。

这会让 “你好！很高兴见到你。” 合并成一个更适合作为首段 teacher chunk 的片段，同时让长回复按约 24 字以内拆开。

### 2. TTS 与 Face 解耦

当前 `_process_segment()` 的顺序是：

```text
LLM chunk -> TTS -> 发送 WAV -> Audio2Face -> Face frames -> UE5 frames -> segment.ready
```

这意味着 Audio2Face 慢时，第二段 TTS 也被卡住。Task 7 改成：

```text
LLM chunk -> TTS -> 发送 WAV -> 创建后台 Face task -> 继续处理后续 LLM / TTS
```

后台 Face task 内部完成：

```text
Audio2Face -> server.face.frames -> UE5 format -> server.ue5.frames -> server.segment.ready
```

主流程在 LLM 结束后等待所有 Face task 完成，再发布 latest 和 `server.pipeline.done`。这样保持最终产物完整，同时让客户端更早拿到后续音频。

### 3. 顺序、取消和 stale 丢弃

每个 Face task 都复用现有 `TurnHandle.emit_if_current()` 和 `TurnContext.cancellation`：

- 如果 turn 被 cancel，后台 task 收到 cancellation 后停止。
- 如果 generation_epoch 已过期，emit 前被拒绝并抛出 `CancelledError`。
- 旧 task 不写 latest；latest 只在所有当前 turn 的 Face task 完成后发布。

当前 `morpheus_max_concurrency: 1` / provider 内部串行限制继续保留，避免同时跑多个 EmoTalk / Morpheus 外部命令。

### 4. 指标与验收

自动化验收新增两个行为测试：

1. `SentenceBuffer` 不会把过短标点片段立即 flush，会等到达到 `sentence_min_chars`。
2. 当 mock Audio2Face 很慢且 LLM 回复被拆成多段时，第二个 `server.tts.audio` 必须早于第一个 `server.face.frames` 出现。

真实验收继续使用现有 benchmark：

```text
PYTHONPATH=src .venv/bin/python scripts/benchmark.py \
  --url http://127.0.0.1:<port> \
  --stream-url ws://127.0.0.1:<port>/pipeline/stream \
  --wav <测试 wav> \
  --runs 2 \
  --output <report.json>
```

预期变化：

- `e2e_first_audible_ms` 不应变慢。
- 多段回复时，后续 TTS chunk 更早到达。
- `face_first_chunk_ms` 可能随首段变短而下降；如果外部命令冷启动占绝对主导，则下降有限。
- `total_turn_duration_ms` 仍可能受 EmoTalk 外部进程影响，后续由 sidecar 解决。

## 风险

1. 太短的 chunk 可能表情质量弱；因此默认 `sentence_min_chars=8`，不追求 0.3 秒极短段。
2. 后台 Face task 增加事件交错可能性；通过 send lock 和 turn current 检查约束。
3. 如果某段 Face 失败，整个 turn 仍进入 `server.pipeline.error`，避免 latest 发布半成品。
4. 真实 EmoTalk 仍按外部命令启动，固定冷启动成本不会在本任务消失。

## 自检

- 无未决项。
- 协议兼容，未引入新事件类型。
- 范围聚焦在方案 A，不包含 sidecar / student / UE5 映射。
- 取消与 stale 丢弃沿用现有 TurnHandle 机制。
