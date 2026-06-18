# 模块化流式仿生人头系统首版设计

**日期：** 2026-06-18

**范围：** 阶段一稳定化与阶段二伪流式原型

**状态：** 已确认设计

## 1. 目标与范围

本轮从零创建一个“外部端到端、内部模块化”的仿生人头交互服务，同时交付：

- 稳定的离线完整链路 `POST /pipeline/audio`。
- 伪流式交互链路 `WS /pipeline/stream`。
- 可替换的 ASR、LLM、TTS、Audio2Face 和 UE5 adapter。
- 可重复的全 Mock 自动化测试基线。
- 可用于部署验收的真实 provider 实现与配置入口。
- 完整的时间线、诊断、错误分类、产物保存和取消机制。

本轮不建设真流式 ASR、裸 PCM 流式 TTS、实时 FaceDriver、多会话并发、UE5 曲线映射、Windows 兼容、容器化、登录鉴权或公网安全。

## 2. 成功标准

### 2.1 P0

- 全 Mock 的 `/pipeline/audio` 稳定通过自动化测试。
- 全 Mock 的 `/pipeline/stream` 协议、状态机、取消、失败和超时测试稳定通过。
- 每个 pipeline 都能产出完整 `timeline.json`。
- `/health`、`/diagnostics`、`/diagnostics/{adapter}` 可用。
- `session_id`、`turn_id`、`event_id` 和事件顺序在整条链路中可追踪。
- 被取消或过期的 turn 不会继续推送，也不会覆盖 latest。

### 2.2 P1

- Ollama、faster-whisper、Piper、Morpheus 按顺序接入。
- 真实 `/pipeline/audio` 能产生回复 WAV、`[N, 52]` blendshape 和 UE5 JSON。
- 真实 `/pipeline/stream` 能按句输出 WAV 和对应 frame chunk。
- WebSocket 测试客户端能够接收协议事件、WAV 二进制和 52 维帧。
- 真实环境生成 P50/P90 延迟报告；1 秒仅作为 stretch goal。

## 3. 技术基线

```text
Python 3.11
FastAPI + Uvicorn
Pydantic v2
asyncio
pytest
pyproject.toml + pip
Linux 本地部署
JSON 配置
```

服务仅面向本机或实验室内网。首版不提供鉴权、HTTPS、访问控制和审计。

## 4. 总体架构

首版采用模块化单体服务。FastAPI、编排层、provider registry、状态管理和产物管理运行在同一进程；Piper 和 Morpheus 使用受控子进程隔离。

未选择微服务，是因为首版只有单机、单活跃会话，拆分服务会增加部署和故障处理成本。未选择 Redis 或消息总线，是因为进程内队列足以满足当前吞吐，并且更容易测试取消和顺序。

adapter 接口不依赖 FastAPI。未来拆分微服务时，编排层调用的领域接口保持不变，只替换 adapter 实现。

```text
FastAPI API 层
├── POST /pipeline/audio
├── WS /pipeline/stream
├── GET /health
├── GET /diagnostics
├── GET /diagnostics/{adapter}
├── GET /pipeline/latest
└── GET /ue5/latest
          │
          ▼
Pipeline Orchestrator
├── OfflineOrchestrator
└── StreamOrchestrator
          │
          ▼
Provider Registry
├── ASR: mock / faster-whisper
├── LLM: mock / ollama
├── TTS: mock / piper
├── Audio2Face: mock / morpheus
└── UE5: mock / morpheus-raw
          │
          ▼
基础设施
├── Session/Turn 状态管理
├── 任务取消与子进程管理
├── Timeline/Diagnostics
├── Artifact 存储
└── Latest 原子发布
```

API 只解析请求和发送事件，不直接调用模型。Offline 和 Stream 共用相同 adapter。阻塞调用通过线程或受控子进程运行，不阻塞 asyncio 事件循环。

## 5. Provider 双线策略

首版同时建设 Mock provider 和真实 provider。

### 5.1 Mock 线

自动化测试和本地开发默认使用：

```text
ASR mock
LLM mock
TTS mock
Audio2Face mock
UE5 formatter mock
```

每个 Mock 支持固定输出、可配置延迟、指定失败、指定超时、asyncio 取消和取消后迟到结果。Mock 用于验证 API、协议、状态机、产物、错误、超时和旧结果丢弃，不追求真实效果。

### 5.2 真实线

部署验收使用：

```text
ASR: faster-whisper base / CPU / int8 / zh
LLM: Ollama http://127.0.0.1:11434 / qwen2.5:3b
TTS: Piper / zh_CN-huayan-medium
Audio2Face: Morpheus / conda env lyyMor
UE5: morpheus_52_raw
```

Piper adapter 使用可配置命令参数数组，不使用 shell 字符串拼接。配置必须提供可执行文件、模型路径、输入方式和输出 WAV 参数；缺失时 diagnostics 返回 unavailable，选择该 provider 执行 pipeline 时返回 `provider_unavailable`。

Morpheus adapter 使用可配置命令参数数组，默认命令形态为：

```text
conda run -n lyyMor <command> --input <wav_path> --output-dir <output_dir>
```

已知环境：

```text
项目路径: /home/hailab/liuyiyu/head-project/Morpheus-Software
Conda 环境: lyyMor
仿真项目: /home/hailab/liuyiyu/head-project/Simulation
```

实际命令与输出命名未配置时，Morpheus diagnostics 返回 unavailable，但不影响 Mock 开发和普通测试。

真实 provider 接入顺序固定为：

1. Ollama streaming。
2. faster-whisper 分段识别。
3. Piper 句子级 TTS。
4. Morpheus 句子级处理。
5. 测试客户端和 UE5 播放验证。

## 6. 配置

配置使用 JSON，至少包含：

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8000,
    "max_active_sessions": 1
  },
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
  "retention": {
    "max_runs": 100
  },
  "adapters": {
    "asr": {"provider": "mock"},
    "llm": {"provider": "mock"},
    "tts": {"provider": "mock"},
    "audio2face": {"provider": "mock"},
    "ue5": {"provider": "mock"}
  },
  "limits": {
    "morpheus_max_concurrency": 1,
    "subprocess_terminate_grace_seconds": 2
  }
}
```

仓库提供 Mock 默认配置和真实 provider 配置示例。真实路径不写死在代码中。

PCM RMS 先归一化到 `[0.0, 1.0]`。连续 1000ms 的 RMS 不高于 `0.01` 时触发静音兜底结束；整段音频的 RMS 不高于该阈值时返回 `no_speech_detected`。

## 7. 领域模型与 Adapter 接口

adapter 接收和返回标准领域对象，编排层不解析厂商私有返回值。

```text
ASR.transcribe(audio, context) -> ASRResult
LLM.chat(text, history, context) -> LLMResult
LLM.chat_stream(text, history, context) -> AsyncIterator[LLMEvent]
TTS.synthesize(text, emotion, intensity, context) -> AudioArtifact
Audio2Face.drive(audio, emotion, intensity, context) -> FaceArtifact
UE5.format(face_artifact, context) -> UE5Payload
```

每个 adapter 还提供：

```text
diagnostics() -> DiagnosticResult
cancel(turn_id) -> None
```

`context` 至少携带：

```text
session_id
turn_id
artifact_directory
cancellation_token
```

统一结果约束：

- `ASRResult` 包含文本、语言、置信信息和音频统计。
- `LLMResult` 包含 reply、emotion、intensity 和原始解析诊断。
- emotion 使用 `neutral/friendly/happy/sad/angry/surprised/thinking/calm`。
- intensity 是 `[0.0, 1.0]` 浮点数。
- `AudioArtifact` 包含 WAV 路径、采样率、时长和字节数。
- `FaceArtifact` 包含 fps、`[N, 52]` 权重和产物路径。
- `UE5Payload` 使用 `bionic-head-ue5-v1` 与 `morpheus_52_raw`。

## 8. 离线 Pipeline

`POST /pipeline/audio` 接收 WAV 文件并执行：

```text
校验并保存 input.wav
-> 音频统计与静音检测
-> ASR
-> LLM 完整回复与情绪解析
-> TTS 完整 WAV
-> Audio2Face [N, 52]
-> UE5 JSON
-> 原子更新 latest
-> 返回完整结果和 timeline
```

静音输入在 ASR 后返回 `no_speech_detected`，不调用 LLM、TTS 或 Audio2Face。

只有全部成功且 turn 仍有效时才更新 latest。任何失败都保留当前运行目录及 timeline。

## 9. 伪流式 Pipeline

客户端输入固定为：

```text
PCM signed 16-bit little-endian
单声道
16000 Hz
每个 chunk 20–100ms
```

数据流：

```text
client.audio.start
-> audio chunk 元数据 + PCM binary
-> client.audio.end / 1000ms 静音 / 30s 最大时长
-> 保存 input.wav
-> ASR final
-> LLM token streaming
-> 句子切分
-> 每句生成完整 WAV
-> server.tts.audio 元数据 + WAV binary
-> Morpheus 处理该 WAV
-> server.face.frames / server.ue5.frames
-> server.segment.ready
-> 后续句子
-> server.pipeline.done
```

阶段二等待 `asr.final` 后才启动 LLM，不基于 partial text 提前生成。

句子切分优先使用 `。！？!?；;\n`。没有标点时，累计到 80 个字符立即切分；已产生可说文本但 500ms 内没有新 token 时强制切分。这两个默认值可以配置，但不构成外部协议。

同一 turn 内句子按顺序执行 TTS 和 Audio2Face，避免音频与面部帧乱序。首版不并行处理多个句子。

## 10. WebSocket 协议

协议名为 `bionic-head-stream-v1`。所有 JSON 事件使用统一信封：

```json
{
  "protocol": "bionic-head-stream-v1",
  "type": "server.asr.final",
  "event_id": "uuid",
  "session_id": "uuid",
  "turn_id": "uuid",
  "sequence": 12,
  "timestamp": "2026-06-18T12:00:00.000Z",
  "payload": {}
}
```

`sequence` 在单条连接内单调递增。

### 10.1 客户端事件

```text
client.session.start
client.audio.start
client.audio.chunk
client.audio.end
client.turn.cancel
client.ping
```

`client.audio.chunk` JSON 元数据之后必须紧跟对应 PCM binary。同一连接不能同时存在两个等待 binary 的元数据事件。

### 10.2 服务端事件

```text
server.session.ready
server.state
server.asr.final
server.llm.token
server.llm.chunk
server.tts.audio
server.face.frames
server.ue5.frames
server.segment.ready
server.turn.cancelled
server.pipeline.done
server.pipeline.error
server.pong
```

`server.tts.audio` JSON 之后紧跟对应 WAV binary，元数据包含：

```text
session_id
turn_id
chunk_id
format: wav
sample_rate
byte_length
duration_seconds
```

`server.segment.ready` 表示同一 `chunk_id` 的 WAV 和 face frames 均已就绪。客户端可提前缓存 WAV，在该事件后同步启动音频和面部播放。

UE5 长数据每个事件最多包含 30 帧，并带：

```text
chunk_id
frame_offset
frame_count
fps
start_time_seconds
is_last
frames
```

## 11. UE5 数据格式

```text
protocol: bionic-head-ue5-v1
format: morpheus_52_raw
fps: 30
channel_count: 52
channels: morpheus_00 ~ morpheus_51
```

每帧包含：

```text
frame_index: int
time_seconds: float
weights: float[52]
```

该格式不宣称兼容 ARKit 或 MetaHuman。曲线映射属于 P2。

## 12. 状态机

正常路径：

```text
IDLE
  -> client.audio.start
LISTENING
  -> client.audio.end / VAD / 最大时长
THINKING
  -> 首个 segment ready
SPEAKING
  -> 全部片段完成
IDLE
```

取消路径：

```text
LISTENING / THINKING / SPEAKING
  -> CANCELLING
  -> IDLE
```

异常路径：

```text
任意活动状态
  -> ERROR
  -> server.pipeline.error
  -> 清理当前 turn
  -> IDLE
```

首版限制：

```text
max_active_sessions: 1
morpheus_max_concurrency: 1
```

协议和内部对象仍完整保留 session/turn 标识，后续提高并发时不修改协议。

## 13. 取消与过期结果

- `client.turn.cancel` 显式取消当前 turn。
- 在 THINKING 或 SPEAKING 收到新的 `client.audio.start` 时，先取消旧 turn，再开始新 turn。
- asyncio task 立即 cancel。
- Piper/Morpheus 子进程先 terminate；2 秒宽限期后仍运行则 kill。
- 取消事件不等待旧子进程退出即可发送。
- 每次推送事件、写产物索引和发布 latest 前都校验 turn 是否仍是当前有效 turn。
- 无法终止的旧计算可以结束，但结果必须丢弃。
- 被取消 turn 保留 timeline 和诊断产物，不更新 latest。
- 同一 turn 只能发出 `server.pipeline.done`、`server.pipeline.error`、`server.turn.cancelled` 中的一个终态事件。

## 14. 产物与 Latest

```text
data/runs/{session_id}/{turn_id}/
  input.wav
  asr.json
  llm.json
  tts/
  face/
  ue5/
  timeline.json
```

```text
data/latest/latest_pipeline.json
data/latest/latest_ue5_blendshape.json
```

latest 使用临时文件加原子替换发布。失败、取消、过期或输出校验失败的 turn 不能覆盖 latest。

首版不自动删除历史结果。配置保留 `retention.max_runs: 100`，后续增加显式清理命令或后台任务。

## 15. Timeline 与延迟

每个阶段记录：

```text
started_at
finished_at
duration_ms
status
provider
error_code
```

UTC 时间戳用于跨系统观察，单调时钟用于计算 duration。

流式链路额外记录：

```text
audio_end -> asr_final
audio_end -> llm_first_token
audio_end -> first_tts_ready
audio_end -> first_face_ready
audio_end -> first_segment_ready
total_turn_duration
```

阶段二目标：

```text
audio_end -> llm_first_token: P50 <= 0.8s, P90 <= 1.5s
audio_end -> first_tts_ready: P50 <= 1.5s, P90 <= 2.5s
audio_end -> first_face_ready: P50 <= 2.0s, P90 <= 3.5s
```

这些指标在真实环境记录和评估，不作为 Mock 单元测试的墙钟断言。

## 16. 健康检查与诊断

```text
GET /health
GET /diagnostics
GET /diagnostics/{adapter}
```

- `/health` 只检查 Web 服务自身是否存活。
- `/diagnostics` 汇总当前 provider 配置、可用性和探测耗时。
- Mock diagnostics 稳定成功。
- faster-whisper 检查库、模型配置和加载能力。
- Ollama 检查 HTTP 可达性和目标模型。
- Piper 检查可执行文件、模型文件和输出目录。
- Morpheus 检查 Conda、环境、项目路径、命令配置和输出目录。
- diagnostics 默认不运行昂贵的完整推理。

## 17. 错误模型

统一错误码：

```text
invalid_request
invalid_audio_format
no_speech_detected
session_limit_reached
protocol_violation
provider_unavailable
provider_timeout
provider_failed
output_validation_failed
turn_cancelled
internal_error
```

错误结果包含 `code`、`stage`、`provider`、`retryable` 和 `message`。

HTTP 和 WebSocket 只返回安全错误信息，不泄漏命令、堆栈或敏感本地路径。详细异常写入服务日志和 turn 诊断。

单个 TTS 或 Morpheus 片段失败即终止当前 turn。首版不自动重试，以避免重复播放和顺序破坏。

## 18. 测试设计

普通 `pytest` 默认全 Mock，不依赖 GPU、Conda、Ollama、Piper、Morpheus 或真实音频。

### 18.1 单元测试

- 配置加载、校验和 provider registry。
- 音频格式验证、统计和静音检测。
- LLM 情绪结果解析。
- 句子切分和无标点强制切分。
- 状态机合法与非法迁移。
- timeline 记录。
- UE5 52 维格式和 frame chunk 拆分。
- 运行目录与 latest 原子发布。
- Mock 延迟、失败、超时、取消和迟到结果。

### 18.2 HTTP 集成测试

- health 与 diagnostics。
- `/pipeline/audio` 成功、静音、失败和超时。
- 输出校验失败。
- pipeline/latest 与 ue5/latest。
- 失败 turn 不覆盖 latest。

### 18.3 WebSocket 集成测试

- 正常事件顺序。
- JSON 与 binary 严格配对。
- 主动结束、静音自动结束与最大时长结束。
- 显式取消和新 turn 打断旧 turn。
- 旧输出和 latest 丢弃。
- provider 失败和超时。
- sequence 单调递增。
- 每个 turn 终态事件唯一。

### 18.4 真实冒烟测试

真实测试使用 `integration` 标记，默认普通测试不执行：

- faster-whisper。
- Ollama streaming。
- Piper CLI。
- Morpheus CLI。
- 完整 `/pipeline/audio`。
- 伪流式 `/pipeline/stream`。

真实验收需要至少一段中文 WAV、一份正常 `[N, 52]` 输出样例和一份 UE5 JSON 样例。

## 19. WebSocket 测试客户端

首版提供测试客户端，用于：

- 发送开始、音频元数据、PCM binary、结束和取消事件。
- 校验 JSON/binary 配对和 sequence。
- 保存收到的 WAV。
- 保存并校验 52 维 frame chunk。
- 展示状态、ASR、LLM 文本、错误和延迟。
- 在收到 cancel 时清空本地待播放队列。

真实 UE5 蓝图接入不属于本轮首版，但协议和数据可直接供后续接入。

## 20. 实施优先级

### P0

```text
Git 与项目骨架
配置与领域模型
Mock provider 与自动化测试
/pipeline/audio
timeline 与 diagnostics
WebSocket 协议
session/turn 状态机与 cancel
```

### P1

```text
全 Mock /pipeline/stream
Ollama streaming
faster-whisper
Piper 句子级 TTS
Morpheus 句子级处理
WebSocket 测试客户端
真实 provider 冒烟测试
```

### P2

```text
真实 UE5 工程接入
Morpheus 到 UE5 曲线映射
多会话并发
真流式 ASR/TTS
实时 FaceDriver
```

## 21. 明确排除项

首版明确不做：

- Redis、数据库或外部消息队列。
- 多进程 worker 下的共享会话状态。
- 真流式 ASR partial。
- Piper 裸 PCM chunk streaming。
- Morpheus 实时增量推理。
- ARKit/MetaHuman 命名或映射承诺。
- 多用户并发性能优化。
- 自动历史清理。
- Windows 兼容。
- Docker 或其他容器化。
- 公网部署、鉴权和 HTTPS。
