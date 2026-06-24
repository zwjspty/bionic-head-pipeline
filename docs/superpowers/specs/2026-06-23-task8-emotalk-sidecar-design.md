# Task 8: EmoTalk Sidecar 常驻化设计

## 背景

Task 7 已经把 stream 调度改成“先发 TTS chunk，再在后台做 Face / UE5”，真实链路中第二个 TTS chunk 不再等待第一个 Face chunk。但 benchmark 仍然显示 Face / Audio2Face 是最大瓶颈：真实 EmoTalk 每段大约 8–9 秒，明显慢于首音频和后续 TTS。

当前 `audio2face.provider = "emotalk"` 的实现复用 Morpheus 外部命令 provider。每个 TTS chunk 都会执行：

```text
conda run -n emotalk python /home/user/code/EmoTalk_release/scripts/export_blendshape_from_audio.py
```

而本机 `export_blendshape_from_audio.py` 的 `main()` 每次都会：

```text
创建 EmoTalk(args)
加载 EmoTalk.pth
初始化并加载两个 Wav2Vec 分支
model.to(device)
model.eval()
librosa.load(wav)
model.predict(audio)
np.save(out_path)
```

这意味着 8–9 秒里有大量重复启动和模型加载成本。Task 8 的核心目标是把 EmoTalk 变成常驻 sidecar：模型只加载一次，主服务每段音频只发一次本地推理请求。

本机环境事实：

- 主服务运行在项目 `.venv`，Python 3.10/3.11 兼容当前测试环境。
- EmoTalk 运行在 Conda env `emotalk`，Python 3.8.8。
- `emotalk` env 已有 `librosa`、`soundfile`、`scipy`、`numpy`、`torch`。
- `emotalk` env 没有 `fastapi`、`uvicorn`、`httpx`。
- 真实 Piper 中文 voice 当前输出 WAV 为 `22050 Hz`、mono、16-bit PCM；EmoTalk 期望 16k 音频。

## 目标

1. 新增 `emotalk-sidecar` Audio2Face provider，保留现有 `emotalk` 命令行 provider 作为 fallback。
2. 新增一个可在 `emotalk` Conda env 内运行的 sidecar 进程，启动时加载 EmoTalk 一次，并可连续处理多个短音频请求。
3. 使用本机二进制协议传输，避免 base64 JSON；第一版可输入 WAV bytes，由 sidecar 在常驻进程内解码和重采样。
4. sidecar 返回 raw float32 `[N, 52]` frames，并附带细粒度 timings，便于确认剩余瓶颈在解码、重采样、Wav2Vec、decoder 还是序列化。
5. 主服务保持现有外部 API、WebSocket 事件协议、UE5 formatter、灰模预览脚本不变。
6. 跑真实 stream benchmark，对比 Task 7 baseline 的 `face_first_chunk_ms`、`e2e_first_visible_face_ms`、`total_turn_duration_ms`。

## 非目标

Task 8 不做以下内容：

- 不把 EmoTalk 改成真正 streaming 模型。
- 不做 student FaceDriver。
- 不做 ONNX / 量化 / torch.compile。
- 不默认把 TTS 输出改成 16k raw PCM。
- 不实现 Unix Domain Socket；协议设计要允许后续从 HTTP localhost 平滑替换为 UDS。
- 不实现 200–300ms overlap、6–9 帧 crossfade、session 级 blink scheduler。这些作为 Task 9。
- 不接真实 UE5 蓝图；继续使用现有 `morpheus_52_raw` formatter 和 EmoTalk 灰模预览。

## 方案选择

### 方案 A：继续优化每段 subprocess

每段继续 `conda run`，只微调命令参数或 torch 设置。实现最小，但无法消除 Python 启动、import、模型加载、Wav2Vec 初始化等重复成本。这个方案不能解决结构性瓶颈。

### 方案 B：FastAPI / Uvicorn sidecar

主服务通过 HTTP 调 FastAPI sidecar。接口清晰，但本机 `emotalk` env 没有 FastAPI / Uvicorn，而且 Python 3.8.8 环境较旧。为 sidecar 先改 Conda 依赖会引入额外环境风险。

### 方案 C：标准库 HTTP sidecar + 二进制 framing

sidecar 使用 Python 标准库 `http.server` 或等价标准库组件实现 `/health` 和 `/infer`。它只依赖 EmoTalk env 已有的 `torch`、`numpy`、`librosa`、`soundfile`。请求和响应使用 length-prefixed JSON header + raw bytes，避免 base64 和大 JSON frames。

推荐方案 C。它保留进程隔离，不污染 EmoTalk env，开发和测试成本低；协议又足够接近后续 UDS / stdin 长连接 / shared memory 的形态。

## 架构

```text
主 FastAPI 服务（.venv）
  └── Audio2Face provider: emotalk-sidecar
        ├── diagnostics: GET http://127.0.0.1:8013/health
        └── drive(): POST http://127.0.0.1:8013/infer
                request: header + WAV bytes
                response: header + float32 frames bytes

EmoTalk sidecar（conda env emotalk）
  ├── 启动时设置 torch threads
  ├── 插入 /home/user/code/EmoTalk_release 到 sys.path
  ├── 创建 args
  ├── load EmoTalk 模型一次
  ├── warmup 1s 静音
  └── 串行处理推理请求
```

现有 provider 名保持：

- `emotalk`：旧命令行 provider，每段外部命令，fallback。
- `emotalk-sidecar`：新常驻 provider。

## 配置

新增配置模型：

```json
{
  "adapters": {
    "audio2face": {
      "provider": "emotalk-sidecar",
      "timeout_seconds": 120
    }
  },
  "providers": {
    "emotalk_sidecar": {
      "base_url": "http://127.0.0.1:8013",
      "timeout_seconds": 120,
      "output_npy_name": "emotalk.npy",
      "fps": 30,
      "level": 1,
      "person": 0
    }
  }
}
```

新增示例配置文件：

```text
config/emotalk-sidecar.example.json
```

它应基于当前 `config/emotalk.example.json`，只把 `audio2face.provider` 切到 `emotalk-sidecar` 并添加 `providers.emotalk_sidecar`。

sidecar 启动命令示例：

```bash
/home/user/miniconda3/bin/conda run -n emotalk python scripts/emotalk_sidecar.py \
  --host 127.0.0.1 \
  --port 8013 \
  --emotalk-root /home/user/code/EmoTalk_release \
  --model-path /home/user/code/EmoTalk_release/pretrain_model/EmoTalk.pth \
  --device cpu \
  --torch-threads 4 \
  --torch-interop-threads 1 \
  --warmup-seconds 1.0
```

## 二进制协议

Task 8 定义 `bionic-head-emotalk-sidecar-v1`。

每个二进制消息由三部分组成：

```text
uint32 big-endian header_length
UTF-8 JSON header
raw body bytes
```

### `/infer` 请求 header

```json
{
  "protocol": "bionic-head-emotalk-sidecar-v1",
  "request_id": "uuid-or-turn-and-call-index",
  "session_id": "uuid",
  "turn_id": "uuid",
  "input_format": "wav",
  "byte_length": 12345,
  "fps": 30,
  "level": 1,
  "person": 0,
  "emotion": "friendly",
  "intensity": 0.8
}
```

请求 body 是完整 WAV bytes。第一版保留 WAV 输入，是因为当前真实 Piper 输出是 22050 Hz WAV；把 TTS provider 统一改成 16k raw PCM 是后续任务。

### `/infer` 成功响应 header

```json
{
  "protocol": "bionic-head-emotalk-sidecar-v1",
  "ok": true,
  "request_id": "uuid-or-turn-and-call-index",
  "dtype": "float32",
  "shape": [128, 52],
  "fps": 30,
  "channel_count": 52,
  "timings_ms": {
    "decode": 1.0,
    "resample": 2.0,
    "tensor": 0.5,
    "predict": 900.0,
    "serialize": 1.0,
    "total": 904.5
  }
}
```

响应 body 是 row-major `float32[N, 52]`。

### `/infer` 失败响应 header

```json
{
  "protocol": "bionic-head-emotalk-sidecar-v1",
  "ok": false,
  "request_id": "uuid-or-turn-and-call-index",
  "error_code": "provider_failed",
  "message": "EmoTalk inference failed"
}
```

失败响应 body 为空。

### `/health`

`GET /health` 返回 JSON：

```json
{
  "ok": true,
  "loaded": true,
  "provider": "emotalk-sidecar",
  "device": "cpu",
  "model_path": "/home/user/code/EmoTalk_release/pretrain_model/EmoTalk.pth",
  "torch_threads": 4,
  "torch_interop_threads": 1
}
```

## Sidecar 行为

启动时：

1. 设置 `torch.set_num_threads(torch_threads)`。
2. 设置 `torch.set_num_interop_threads(torch_interop_threads)`。
3. 插入 `emotalk_root` 到 `sys.path`。
4. 构造与 `export_blendshape_from_audio.py` 兼容的 args。
5. 创建 `EmoTalk(args)`。
6. 加载 `EmoTalk.pth`。
7. `model.to(device)`。
8. `model.eval()`。
9. 用 1 秒 16k 静音做 warmup。

每次 `/infer`：

1. 读取 request header 和 WAV bytes。
2. 解码为 mono float32。
3. 如果 sample rate 不是 16000，重采样到 16000。
4. 转为 `torch.FloatTensor(samples).unsqueeze(0).to(device)`。
5. 构造 `level` 和 `person` tensor。
6. 在 `torch.inference_mode()` 内调用 `model.predict(audio, level, person)`。
7. 验证输出是二维 `[N, 52]`、非空、finite。
8. 转成 contiguous float32。
9. 返回二进制 frames 和 timings。

Sidecar 第一版串行处理推理。即使标准库 HTTP server 支持并发，也必须用进程内 lock 保护模型推理，避免多个请求同时进 PyTorch 模型。

## 主服务 provider 行为

新增文件：

```text
src/bionic_head/adapters/emotalk_sidecar.py
```

核心行为：

1. `drive(audio, emotion, intensity, context)` 读取 `audio.path` 的 WAV bytes。
2. 创建 `face/emotalk_sidecar_XXXX/` 输出目录。
3. 用二进制协议 POST 到 sidecar `/infer`。
4. 验证响应：
   - `ok == true`
   - `dtype == "float32"`
   - `shape == [N, 52]`
   - body 长度等于 `N * 52 * 4`
   - 所有值 finite
5. 写入：

```text
face/emotalk_sidecar_XXXX/emotalk.npy
face/emotalk_sidecar_XXXX/meta.json
```

6. 返回 `FaceArtifact`，`path` 指向 `.npy`，`auxiliary_paths` 包含 `meta.json`。

错误映射：

- 缺少 `httpx`：`PROVIDER_UNAVAILABLE`。
- sidecar 连接失败：`PROVIDER_UNAVAILABLE`，retryable。
- 请求超时：`PROVIDER_TIMEOUT`，retryable。
- sidecar 返回 `ok=false`：`PROVIDER_FAILED`，retryable。
- 响应 shape / dtype / body 长度 / NaN 不合法：`OUTPUT_VALIDATION_FAILED`，不可重试。
- 本地 WAV 文件缺失或为空：`OUTPUT_VALIDATION_FAILED`，不可重试。

取消策略：

- provider 发请求前后都检查 `context.cancellation`。
- 如果主服务 task 被取消，HTTP request coroutine 也取消。
- Task 8 不保证立即中断 sidecar 内正在运行的 PyTorch 前向；旧 turn 的结果仍由现有 `turn_id` / stale-drop 机制丢弃。
- 因为首版仍是单活跃 session，sidecar 串行阻塞是可接受限制；后续要用 generation epoch、队列长度和 sidecar worker restart 进一步优化。

## 协议 helper

新增一个纯标准库、Python 3.8 兼容的协议 helper：

```text
src/bionic_head/sidecar_protocol.py
```

这个文件不能依赖 Pydantic、FastAPI、httpx，也不能使用 Python 3.10+ 语法。主服务 provider 和 `scripts/emotalk_sidecar.py` 都使用它，避免协议打包/解包逻辑分叉。

## 测试策略

自动化测试默认不依赖真实 EmoTalk。

单元测试覆盖：

1. `sidecar_protocol` 能编码/解码 header + body。
2. `EmoTalkSidecarAudio2FaceAdapter` 能从 fake HTTP transport 读取合法 float32 `[N, 52]` 并写 `.npy` / `meta.json`。
3. adapter 拒绝错误 dtype。
4. adapter 拒绝错误 shape。
5. adapter 拒绝 body 长度不匹配。
6. adapter 拒绝 NaN / Inf。
7. adapter 将 timeout 映射为 `PROVIDER_TIMEOUT`。
8. adapter 将连接失败映射为 `PROVIDER_UNAVAILABLE`。
9. registry 能构建 `audio2face.provider = "emotalk-sidecar"`。
10. `config/emotalk-sidecar.example.json` 能加载。
11. sidecar 参数解析和 health payload 不依赖 FastAPI。

真实验收测试手动执行：

1. 启动 sidecar。
2. `curl http://127.0.0.1:8013/health` 确认 loaded。
3. 启动主服务：

```bash
PYTHONPATH=src BIONIC_CONFIG=config/emotalk-sidecar.example.json \
  .venv/bin/uvicorn bionic_head.api.app:create_app \
  --factory --host 127.0.0.1 --port 8005
```

4. 跑 stream client。
5. 确认输出：

```text
server.tts.audio
server.face.frames
server.ue5.frames
server.pipeline.done
```

6. 渲染 EmoTalk 灰模视频。
7. 跑 benchmark，与 Task 7 baseline 对比。

## Benchmark 口径

Task 8 完成后至少跑两组真实 stream benchmark：

```text
baseline: audio2face.provider = emotalk
candidate: audio2face.provider = emotalk-sidecar
```

重点指标：

```text
face_first_chunk_ms
e2e_first_visible_face_ms
total_turn_duration_ms
sidecar timings_ms.predict
sidecar timings_ms.total
```

接受标准：

- `emotalk-sidecar` 能稳定输出 finite `[N, 52]`。
- `/diagnostics` 显示 `audio2face.provider = emotalk-sidecar` 可用。
- 真实 stream 能完成 `server.pipeline.done`。
- 如果 sidecar 后 `face_first_chunk_ms` 仍接近 8–9 秒，timings 必须能解释瓶颈主要落在 `predict` 而不是启动/加载/IO。
- 如果 sidecar 后明显下降，记录 p50/p90 对比，并把结果写入交付说明。

## 后续任务

Task 9：短音频分段质量优化。

```text
1.0s face subchunk
200–300ms left context
6–9 帧 crossfade
session 级 blink scheduler
更细的 frame pts / segment metadata
```

Task 10：进一步降低 IPC 和模型推理成本。

```text
raw 16k PCM 输入
Unix Domain Socket 或 stdin/stdout 长连接
shared_memory 评估
torch thread sweep benchmark
ONNX / quantization 实验
student FaceDriver 数据生成
```

## 自查结论

- 本设计聚焦一个独立任务：常驻化 EmoTalk，新增 provider 和 sidecar。
- 现有外部 API、stream 协议、UE5 payload 不改变。
- 旧 `emotalk` provider 保留，不会失去 fallback。
- WAV 输入是 Task 8 的明确取舍；raw PCM 被拆到 Task 10。
- overlap / crossfade / blink 被拆到 Task 9，避免本任务同时修改推理架构和动画后处理。
- 没有未定的 provider 名、配置字段、协议版本或验收指标。
