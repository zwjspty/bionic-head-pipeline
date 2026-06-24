# 2026-06-24 当前状态与下一阶段路线

## 总体判断

项目目前已经完成一个真实可运行、可观测、低延迟、具备最小打断能力的准实时数字人原型。

更准确地说：

```text
已完成：阶段四“句子级增量”与 Face 连续性优化。
部分完成：阶段三 barge-in 控制面闭环。
尚未完成：真实客户端播放、AEC、执行中模型取消可靠性、产品级全双工。
```

当前阶段名称：

```text
实验室级准实时数字人 MVP
—— 正在从“管线能跑且较快”
进入“真实交互正确性与体验收口”
```

## 当前能力状态

| 能力 | 当前状态 | 判断 |
| --- | --- | --- |
| 稳定 HTTP / WebSocket、Mock / Real provider | 已完成 | 工程基线稳定 |
| 延迟观测、benchmark | 已完成 | 指标较完整 |
| Ollama、Piper、EmoTalk 常驻 | 已完成 | 固定冷启动基本解决 |
| LLM → 句子 TTS → 后台 Face | 已完成 | 有效准实时管线 |
| generation_epoch / stale-drop | 已完成 | 旧结果不会正常发出 |
| RMS barge-in / playback.stop | 最小实现完成 | 缺真实播放/AEC验收 |
| Face crossfade | 已完成 | 真实多段样本有效 |
| Eye continuity | 框架完成 | 默认配置 no-op |
| 多轮对话记忆 | 未实现 | 每个 turn 仍独立问答 |
| chunked / partial ASR | 未实现 | ASR 仍在端点后整段识别 |
| 真实 UE5 | 未实现 | 当前只是 raw 52 维协议 |
| Student FaceDriver | 未开始 | EmoTalk 仍是最终推理瓶颈 |

## 真实 benchmark 基线

Task 8+9+10 之后的真实 stream 基线：

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

- Face 热路径已经从旧方式 8–16s 降到约 0.4–0.5s。
- TTS 音频先出来，Face 首帧晚约 0.48s。
- 当前 benchmark 测到的是网络事件时间，不是真实扬声器/渲染时间。
- 下一步更重要的是取消正确性、真实音画同步和多轮记忆，而不是盲目继续压模型耗时。

## 已完成的关键工程升级

### 1. 模块边界清晰

ASR、LLM、TTS、Audio2Face、UE5 formatter 都通过 provider / registry 组合。编排层不依赖具体厂商格式，后续替换 CosyVoice、Morpheus 或 Student FaceDriver 成本可控。

### 2. EmoTalk sidecar 路线成功

从：

```text
每段 conda run / Python 启动 / checkpoint load / predict
```

升级为：

```text
startup prewarm
+ persistent stdin/stdout worker
+ binary protocol
+ warm predict
```

结果：

```text
旧方式：8–16s
sidecar cold：7–11s，但移到 startup/prewarm
sidecar warm：370–520ms
```

### 3. stale guard 设计扎实

`TurnHandle` 保护：

```text
current
cancellation
terminal_event
generation_epoch
```

旧 turn 的 emit 和 latest commit 均受保护。

### 4. 测试资产完整

测试覆盖：

```text
sidecar 协议截断和短读
sidecar 进程复用
prewarm
进程退出
错误响应
timeout
旧 turn stale
Face 乱序完成
crossfade
eye continuity
真实 provider smoke
```

## 当前最高优先级风险

### P0-1：取消正在进行的 sidecar 请求可能导致通信错位

风险路径：

```text
主服务写入旧 turn 请求
-> EmoTalk worker 阻塞 model.predict
-> 用户打断
-> asyncio face task 被 cancel
-> 主服务不再等待旧 response
-> worker 后续把旧 response 写到 stdout
-> 新 turn 可能读到旧 response
```

推荐修法：

```text
sidecar request pump
-> pump 独占 stdin/stdout
-> 调用方取消后 Future 作废
-> pump 继续读完旧 response 并丢弃
-> 通信流保持对齐
-> 下一请求继续复用 warm worker
```

最低回归测试：

```text
启动一个慢 sidecar request
-> request 已写入后取消
-> 立刻提交新 turn request
-> 新请求必须成功
-> 不得读到旧 turn response
-> 不得出现 protocol mismatch
```

### P0-2：Face 后处理状态依赖任务完成顺序

当前多个 Face task 共用同一个 `FaceSegmentStitcher` 和 `EyeContinuityProcessor`。如果未来多个 sidecar worker / GPU 并发 / Student FaceDriver batch 导致 segment 2 先于 segment 1 返回，状态会错误。

推荐修法：

```text
并行或异步推理
-> 按 segment_index 缓存结果
-> OrderedFaceSequencer 顺序释放
-> stitch
-> eye continuity
-> UE5 format
-> emit
```

### P0-3：当前测到的是网络事件延迟，还不是真实音画同步

当前客户端只写 WAV 和 JSON 文件，不真实播放或渲染，因此还没有验证：

```text
什么时候真正听到声音
什么时候真正显示第一帧
播放缓存是否被清空
playback.stop 后多久真正静音
音频和嘴型的实际偏移
```

第一版可先选择：

```text
收到 TTS 后先缓存
-> 等匹配第一批 Face frames
-> 音频和 Face 同时开始
```

这会增加首次可听延迟，但会显著改善音画同步。

### P0-4：网络发送期间取消可能被阻塞

`TurnHandle.emit_if_current()` 当前在内部锁中等待发送操作。大 JSON 发送期间，`cancel()` 也需要同一把锁，可能延迟 `server.playback.stop`。

推荐修法：

```text
connection 级出站队列
高优先级：server.playback.stop / terminal
普通优先级：TTS / Face / UE5
发送前重新检查 generation_epoch
不要在 turn 状态锁内等待网络 I/O
```

## Task 10 状态校准

Task 10 更准确地说是：

```text
Eye continuity framework：完成
真实眼部通道映射：未完成
真实 blink 视觉验收：未完成
```

原因：

- 默认配置为 no-op：

```json
"eye_smooth_channel_indices": [],
"blink_enabled": false,
"blink_channel_indices": []
```

- 当前 processor 在每次 `StreamOrchestrator.run()` 里创建；因为每个 turn 都会新建 orchestrator，所以 blink 状态实际仍是 per-turn，不是产品级 session-level。

后续建议拆分：

```text
Eye smoothing state：每个 turn 独立
Blink scheduler state：每个 session 保留
```

由 `StreamConnection` 持有 `SessionFaceState`，再注入新的 orchestrator。

## 其他功能缺口

### 多轮对话记忆

当前 LLM 调用仍是：

```python
chat_stream(asr_text, [], context)
chat(asr_text, [], context)
```

因此第二轮“刚才那个问题”无法引用历史。

### 情绪尚未真正驱动声音和脸

LLM 生成 `emotion` / `intensity`，但当前真实 TTS 和 EmoTalk 没有可靠使用它们改变韵律或表情。

### Barge-in 仍是 RMS 实验版

RMS 无法可靠区分用户讲话、扬声器回声、背景噪声和敲击声。`interrupt_to_playback_stop_ms` 的语义也需要改成真实用户插话起点到真正静音。

### ASR 仍是 endpoint 后整段识别

faster-whisper 仍是音频结束后一次性转录，没有 partial/stable/final 层级。

### UE5 仍是中间格式

通道名仍是：

```text
morpheus_00 ... morpheus_51
```

没有 ARKit、MetaHuman 或 UE5 曲线映射。

## 推荐下一阶段路线

### Task 11：取消正确性与有序 Face 管线

范围：

```text
1. sidecar request pump / drain-and-discard
2. 取消期间通信不失步
3. 去掉或协调外层/内层重复 timeout
4. provider cancellation 统一语义
5. Face result 按 segment_index 顺序后处理和发送
6. playback.stop 使用高优先级出站通道
```

验收：

```text
取消正在 model.predict 的旧 turn
新 turn 立即进入 ASR/LLM/TTS
下一次 Face 请求成功
无 response ID mismatch
无旧 Face/UE5 输出
连续执行 50 次 interrupt 不失步
```

### Task 12：真实播放客户端与音画同步

范围：

```text
麦克风持续采集
TTS 播放队列
Face/UE5 播放队列
server.playback.stop 清空缓冲
audio/face 共用 PTS
浏览器或桌面端 AEC
真实 interrupt 时间戳
```

验收：

```text
interrupt 到真正静音 p90 < 200ms
音频与嘴型偏移 p90 控制在 ±100ms 左右
old turn audio/face leak = 0
```

### Task 13：session 多轮记忆

加入：

```text
ConversationHistory
成功 turn 原子提交
取消 turn 不提交 assistant 回复
最大轮数/token 限制
client.session.reset
history benchmark
```

### Task 14：确认 52 维通道映射并激活视觉能力

确认 EmoTalk / Morpheus 52 维顺序后，再启用：

```text
eyeBlinkLeft / eyeBlinkRight
mouth / jaw / brow / cheek channel groups
session-level blink
嘴部 3–5 帧 crossfade
表情 6–9 帧 crossfade
```

### Task 15：AEC、VAD 与 partial ASR

顺序：

```text
客户端 AEC
-> 客户端 speech_start
-> 服务端 RMS fallback
-> Silero VAD / endpointing
-> unstable_partial
-> stable_partial
-> final
```

### Task 16：真实 UE5 接入

需要：

```text
52 维 mapping table
全局 frame PTS
segment buffer
audio/face 同步启动
playback.stop / flush
generation_epoch drop
UE5 ack / buffer depth
重连处理
```

### Task 17：Student FaceDriver

仅当真实播放仍认为 470–520ms Face 延迟不可接受时再开始。

第一版 student 建议：

```text
audio chunk -> mouth/jaw 关键通道
```

眼睛和 blink 可继续规则化。

## 小型工程清理池

这些不应阻塞 Task 11，但适合后续穿插处理：

```text
1. 合并两个 SessionManager 实现。
2. 实现或删除未实际执行的 retention.max_runs、morpheus_max_concurrency、face_stitching.reset_on_new_turn。
3. 合并 sidecar meta.json 重复写入。
4. 拆分 /health/live 和 /health/ready。
5. config/emotalk.example.json 的绝对路径改成环境变量或 local.json 示例。
6. Ollama adapter 复用 httpx.AsyncClient 连接池。
7. 生产模式可关闭 server.face.frames debug JSON，或改成 float32 binary。
8. 增加 Ruff、类型检查、coverage。
```
