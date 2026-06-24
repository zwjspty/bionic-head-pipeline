# 项目说明 goal：中文端到端数字人低延迟可打断系统

## 0. 2026-06-24 当前状态校准

当前项目已经完成“实验室级准实时数字人 MVP”：

```text
真实 HTTP /pipeline/audio 可运行
真实 WebSocket /pipeline/stream 可运行
faster-whisper / Ollama / Piper / EmoTalk sidecar / morpheus_52_raw 已形成真实链路
EmoTalk 热路径约 0.37–0.52s
stream 具备 benchmark、generation_epoch、stale-drop、face crossfade、eye continuity 框架
```

当前更准确的判断：

```text
已完成:
  阶段四“句子级增量”
  Face 连续性优化
  sidecar 常驻化和预热

部分完成:
  RMS barge-in / playback.stop 控制面
  Eye continuity 框架

尚未完成:
  产品级全双工
  真实播放客户端 / AEC
  多轮对话记忆
  真实 UE5 曲线映射
  partial ASR
  session-level blink 视觉验收
```

最新真实 stream benchmark 基线：

```text
10/10 success
tts_first_audio_ms p50        ≈ 583ms
e2e_first_visible_face_ms p50 ≈ 1062ms
face_total_ms p50             ≈ 469ms
face_total_ms p90             ≈ 519ms
old_turn_face_leak_count      = 0
stale_face_drop_count         = 0
```

下一阶段 P0 不再是盲目继续压模型耗时，而是：

```text
Task 11:
  取消正确性
  sidecar request pump / drain-and-discard
  有序 Face 后处理与发送
  高优先级 playback.stop 出站通道
```

完整当前状态见：

```text
docs/status/2026-06-24-current-state.md
```

## 1. 项目目标

本项目目标是在现有仿生人头链路基础上，构建一个支持中文语音交互、情绪表达、面部 blendshape 生成、灰模预览和未来 UE5 实时驱动的模块化端到端系统。

系统外部体验应接近：

```text
用户说话
  -> 数字人理解
  -> 数字人快速开口回复
  -> 语音、口型、表情同步
  -> 用户可以随时插话打断
```

系统内部仍保持模块化：

```text
ASR
LLM
TTS
Audio2Face
UE5 formatter
Preview renderer
```

本项目不追求短期内把所有模块合成一个不可解释的大模型。近期目标是“外部端到端、内部模块化”：用户感知到连续交互，工程上仍能独立调试、替换、评测和蒸馏各个模块。

## 2. 当前已完成基础

当前已经完成：

```text
Mock provider 全链路
真实 HTTP /pipeline/audio
真实 WebSocket /pipeline/stream
faster-whisper ASR
Ollama qwen2.5:3b LLM
Piper 中文 TTS
EmoTalk Audio2Face
morpheus_52_raw UE5 formatter
EmoTalk Blender 3D 灰模头预览
```

当前本机真实链路：

```text
用户 wav
  -> faster-whisper
  -> Ollama / qwen2.5:3b
  -> Piper / zh_CN-huayan-medium
  -> EmoTalk 输出 [N, 52]
  -> morpheus_52_raw JSON
  -> Blender grey-head MP4
```

当前已知边界：

```text
真正 Morpheus:
  目标服务器尚未确认 lyyMor / Morpheus-Software 命令。

真正 UE5:
  工程和曲线映射尚未接入。

灰模头:
  UE5 完成前的正式视觉验收方式。
```

## 3. 最新调研后的核心决策

基于低延迟与全双工调研，当前最应该做的不是立刻追求电话级真全双工，而是先把系统改造成：

```text
常驻模型
+ 句子级增量
+ turn 级可取消
+ stale output 丢弃
+ VAD / barge-in
+ 客户端 AEC
```

短期优化优先级：

```text
1. 建立延迟 benchmark。
2. Ollama 预热和 keep_alive。
3. Piper 常驻化，避免每句 CLI 子进程重复加载模型。
4. WebSocket 出站事件带 generation_epoch。
5. 用户插话时立即 server.playback.stop。
6. 旧 turn 的音频、脸部帧、latest 写入全部丢弃。
```

暂缓事项：

```text
1. 暂不硬改 EmoTalk 为真 streaming。
2. 暂不让 ASR partial 直接驱动 LLM。
3. 暂不把服务端作为主要 echo cancellation 实现。
4. 暂不把 UE5 未确认工程说成已接入。
```

## 4. 主要接口

系统保留两个核心接口。

### 4.1 POST /pipeline/audio

用途：

```text
离线完整链路测试
真实 provider 验收
数据生成
回归测试
teacher 数据采集
```

### 4.2 WS /pipeline/stream

用途：

```text
实时语音交互
音频 chunk 输入
LLM token / segment 输出
TTS audio chunk 输出
UE5 frame chunk 输出
可打断对话
```

## 5. Provider 架构

所有 provider 通过 JSON 配置切换。

```text
ASR provider
LLM provider
TTS provider
Audio2Face provider
UE5 formatter provider
Preview renderer script
```

Mock provider 用于：

```text
自动化测试
协议验证
状态机验证
cancel / interrupt / stale_drop 验证
失败、超时、慢响应验证
```

真实 provider 用于：

```text
部署验收
效果验证
真实延迟 benchmark
灰模头或 UE5 视觉验收
```

## 6. WebSocket 交互目标

近期目标不是所有模块真流式，而是实现可感知的准实时：

```text
客户端持续上传 PCM chunk
服务端 endpoint 后 ASR final
LLM streaming 输出 token
sentence segmenter 切出第一句
TTS 生成第一句 wav
Audio2Face 生成该句 face frames
客户端播放第一句
后续句子继续追加
```

中期目标加入 barge-in：

```text
bot speaking 时用户开口
  -> server.turn.interrupting
  -> server.playback.stop
  -> cancel 旧 turn
  -> generation_epoch 增加
  -> 旧结果 stale_drop
  -> 处理新 turn
```

## 7. 状态机目标

推荐状态：

```text
idle
listening
thinking
speaking
interrupt_pending
cancelled
recovering
```

关键原则：

```text
用户插话时，先停播放，再处理识别。
能 cancel 的 asyncio task 立即 cancel。
Piper / EmoTalk / Morpheus 子进程尽量 terminate / kill。
无法及时终止的旧结果按 turn_id / generation_epoch 丢弃。
旧 turn 不再写 latest。
旧 turn 不再推给客户端或 UE5。
```

## 8. 事件协议目标

现有事件保留：

```text
client.session.start
client.audio.start
client.audio.chunk
client.audio.end

server.session.ready
server.asr.partial
server.asr.final
server.llm.delta
server.segment.ready
server.tts.audio
server.ue5.frames
server.pipeline.done
server.pipeline.error
server.turn.cancelled
```

新增或细化：

```text
client.audio.vad.start
client.audio.vad.stop
client.playback.state
client.turn.interrupt

server.turn.started
server.turn.interrupting
server.playback.stop
server.turn.stale_drop
server.segment.accepted
server.segment.skipped
```

所有出站事件必须携带：

```text
session_id
turn_id
generation_epoch
event_id
timestamp
```

## 9. 低延迟指标

项目必须持续记录：

```text
asr_ms
llm_ttft_ms
llm_total_ms
tts_first_audio_ms
tts_total_ms
face_first_chunk_ms
face_total_ms
e2e_first_audible_ms
e2e_first_visible_face_ms
interrupt_to_playback_stop_ms
stale_drop_count
old_turn_leak_count
```

验收不能只说“感觉快了”，必须有 benchmark 对比。

## 10. TTS 路线

短期：

```text
继续使用 Piper 中文音色。
把 Piper 从 CLI per sentence 改成常驻 provider。
优先使用 Piper Python API 或 Piper HTTP sidecar。
```

中期 A/B：

```text
CosyVoice2 / CosyVoice3 / LightTTS
MeloTTS
Kokoro
ChatTTS / GPT-SoVITS / IndexTTS 作为探索项
```

选择标准：

```text
中文自然度
首包延迟
RTF
是否支持 streaming
本地部署复杂度
asyncio 集成难度
许可证风险
```

## 11. Audio2Face / FaceDriver 路线

当前：

```text
EmoTalk 输入完整 wav，输出 [N, 52]。
Morpheus 目标也按完整 wav / 输出目录方式接入。
```

短期不追求真 streaming EmoTalk。推荐：

```text
按 TTS 句子或 0.8~1.2s 小块生成 face。
face 允许相对 audio 滞后 100~300ms。
后续加入 overlap 200~300ms + crossfade。
```

长期：

```text
使用 teacher 数据蒸馏低延迟 FaceDriver。
输入 audio chunk + emotion + intensity。
输出未来 200~500ms 的 52 维 blendshape。
```

## 12. UE5 与灰模预览

UE5 完成前，正式视觉验收使用：

```text
scripts/render_emotalk_grey_head.py
```

输入：

```text
reply.wav
face.npy
```

输出：

```text
3D grey-head MP4
```

UE5 阶段需要补齐：

```text
UE5 工程路径
角色类型：MetaHuman / ARKit / 自定义 morph target
52 维到 UE5 曲线映射
WebSocket 或 HTTP 接收方式
播放 buffer
playback.stop
turn_id / generation_epoch stale drop
```

## 13. 阶段性目标

### 阶段一：稳定基线

```text
保留 HTTP /pipeline/audio。
保留 WS /pipeline/stream。
保留 mock provider。
保留真实 provider。
保留灰模预览。
```

### 阶段二：延迟观测与常驻模型

```text
latency metrics
Ollama prewarm / keep_alive
Ollama options
Piper 常驻化
```

### 阶段三：barge-in 可打断

```text
VAD / speech_start
server.playback.stop
server.turn.interrupting
generation_epoch
stale_drop
old_turn_leak_count
```

### 阶段四：句子级增量

```text
LLM streaming
sentence segmenter
句子级 TTS
句子级 / 小块 Audio2Face
face 追赶音频
```

### 阶段五：chunked ASR / partial transcript

```text
unstable_partial
stable_partial
final_transcript
partial 先展示，不先驱动 LLM
```

### 阶段六：UE5 实时接入

```text
UE5 frame chunk
timestamp
mapping table
playback buffer
playback.stop
```

### 阶段七：teacher 数据与 student FaceDriver

```text
批量生成 teacher 数据。
训练 audio chunk -> 52 维 blendshape student。
用 provider 切换 student / EmoTalk / Morpheus。
```

### 阶段八：TTS-Face 联合优化和长期端到端

```text
共享 emotion/prosody state。
统一音频与脸部时间轴。
探索 speech-to-speech-to-face 端到端模型。
```

## 14. 成功标准

项目成功标准：

```text
用户可以自然说话。
系统可以快速理解。
数字人可以快速开口。
用户插话时数字人可以快速停下。
旧 turn 不再泄漏音频或脸部帧。
语音与嘴型基本同步。
情绪和表情一致。
灰模头或 UE5 可以稳定视觉播放。
模块可以独立替换。
系统可以产出 teacher 数据。
后续可以蒸馏更低延迟模型。
```

最终路线：

```text
模块化离线 pipeline
  -> 模块化准实时 pipeline
  -> 常驻模型降延迟
  -> 可打断 barge-in
  -> partial ASR / 更低延迟 TTS
  -> UE5 实时驱动
  -> student FaceDriver
  -> TTS-Face 联合模型
  -> 更完整端到端模型探索
```
