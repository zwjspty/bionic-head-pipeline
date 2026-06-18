# 模块化端到端流式仿生人头交互系统

## 1. 总体目标

本项目目标是在现有仿生人头仿真链路基础上，构建一个支持实时语音交互、情绪表达和面部动画驱动的模块化端到端系统。

系统应能够接收用户语音输入，自动完成语音识别、语义理解、回复生成、情绪预测、语音合成和面部 blendshape 生成，并将结果实时传输给 UE5 或其他数字人渲染端，实现低延迟、可交互、可替换、可扩展的仿生人头对话能力。

最终目标不是把所有模块合成一个不可解释的大模型，而是构建一个“外部端到端、内部模块化”的实时系统：外部用户只感知到从说话到数字人回应的连续交互过程，内部系统仍保留 ASR、LLM、TTS、Audio2Face、UE5 Formatter 等清晰模块边界，方便调试、替换、蒸馏和长期演进。

## 2. 当前基础

当前系统已经完成一条可运行的离线链路：

```text
用户 wav 音频
  -> ASR 语音识别
  -> LLM 生成回复文本、情绪标签和情绪强度
  -> TTS 生成回复语音
  -> Morpheus 根据回复语音生成 52 维 blendshape
  -> 输出 npy / json / UE5 JSON
```

现有系统的优点是模块边界清晰，已经采用 adapter 架构。后续替换 ASR、LLM、TTS、Audio2Face 或 UE5 输出格式时，不需要大规模修改外部 HTTP API，只需要新增 adapter、注册 provider、修改配置并重启服务。

但当前系统仍然是整段式处理：

```text
完整 wav 输入
  -> 等 ASR 完成
  -> 等 LLM 完成
  -> 等 TTS 完成
  -> 等 Morpheus 完成
  -> 输出完整 UE5 JSON
```

因此，当前系统的主要瓶颈不是单个模块本身，而是每一阶段都在等待上一阶段完整结束，导致端到端交互延迟较高。

## 3. 核心建设目标

本项目的核心建设目标是将当前整段式 pipeline 升级为模块化端到端 streaming pipeline。

目标架构如下：

```text
用户边说话
  -> 音频 chunk 输入
  -> ASR 边输出 partial text
  -> LLM 边输出 token / sentence chunk
  -> TTS 边输出 audio chunk
  -> FaceDriver 边输出 blendshape chunk
  -> UE5 边接收、边缓存、边播放
```

系统需要同时保留两个接口：

```text
POST /pipeline/audio
用于离线完整链路测试、数据生成、回归测试和 teacher 数据采集。

WS /pipeline/stream
用于实时交互、音频 chunk 输入、文本 chunk 输出、音频 chunk 输出、blendshape chunk 输出和 UE5 实时播放。
```

也就是说，`/pipeline/audio` 负责稳定性和可复现，`/pipeline/stream` 负责实时性和交互体验。

## 4. 工程目标

### 4.1 保留现有模块化架构

继续沿用现有 adapter 设计：

```text
ASR adapter
LLM adapter
TTS adapter
Audio2Face adapter
UE5 formatter
Video renderer
```

每个模块保留原有离线接口，同时新增 streaming 接口。

示例：

```text
ASR:
  transcribe(wav_path)
  transcribe_stream(audio_chunk)

LLM:
  chat(text, history)
  chat_stream(text, history)

TTS:
  synthesize(text, emotion, intensity)
  synthesize_stream(text_chunk, emotion, intensity)

Audio2Face:
  drive(wav_path, emotion, intensity)
  drive_stream(audio_chunk, emotion, intensity)

UE5:
  build_payload(full_result)
  build_stream_payload(frame_chunk)
```

这样可以在不破坏现有系统的情况下逐步加入流式能力。

### 4.2 建立实时 Orchestrator

新增一个实时编排层，用于协调 ASR、LLM、TTS、FaceDriver 和 UE5 输出。

模块之间不再通过完整文件阻塞传递，而是通过事件队列或异步队列传递：

```text
audio_queue
asr_text_queue
llm_chunk_queue
tts_audio_queue
face_frame_queue
ue5_frame_queue
```

每个事件都必须带时间戳，方便做同步、调试和延迟统计。

事件类型包括：

```text
audio.chunk
asr.partial
asr.final
llm.token
llm.chunk
tts.audio
face.frames
ue5.frames
pipeline.done
pipeline.error
```

### 4.3 降低端到端延迟

系统应从“整段等待”改为“边生成边消费”。

重点降低以下几个时间：

```text
用户说完 -> ASR final
ASR final -> LLM first token
LLM first token -> TTS first audio
TTS first audio -> FaceDriver first frame
FaceDriver first frame -> UE5 first rendered frame
用户说完 -> 数字人开始说话
用户说完 -> 数字人开始动嘴
```

短期目标是减少用户说完后的明显空白等待；中期目标是让数字人可以在合理时间内开始说话和动嘴；长期目标是实现接近自然对话的实时响应。

## 5. 研究目标

### 5.1 模块化 teacher pipeline 数据生成

利用现有稳定 pipeline 批量生成训练数据：

```text
user.wav
asr.text
llm.reply
llm.emotion
llm.intensity
tts.wav_path
morpheus.output_npy
morpheus.output_json
ue5.output_json
```

这些数据可以作为 teacher labels，用于后续训练更轻量、更低延迟的 student 模型。

### 5.2 Morpheus 蒸馏为实时 FaceDriver

优先研究将 Morpheus 蒸馏为轻量级实时 FaceDriver。

Teacher 输入输出：

```text
输入：
  response.wav
  emotion
  intensity

输出：
  52 维 blendshape frames
```

Student 目标：

```text
输入：
  audio chunk
  emotion
  intensity

输出：
  未来 200ms 到 500ms 的 52 维 blendshape frame chunk
```

该方向的目标是减少 Morpheus 对完整回复音频的依赖，让表情驱动可以边听 TTS 音频边生成，从而降低数字人嘴型和表情启动延迟。

### 5.3 TTS 与 FaceDriver 联合建模

在 Morpheus 蒸馏稳定后，进一步研究 TTS 与 FaceDriver 的联合建模。

目标是让语音韵律、情绪强度和面部表情共享同一个中间状态，避免出现：

```text
声音有情绪，但脸部表情不一致
TTS 断句和面部动作断句不一致
口型与语音节奏错位
情绪强度变化不连续
```

联合建模目标：

```text
text chunk + emotion state
  -> audio chunk
  -> synchronized blendshape chunk
```

### 5.4 长期端到端模型探索

长期可以探索更完整的端到端模型：

```text
user audio
  -> reply audio
  -> 52-dim blendshape
```

但该方向只作为研究目标，不作为短期生产目标。原因是完整端到端模型需要大量高质量多模态数据，调试难度高，可控性弱，不适合当前阶段直接替代已有系统。

## 6. 阶段性目标

### 阶段一：稳定现有离线 pipeline

目标：

```text
保留 POST /pipeline/audio
完善日志
完善 health check
完善 diagnostics
完善最新结果保存
完善错误处理
建立基础评测脚本
```

验收标准：

```text
完整链路可以稳定输出 asr.text
完整链路可以稳定输出 llm.reply
完整链路可以稳定输出 tts.wav_path
完整链路可以稳定输出 morpheus.output_npy
完整链路可以稳定输出 ue5.output_json
静音输入可以正确返回 no_speech_detected
```

### 阶段二：新增伪流式 pipeline

目标：

```text
新增 WS /pipeline/stream
支持音频 chunk 输入
ASR 先采用分段识别
LLM 支持 token streaming
TTS 采用句子级合成
Morpheus 采用句子级生成
UE5 支持 frame chunk 接收和播放 buffer
```

这一阶段不要求所有模块都是真流式，但要求系统体验从“等完整回复”变成“分句生成、分句播放”。

验收标准：

```text
用户说完后，系统可以较快输出第一句回复
UE5 可以先播放第一句对应的 blendshape
后续句子继续生成和播放
老接口 /pipeline/audio 不受影响
```

### 阶段三：真流式 ASR 和 TTS

目标：

```text
替换或增强 ASR，使其支持 partial text
替换或增强 TTS，使其支持 audio chunk 输出
建立时间戳对齐机制
建立 jitter buffer
建立中断机制
支持用户打断数字人说话
```

验收标准：

```text
ASR 可以边听边输出 partial result
LLM 可以基于稳定文本片段开始回复
TTS 可以输出首个 audio chunk
UE5 可以持续消费 face frame chunk
系统支持低延迟首包响应
```

### 阶段四：Morpheus teacher 蒸馏

目标：

```text
批量使用现有 pipeline 生成训练数据
训练 audio-to-blendshape student model
支持 audio chunk -> 52维 blendshape chunk
与 Morpheus 输出做对齐评估
替换或旁路 Morpheus 离线处理
```

验收标准：

```text
student FaceDriver 可以实时输出 52维 blendshape
输出帧率稳定在 30fps
口型同步质量接近 Morpheus
首帧延迟明显低于完整 Morpheus pipeline
UE5 可以直接消费 student 输出
```

### 阶段五：TTS-Face 联合优化

目标：

```text
建立共享 emotion/prosody state
让 TTS 和 FaceDriver 在同一时间轴上输出
优化语音、口型、表情的一致性
支持不同情绪强度下的面部表现
```

验收标准：

```text
语音情绪和面部情绪一致
口型与语音时间对齐
表情过渡平滑
不同 emotion/intensity 下有可感知差异
```

## 7. 最终交付目标

最终系统应交付以下能力：

```text
1. 一个稳定的离线完整链路接口：
   POST /pipeline/audio

2. 一个实时交互接口：
   WS /pipeline/stream

3. 一套模块化 adapter 框架：
   支持 ASR、LLM、TTS、Audio2Face、UE5 Formatter 快速替换

4. 一套 teacher 数据生成工具：
   用当前 pipeline 批量生成 audio、text、emotion、tts、blendshape 数据

5. 一个低延迟 FaceDriver student：
   支持 audio chunk -> 52维 blendshape chunk

6. 一个 UE5 实时驱动协议：
   支持 frame chunk、fps、timestamp、channel mapping

7. 一套评测指标：
   覆盖延迟、同步、稳定性、口型质量、情绪一致性

8. 一套可扩展路线：
   支持后续接入 SenseVoice、CosyVoice、Audio2Face-3D、MetaHuman、ARKit 或自研模型
```

## 8. 成功标准

项目成功的判断标准不是“模型是否完全端到端”，而是系统是否具备以下能力：

```text
用户可以自然说话
系统可以快速理解
数字人可以快速开口
语音和嘴型基本同步
表情和情绪一致
UE5 能稳定实时播放
模块可以独立替换
系统可以产生训练数据
后续可以蒸馏更低延迟模型
```

最终希望达到的状态是：

```text
外部体验：
  用户说话 -> 数字人自然回应

内部架构：
  模块清晰、可替换、可调试、可蒸馏

长期演进：
  从模块化 pipeline
  -> 模块化 streaming pipeline
  -> Morpheus student
  -> TTS-Face 联合模型
  -> 更完整的 speech-to-speech-to-face 端到端模型
```
