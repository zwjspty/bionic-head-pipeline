# bionic-head-stream-v1

`WS /pipeline/stream` 使用 JSON 事件和二进制帧组合。所有 JSON 事件都使用同一个 envelope：

```json
{
  "protocol": "bionic-head-stream-v1",
  "type": "server.asr.final",
  "event_id": "uuid",
  "session_id": "uuid",
  "turn_id": "uuid-or-null",
  "sequence": 1,
  "timestamp": "2026-06-22T00:00:00Z",
  "payload": {}
}
```

服务端 sequence 与客户端 sequence 分开计数，各自必须严格递增。

## 客户端事件

- `client.session.start`：`turn_id = null`，payload 可为空。
- `client.audio.start`：开始一个 turn，payload 默认 `sample_rate=16000`、`channels=1`、`sample_width_bytes=2`。
- `client.audio.chunk`：JSON 元数据，必须紧跟一个 PCM binary 帧。payload 必须包含 `byte_length` 和 `duration_ms`，duration 必须是 20–100ms。
- `client.audio.end`：结束当前 turn。
- `client.turn.cancel`：取消当前 turn。
- `client.ping`：可在 session 级或 turn 级发送。

输入音频二进制格式固定为 PCM signed 16-bit little-endian、单声道、16000 Hz。

## 服务端事件

- `server.session.ready`
- `server.state`
- `server.asr.final`
- `server.llm.token`
- `server.llm.chunk`
- `server.tts.audio`
- `server.face.frames`
- `server.ue5.frames`
- `server.segment.ready`
- `server.turn.cancelled`
- `server.pipeline.done`
- `server.pipeline.error`
- `server.pong`

除 session 级事件外，payload 会重复 `session_id` 和 `turn_id`。

## JSON/binary 配对

客户端：

1. 发送 `client.audio.chunk` JSON。
2. 下一帧必须是对应 PCM binary。
3. binary 长度必须等于 `payload.byte_length`。

服务端：

1. 发送 `server.tts.audio` JSON。
2. 下一帧一定是对应 WAV binary。
3. JSON payload 包含 `chunk_id`、`format: wav`、`sample_rate`、`byte_length`、`duration_seconds`。

## 分段事件顺序

`/pipeline/stream` 采用短文本分段和 Face 后台追赶策略。服务端保证：

```text
每个 server.tts.audio JSON 后面紧跟对应 WAV binary。
同一个 chunk_id 的 server.face.frames / server.ue5.frames / server.segment.ready 会在对应 TTS 之后出现。
server.pipeline.done 只会在当前 turn 的所有 Face / UE5 分段处理完成后出现。
```

客户端不能假设每个分段都严格按：

```text
server.tts.audio -> server.face.frames -> server.ue5.frames -> server.segment.ready
```

连续成组出现。为了降低可听延迟，服务端可能先连续发送多个 `server.tts.audio`，再发送较早分段的 Face / UE5 事件。客户端必须使用 `chunk_id` 关联 TTS、Face、UE5 与 segment ready。

其中 `server.tts.audio`、`server.face.frames`、`server.segment.ready` 使用分段级 `chunk_id`，例如：

```text
chunk-0001
```

`server.ue5.frames` 可能会把同一个分段继续拆成 frame 子块，它的 `chunk_id` 形式为：

```text
chunk-0001-0000
chunk-0001-0001
```

客户端应把最后一个纯数字后缀视为 UE5 子块编号，父分段 ID 为前缀，例如 `chunk-0001-0000` 对应分段 `chunk-0001`。

## 终态与取消

每个 turn 只能出现一个终态：

- `server.pipeline.done`
- `server.pipeline.error`
- `server.turn.cancelled`

显式 `client.turn.cancel` 会取消当前 turn。正在运行的旧任务如果之后产出结果，会按 turn current guard 丢弃，不会覆盖 latest，也不会继续推送给客户端。

新的 `client.audio.start` 在旧 turn THINKING/SPEAKING/LISTENING 时会先取消旧 turn，再开始新 turn。

## 错误

`server.pipeline.error` payload：

```json
{
  "error": {
    "code": "protocol_violation",
    "stage": "websocket",
    "provider": null,
    "retryable": false,
    "message": "safe message"
  }
}
```

错误消息只包含安全信息，不返回堆栈、命令行或本地 provider 路径。
