# bionic-head-ue5-v1

P0 UE5 输出协议固定为 Morpheus 原始 52 维格式。

```json
{
  "protocol": "bionic-head-ue5-v1",
  "format": "morpheus_52_raw",
  "fps": 30,
  "channel_count": 52,
  "channels": ["morpheus_00", "morpheus_01"],
  "frame_count": 1,
  "frames": [
    {
      "frame_index": 0,
      "time_seconds": 0.0,
      "weights": [0.0]
    }
  ]
}
```

约束：

- `channel_count` 必须是 52。
- `channels` 必须是 `morpheus_00` 到 `morpheus_51`。
- 每帧 `weights` 必须正好 52 个有限数字。
- `frame_index` 从 0 递增。
- `time_seconds = frame_index / fps`。
- WebSocket `server.ue5.frames` 每个 chunk 最多 30 帧，不重置 frame index 或 time。

当前格式不声明是 ARKit、MetaHuman 或任何 UE5 曲线标准。Morpheus 52 维到 UE5 曲线名的映射表后续单独补。

播放语义、`generation_epoch` stale drop、`server.playback.stop` 清 buffer、segment buffering、audio ownership 等规则见：

- `docs/protocols/bionic-head-ue5-playback-v1.md`
