# Mock provider development

默认自动化测试和本地开发使用 `config/mock.json`。

## Provider 切换

```json
{
  "adapters": {
    "asr": {"provider": "mock"},
    "llm": {"provider": "mock"},
    "tts": {"provider": "mock"},
    "audio2face": {"provider": "mock"},
    "ue5": {"provider": "mock"}
  }
}
```

P0 registry 只接受 `mock`。真实 provider 名称保留在 `config/real.example.json`，P1 再接入。

## 延迟

```json
{
  "mock": {
    "latency_ms": {
      "asr": 200,
      "llm_first_token": 300,
      "llm_token": 10,
      "tts": 400,
      "face": 300
    }
  }
}
```

这可以模拟慢 ASR、慢首 token、逐 token 输出、慢 TTS 和慢 Audio2Face。

## 失败与超时

```json
{
  "mock": {
    "fail_stage": "tts",
    "timeout_stage": null
  }
}
```

支持 stage：

- `asr`
- `llm`
- `tts`
- `audio2face` / `face`
- `ue5`

`fail_stage` 映射为 `provider_failed`。`timeout_stage` 会触发 registry timeout wrapper，映射为 `provider_timeout`。

## 固定输出

```json
{
  "mock": {
    "asr_text": "你好",
    "reply": "你好！很高兴见到你。",
    "emotion": "friendly",
    "intensity": 0.8
  }
}
```

Mock TTS 会生成 250ms、16kHz、mono、16-bit PCM WAV。Mock Audio2Face 会按 WAV 时长生成 30fps、52 维 deterministic frames。

## 常用验证

```bash
.venv/bin/python -m pytest -m 'not integration' -v
.venv/bin/python -m pytest tests/integration/test_http_api.py -v
.venv/bin/python -m pytest tests/integration/test_websocket_api.py -v
```
