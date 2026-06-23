# WebSocket validation client

Run against a local server:

```bash
.venv/bin/python scripts/stream_client.py \
  --url ws://127.0.0.1:8000/pipeline/stream \
  --wav /path/to/input.wav \
  --output-dir client-output \
  --chunk-ms 40
```

Input WAV must be mono PCM16 at 16 kHz. The client sends full `bionic-head-stream-v1` envelopes, pairs each `client.audio.chunk` JSON message with the next PCM binary frame, and saves received artifacts:

- `events.jsonl`
- `tts/{chunk_id}.wav`
- `ue5/{chunk_id}.json`
- `summary.json`

The receiver validates server sequence ordering, `server.tts.audio` JSON/binary pairing, binary byte length, and UE5 frame chunk continuity. A terminal event is one of `server.pipeline.done`, `server.pipeline.error`, or `server.turn.cancelled`.

Task 7 note: the server may emit later `server.tts.audio` chunks before earlier Audio2Face / UE5 chunks finish. Treat the segment `chunk_id` as the stable join key; do not assume Face frames arrive immediately after each TTS chunk. UE5 frame files may use sub-chunk IDs such as `chunk-0001-0000`, where `chunk-0001` is the parent segment.
