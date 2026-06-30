# UE5 playback contract validation

Task 20 defines the UE5-side playback contract before building the real Unreal receiver.

This is not a UE5 Blueprint implementation. It is a local validation and replay package for checking that `server.ue5.frames` and `server.playback.stop` events can be consumed safely by a future UE5 receiver.

## Contract documents

- Protocol payload format: `docs/protocols/bionic-head-ue5-v1.md`
- Playback behavior contract: `docs/protocols/bionic-head-ue5-playback-v1.md`

The stream keeps the existing field name:

```text
start_frame_index
```

Do not introduce `frame_start_index` in UE5-side code unless the server protocol is intentionally migrated.

## Validate fixtures

```bash
cd /home/user/code/端到端
source .venv/bin/activate

PYTHONPATH=src .venv/bin/python scripts/validate_ue5_playback_contract.py \
  tests/fixtures/ue5_playback_contract/valid_segment.json
```

Expected result:

```json
{
  "success": true,
  "validated_count": 1,
  "failure_count": 0
}
```

Invalid fixtures should fail with a non-zero exit code:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_ue5_playback_contract.py \
  tests/fixtures/ue5_playback_contract/invalid_channel_count.json
```

Expected:

```text
success: false
error includes channel_count
```

## Replay UE5 events

Replay verifies receiver-side buffer rules: generation protection, playback-stop clear, stale drops, duplicate drops, and frame-gap metrics.

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_ue5_frames.py \
  tests/fixtures/ue5_playback_contract/playback_stop_and_stale.json
```

Expected key metrics:

```text
playback_stop_count = 1
stale_drop_count = 1
buffer_clear_count = 1
```

## Receiver rules

A UE5 receiver should:

1. Validate every `server.ue5.frames` payload before adding it to a playback buffer.
2. Use `generation_epoch` as the primary stale-drop guard.
3. Drop duplicate `chunk_id` values.
4. Track expected `start_frame_index` per segment and record gaps.
5. On `server.playback.stop`, clear all buffered frames for old turns immediately.
6. Treat audio as externally owned by default. The existing local clients already own TTS playback and A/V sync strategy.

## What Task 20 intentionally does not do

- No real UE5 Blueprint.
- No MetaHuman / ARKit mapping.
- No Live Link.
- No audio playback in UE5.
- No backend protocol migration.
- No ASR/TTS/LLM/EmoTalk changes.
