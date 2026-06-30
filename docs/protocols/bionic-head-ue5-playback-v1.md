# bionic-head-ue5-playback-v1

This document defines playback semantics for UE5 receivers that consume `/pipeline/stream` `server.ue5.frames` events. It complements `bionic-head-ue5-v1`, which defines the Morpheus 52-channel frame format.

Task 20 is a contract only. It does not implement UE5 Blueprints, Live Link, MetaHuman mapping, UE5 audio ownership, WebRTC, AEC, or provider/model changes.

## Required frame chunk payload

`server.ue5.frames` uses the normal `bionic-head-stream-v1` envelope. The payload must contain or inherit the following fields:

```json
{
  "protocol": "bionic-head-ue5-v1",
  "format": "morpheus_52_raw",
  "session_id": "session-abc",
  "turn_id": "turn-001",
  "generation_epoch": 3,
  "chunk_id": "chunk-0001-0000",
  "segment_id": "chunk-0001",
  "segment_index": 0,
  "fps": 30,
  "channel_count": 52,
  "channels": ["morpheus_00"],
  "start_frame_index": 0,
  "frame_count": 1,
  "pts_start_ms": 0.0,
  "is_last": true,
  "frames": [
    {
      "frame_index": 0,
      "time_seconds": 0.0,
      "weights": [0.0]
    }
  ]
}
```

The existing wire field is `start_frame_index`. UE5 receivers should not wait for a `frame_start_index` field.

## Validation rules

A receiver or validator must reject frame chunks when:

- `format != "morpheus_52_raw"`;
- `channel_count != 52`;
- `fps <= 0`;
- `generation_epoch < 0`;
- `segment_index < 0` when present;
- `start_frame_index < 0`;
- `pts_start_ms < 0` when present;
- `frame_count != len(frames)`;
- frame indices are not contiguous from `start_frame_index`;
- any frame weight is non-numeric, non-finite, or not exactly 52 values.

## Generation epoch and stale drop

The receiver maintains `active_generation_epoch`.

- `generation_epoch < active_generation_epoch`: drop the frame chunk as stale.
- `generation_epoch == active_generation_epoch`: buffer/play normally.
- `generation_epoch > active_generation_epoch`: advance active epoch and clear older pending face buffers.

This rule prevents old turns from animating the face after barge-in or cancel.

## `server.playback.stop`

On `server.playback.stop`, UE5 must:

1. stop current face playback immediately;
2. clear pending frame buffers;
3. update `active_generation_epoch` from the event or payload;
4. drop late frames from older generations;
5. record `stop_to_face_clear_ms` or equivalent diagnostic timing.

`server.playback.stop` has priority over normal frame playback.

## Segment ordering and buffering

UE5 should group frames by:

```text
generation_epoch -> turn_id -> segment_id
```

Within a segment:

- order chunks by `start_frame_index`;
- do not play a duplicate `chunk_id` twice;
- do not play overlapping frame ranges twice;
- gaps may be skipped with a diagnostic warning instead of blocking forever;
- bounded buffers should drop stale generations before dropping current generation data.

## Audio ownership

Default mode:

```text
external_audio_clock
```

In this mode, UE5 only plays face frames. Audio is owned by the local demo client or another upstream player. UE5 aligns face frames using `segment_id`, `pts_start_ms`, `fps`, and external playback start timing.

Reserved future mode:

```text
ue5_audio_owner
```

In this mode UE5 may receive and play both audio and face. It is documented as a future option only and is not implemented in Task 20.

## Disconnect and reconnect

On disconnect, UE5 must discard unplayed buffers. A reconnect starts from fresh session/generation state. The current contract has no resume token and does not support replaying old buffers after reconnect.

## Suggested metrics

- `ue5_received_frame_chunks`
- `ue5_buffered_frame_count`
- `ue5_stale_drop_count`
- `ue5_duplicate_drop_count`
- `ue5_missing_or_gap_count`
- `ue5_playback_stop_count`
- `ue5_buffer_clear_count`

## Fixture validation

Use:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_ue5_playback_contract.py \
  tests/fixtures/ue5_playback/valid_segment.json

PYTHONPATH=src .venv/bin/python scripts/replay_ue5_frames.py \
  tests/fixtures/ue5_playback/playback_stop.json
```

The scripts are pure local checks and do not require UE5, GPU, Ollama, Piper, EmoTalk, microphone, or speaker.
