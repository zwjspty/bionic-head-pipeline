# Task 20: UE5 Playback Contract Design

## Goal

Define a stable UE5 playback contract for the existing `/pipeline/stream` output. This task does not implement UE5 Blueprints, Live Link, MetaHuman mapping, WebRTC, AEC, or provider/model changes. It defines what a UE5 receiver must accept, buffer, play, stop, drop, validate, and replay.

## Scope

Task 20 covers:

- `server.ue5.frames` playback semantics.
- `server.playback.stop` receiver behavior.
- `generation_epoch` stale-drop behavior.
- Segment and frame ordering rules.
- Default audio ownership: `external_audio_clock`.
- A Python validator and replay model that can be used before real UE5 integration.
- Fixtures and CLI tools for validating/replaying contract examples.

Task 20 does not change the existing stream wire protocol. The current payload field is `start_frame_index`; although earlier notes used `frame_start_index`, this contract standardizes on the existing `start_frame_index` field to avoid breaking current clients and tests.

## Existing protocol relationship

`docs/protocols/bionic-head-ue5-v1.md` defines the Morpheus 52-channel frame format:

- `protocol = bionic-head-ue5-v1`
- `format = morpheus_52_raw`
- `fps = 30` by default
- `channel_count = 52`
- `channels = morpheus_00 ... morpheus_51`
- each frame has `frame_index`, `time_seconds`, and `weights[52]`

Task 20 adds playback semantics around chunks of those frames, not a new facial rig format.

## Required `server.ue5.frames` payload fields

A UE5 playback receiver must support these fields:

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
    {"frame_index": 0, "time_seconds": 0.0, "weights": [0.0]}
  ]
}
```

Required for validation:

- `format == "morpheus_52_raw"`
- `channel_count == 52`
- `fps > 0`
- `generation_epoch >= 0`
- `segment_index >= 0` when present
- `start_frame_index >= 0`
- `pts_start_ms >= 0` when present
- `frame_count == len(frames)`
- every frame has exactly 52 finite numeric weights
- frame indices are contiguous from `start_frame_index`

`session_id`, `turn_id`, `generation_epoch`, and `segment_id` may be supplied by either the stream envelope or the payload. A standalone replay/fixture payload must include them in the payload.

## `generation_epoch` and stale drop

UE5 must keep one active generation epoch per connection/session.

- Frames with `generation_epoch < active_generation_epoch` are stale and must be dropped.
- Frames with `generation_epoch == active_generation_epoch` may be buffered or played.
- Frames with `generation_epoch > active_generation_epoch` advance the active epoch and imply previous pending face frames are no longer playable.

`server.playback.stop` is authoritative. On receipt, UE5 must:

1. stop current face playback immediately;
2. clear pending face buffers;
3. update `active_generation_epoch` from the event/payload;
4. drop late frames from older epochs;
5. record `stop_to_face_clear_ms` or equivalent diagnostic timing.

## Segment buffer and ordering

UE5 should group by `generation_epoch`, `turn_id`, and `segment_id`.

Within a segment:

- chunks are ordered by `start_frame_index`;
- a repeated chunk with the same `chunk_id` or same `start_frame_index` must not be played twice;
- a gap may be skipped with a warning metric, but must not block newer valid segments forever;
- an overlap must be ignored or overwritten deterministically, not double-applied;
- buffer size limits should drop oldest stale data first.

Recommended metrics:

- `ue5_received_frame_chunks`
- `ue5_buffered_frame_count`
- `ue5_stale_drop_count`
- `ue5_duplicate_drop_count`
- `ue5_missing_or_gap_count`
- `ue5_playback_stop_count`
- `ue5_buffer_clear_count`

## Audio ownership

Default mode is `external_audio_clock`.

In this mode:

- local demo client or another upstream player owns audio playback;
- UE5 plays only face frames;
- UE5 aligns face playback using `segment_id`, `pts_start_ms`, `fps`, and external playback start timing;
- `server.playback.stop` still clears UE5 face buffers immediately.

Reserved mode is `ue5_audio_owner`.

In this future mode, UE5 may receive and own both audio and face playback. Task 20 documents the reservation only; it does not implement UE5 audio ownership.

## Disconnect/reconnect behavior

If a connection is lost, UE5 must treat all unplayed buffers as invalid. A new connection/session must wait for a new `server.session.ready` and fresh `generation_epoch` state. A receiver should not resume old face buffers after reconnect unless a future protocol explicitly adds resume tokens.

## Replay and validation

Task 20 adds a Python contract module plus two CLI tools:

- `scripts/validate_ue5_playback_contract.py`: validate fixture/event JSON.
- `scripts/replay_ue5_frames.py`: replay fixture events through a receiver-state model and print pass/drop/clear metrics.

Fixtures cover:

- valid segment;
- stale generation drop;
- playback stop buffer clear;
- invalid channel count;
- invalid frame shape.

## Acceptance criteria

- Full pytest passes.
- UE5 playback contract docs exist.
- Contract defines required `server.ue5.frames` fields.
- Contract defines stale drop by `generation_epoch`.
- Contract defines `server.playback.stop` buffer clearing.
- Contract defines default `external_audio_clock` ownership.
- Validator accepts valid payloads and rejects invalid shape/count/fps/epoch data.
- Fixtures and CLI help exist.
- No ASR/TTS/LLM/EmoTalk/provider behavior changes.
