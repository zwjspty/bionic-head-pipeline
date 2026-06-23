# Task 5: interrupt epoch and stale-drop design

## Goal

Make WebSocket interruption safe before true full-duplex VAD lands: when a turn is interrupted, clients stop playback immediately, old turn output is tagged stale, and stale results never overwrite latest artifacts.

## Scope

This task covers only server-side protocol/control semantics and the test client behavior needed to validate them.

In scope:

- Add `generation_epoch` to every server event.
- Add `server.playback.stop` as the immediate playback-clearing event.
- Increment the session epoch when a current turn is cancelled or replaced by a new turn.
- Ensure old turn output is suppressed after cancellation.
- Record `server.turn.stale_drop` internally and expose it only when a suppressed send path can safely emit it.
- Update `scripts/stream_client.py` to clear pending playback buffers on `server.playback.stop`.

Out of scope:

- Real client-side echo cancellation.
- Real VAD/barge-in detection. That is Task 6.
- UE5 Blueprint integration.
- Multi-session scheduling beyond the existing single-session-safe structure.

## Protocol changes

Server events keep the existing envelope fields and additionally include:

```json
{
  "generation_epoch": 1,
  "payload": {
    "generation_epoch": 1
  }
}
```

`server.playback.stop` is emitted before `server.turn.cancelled` when the server actively cancels an existing turn due to user cancellation or a replacement `client.audio.start`.

Clients must treat `server.playback.stop` as stronger than segment completion:

- clear queued TTS chunks;
- clear pending UE5/face chunks;
- ignore old chunks with lower `generation_epoch`.

## Stale-drop behavior

`TurnHandle` owns the epoch it was created with. It is current only while:

- it is not cancelled;
- it has no terminal event;
- its epoch matches the session epoch.

If an async provider returns late after cancellation, `emit_if_current` and `commit_if_current` reject the operation. The late result is not sent and not published to `data/latest`.

## Testing approach

- Unit-test server event envelopes include `generation_epoch`.
- Unit-test `TurnHandle` suppresses emits after the session epoch changes.
- Unit-test connection cancellation emits `server.playback.stop` before `server.turn.cancelled`.
- Unit-test stream client clears pending playback on `server.playback.stop`.
- Run all non-integration tests.

## Acceptance

- `server.playback.stop` exists and is emitted on interrupt/cancel.
- New server events carry `generation_epoch`.
- Cancelled/stale turns do not emit normal audio/face/UE5 events after interruption.
- Existing `/pipeline/stream` happy path still ends in `server.pipeline.done`.
