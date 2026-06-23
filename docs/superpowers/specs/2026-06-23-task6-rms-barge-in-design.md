# Task 6: RMS VAD barge-in design

## Goal

Add the smallest useful automatic barge-in behavior: while the bot is thinking or speaking, new microphone audio that looks like speech triggers `server.playback.stop`, cancels the old turn, and starts a new turn without waiting for Silero VAD.

## Scope

In scope:

- Add RMS VAD configuration.
- Detect interrupt speech from incoming PCM16 chunks while the current turn is `THINKING` or `SPEAKING`.
- Require accumulated interrupt speech duration before cancelling the old turn.
- Reuse Task 5 `generation_epoch` and `server.playback.stop` behavior.
- Keep mock tests deterministic and dependency-free.

Out of scope:

- Silero VAD model loading.
- Acoustic echo cancellation.
- Browser microphone configuration.
- True streaming ASR.

## Recommended defaults

```json
{
  "vad": {
    "engine": "rms",
    "interrupt_min_speech_ms": 80,
    "interrupt_rms_threshold": 0.02
  }
}
```

## Behavior

When the server receives a new `client.audio.start` while an existing turn is `THINKING` or `SPEAKING`, it creates an interrupt candidate instead of cancelling immediately. Candidate chunks are buffered separately. Once chunks above `interrupt_rms_threshold` accumulate at least `interrupt_min_speech_ms`, the server:

1. emits `server.playback.stop`;
2. emits `server.turn.cancelled` for the old turn;
3. increments `generation_epoch`;
4. promotes the candidate to the active listening turn with its buffered audio.

If the candidate ends before enough speech is detected, it is discarded and the old turn continues.

## Acceptance

- New audio start alone does not stop playback.
- Consecutive speech chunks above threshold trigger `server.playback.stop`.
- Below-threshold audio does not interrupt.
- The accepted new turn can continue through `/pipeline/stream`.
- Existing mock happy path remains compatible.
