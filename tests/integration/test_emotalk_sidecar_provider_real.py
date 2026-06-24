from __future__ import annotations

import os
import shlex
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from bionic_head.adapters.emotalk_sidecar import EmoTalkSidecarAudio2FaceAdapter
from bionic_head.core.audio import audio_artifact_from_wav
from bionic_head.core.cancellation import CancellationToken
from bionic_head.domain.models import Emotion, TurnContext


pytestmark = pytest.mark.skipif(
    os.environ.get("BIONIC_HEAD_RUN_REAL_EMOTALK") != "1",
    reason="set BIONIC_HEAD_RUN_REAL_EMOTALK=1 to run the real EmoTalk sidecar provider smoke test",
)


def _write_silence_wav(path: Path) -> Path:
    samples = np.zeros(16000, dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    return path


@pytest.mark.asyncio
async def test_real_provider_prewarm_then_drive_reuses_worker(tmp_path: Path) -> None:
    command = shlex.split(
        os.environ.get(
            "BIONIC_HEAD_REAL_EMOTALK_COMMAND",
            "/home/user/miniconda3/envs/emotalk/bin/python -m bionic_head.emotalk_sidecar_worker",
        )
    )
    adapter = EmoTalkSidecarAudio2FaceAdapter(
        sidecar_command=command,
        sidecar_cwd=Path.cwd(),
        sidecar_env={"PYTHONPATH": "src:."},
        sample_rate=16000,
        fps=30,
        timeout_seconds=20.0,
        prewarm_timeout_seconds=30.0,
    )

    try:
        result = await adapter.prewarm()
        prewarm_pid = adapter.process_pid
        audio = audio_artifact_from_wav(_write_silence_wav(tmp_path / "input.wav"))
        context = TurnContext(
            session_id=uuid4(),
            turn_id=uuid4(),
            artifact_dir=tmp_path / "turn",
            cancellation=CancellationToken(),
            generation_epoch=0,
        )
        face = await adapter.drive(audio, Emotion.NEUTRAL, 0.5, context)

        assert result.available is True
        assert prewarm_pid is not None
        assert adapter.process_pid == prewarm_pid
        assert adapter.process_start_count == 1
        assert face.channel_count == 52
        assert face.frame_count > 0
    finally:
        await adapter.close()
