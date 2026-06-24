from __future__ import annotations

import os
import subprocess
import shlex

import numpy as np
import pytest

from bionic_head.sidecar_protocol import SidecarRequest, decode_response, encode_message, encode_request, read_message


pytestmark = pytest.mark.skipif(
    os.environ.get("BIONIC_HEAD_RUN_REAL_EMOTALK") != "1",
    reason="set BIONIC_HEAD_RUN_REAL_EMOTALK=1 to run the real EmoTalk sidecar smoke test",
)


def _request(turn_id: str) -> bytes:
    samples = np.zeros(16000, dtype=np.int16)
    return encode_request(
        SidecarRequest(
            session_id="real-session",
            turn_id=turn_id,
            generation_epoch=0,
            sample_rate=16000,
            channels=1,
            dtype="int16",
            num_samples=int(samples.size),
            fps=30,
            audio=samples.tobytes(),
        )
    )


def test_real_worker_smoke_two_requests_then_clean_exit() -> None:
    command = shlex.split(
        os.environ.get(
            "BIONIC_HEAD_REAL_EMOTALK_COMMAND",
            "conda run -n emotalk python -m bionic_head.emotalk_sidecar_worker",
        )
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, ["src", os.environ.get("PYTHONPATH", "")]))}
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    process.stdin.write(_request("one"))
    process.stdin.flush()
    first_header, first_body = read_message(process.stdout)
    first = decode_response(encode_message(first_header, first_body))

    process.stdin.write(_request("two"))
    process.stdin.flush()
    second_header, second_body = read_message(process.stdout)
    second = decode_response(encode_message(second_header, second_body))

    assert first.ok is True
    assert second.ok is True
    assert first.channel_count == 52
    assert second.channel_count == 52
    assert first.frame_count > 0
    assert second.frame_count > 0
    assert first.metrics is not None
    assert second.metrics is not None
    assert isinstance(first.metrics["worker_total_ms"], float)
    assert isinstance(first.metrics["model_predict_ms"], float)
    assert first.metrics["model_predict_ms"] <= first.metrics["worker_total_ms"]
    assert second.metrics["model_predict_ms"] <= second.metrics["worker_total_ms"]
    first_frames = np.frombuffer(first.frames, dtype=np.float32).reshape(first.frame_count, 52)
    second_frames = np.frombuffer(second.frames, dtype=np.float32).reshape(second.frame_count, 52)
    assert first_frames.shape == (first.frame_count, 52)
    assert second_frames.shape == (second.frame_count, 52)

    process.stdin.close()
    assert process.wait(timeout=10) == 0
