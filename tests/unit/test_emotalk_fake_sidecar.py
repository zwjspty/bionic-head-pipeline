import io
import os
import math
import subprocess

from bionic_head.emotalk_fake_sidecar import build_fake_frames, handle_request_payload, serve
from bionic_head.sidecar_protocol import (
    SidecarRequest,
    decode_response,
    encode_message,
    encode_request,
    read_message,
)


def _valid_request(
    *,
    session_id: str = "session-1",
    turn_id: str = "turn-1",
    generation_epoch: int = 3,
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: str = "int16",
    num_samples: int = 16000,
    fps: int = 30,
    audio: bytes | None = None,
) -> SidecarRequest:
    return SidecarRequest(
        session_id=session_id,
        turn_id=turn_id,
        generation_epoch=generation_epoch,
        sample_rate=sample_rate,
        channels=channels,
        dtype=dtype,
        num_samples=num_samples,
        fps=fps,
        audio=audio if audio is not None else b"\x00" * (num_samples * 2),
    )


def _read_responses(stdout: bytes):
    stream = io.BytesIO(stdout)
    while stream.tell() < len(stdout):
        header, body = read_message(stream)
        yield decode_response(encode_message(header, body))


def test_build_fake_frames_uses_ceil_duration_times_fps() -> None:
    frame_count, frames = build_fake_frames(num_samples=16001, sample_rate=16000, fps=30)

    assert frame_count == 31
    assert len(frames) == frame_count * 52 * 4


def test_handle_request_payload_returns_success_response() -> None:
    request = _valid_request(
        num_samples=10,
        generation_epoch=11,
        session_id="sess-a",
        turn_id="turn-a",
        fps=24,
    )
    response_payload = handle_request_payload(encode_request(request))
    response = decode_response(response_payload)

    assert response.ok is True
    assert response.session_id == "sess-a"
    assert response.turn_id == "turn-a"
    assert response.generation_epoch == 11
    assert response.fps == 24
    assert response.channel_count == 52
    assert response.dtype == "float32"
    assert response.frame_count == math.ceil(request.num_samples / request.sample_rate * request.fps)
    assert len(response.frames) == response.frame_count * 52 * 4


def test_handle_request_payload_returns_error_for_invalid_sample_rate() -> None:
    request = _valid_request(sample_rate=8000)
    response_payload = handle_request_payload(encode_request(request))
    response = decode_response(response_payload)

    assert response.ok is False
    assert response.error_code == "invalid_request"
    assert response.error_message == "sample_rate must be 16000"
    assert response.frames == b""


def test_handle_request_payload_returns_error_for_invalid_dtype() -> None:
    request = _valid_request(dtype="float32")
    response_payload = handle_request_payload(encode_request(request))
    response = decode_response(response_payload)

    assert response.ok is False
    assert response.error_code == "invalid_request"
    assert response.error_message == "dtype must be int16"
    assert response.frames == b""


def test_handle_request_payload_returns_error_for_audio_length_mismatch() -> None:
    request = _valid_request(num_samples=4, audio=b"\x00")
    response_payload = handle_request_payload(encode_request(request))
    response = decode_response(response_payload)

    assert response.ok is False
    assert response.error_code == "invalid_request"
    assert response.error_message == "audio body length does not match num_samples * 2"
    assert response.frames == b""


def test_serve_handles_three_consecutive_requests_without_cross_talk() -> None:
    stdin = io.BytesIO(
        encode_request(_valid_request(session_id="s1", turn_id="t1", generation_epoch=0))
        + encode_request(_valid_request(session_id="s2", turn_id="t2", generation_epoch=1))
        + encode_request(_valid_request(session_id="s3", turn_id="t3", generation_epoch=2))
    )
    stdout = io.BytesIO()

    exit_code = serve(stdin, stdout)
    assert exit_code == 0

    responses = list(_read_responses(stdout.getvalue()))
    assert len(responses) == 3
    assert [(r.session_id, r.turn_id, r.generation_epoch) for r in responses] == [
        ("s1", "t1", 0),
        ("s2", "t2", 1),
        ("s3", "t3", 2),
    ]
    assert all(r.ok for r in responses)


def test_subprocess_single_request_ok() -> None:
    request = _valid_request(
        session_id="proc",
        turn_id="single",
        generation_epoch=42,
        num_samples=8,
    )
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [".venv/bin/python", "-m", "bionic_head.emotalk_fake_sidecar"],
        input=encode_request(request),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    response = decode_response(result.stdout)
    assert response.ok is True
    assert response.session_id == "proc"
    assert response.turn_id == "single"
    assert response.generation_epoch == 42


def test_subprocess_continuous_requests_and_error_response() -> None:
    valid_a = encode_request(_valid_request(session_id="proc", turn_id="ok-1", generation_epoch=1))
    invalid_sample_rate = encode_request(
        _valid_request(
            session_id="proc",
            turn_id="bad",
            generation_epoch=2,
            dtype="float32",
        )
    )
    valid_b = encode_request(_valid_request(session_id="proc", turn_id="ok-2", generation_epoch=3))
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [".venv/bin/python", "-m", "bionic_head.emotalk_fake_sidecar"],
        input=valid_a + invalid_sample_rate + valid_b,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    responses = list(_read_responses(result.stdout))
    assert len(responses) == 3
    assert responses[0].ok is True
    assert responses[1].ok is False
    assert responses[1].error_code == "invalid_request"
    assert responses[2].ok is True
    assert responses[0].turn_id == "ok-1"
    assert responses[2].turn_id == "ok-2"
    assert result.returncode == 0
    assert b"\x00" not in result.stderr
    assert b"\n" not in result.stdout


def test_serve_continuous_requests_with_invalid_dtype_continues() -> None:
    stdin = io.BytesIO(
        encode_request(_valid_request(session_id="s1", turn_id="ok-1", generation_epoch=1))
        + encode_request(
            _valid_request(
                session_id="s2", turn_id="bad-dtype", generation_epoch=2, dtype="float32"
            )
        )
        + encode_request(_valid_request(session_id="s3", turn_id="ok-2", generation_epoch=3))
    )
    stdout = io.BytesIO()

    exit_code = serve(stdin, stdout)
    assert exit_code == 0

    responses = list(_read_responses(stdout.getvalue()))
    assert len(responses) == 3
    assert responses[0].ok is True
    assert responses[1].ok is False
    assert responses[1].error_code == "invalid_request"
    assert responses[1].error_message == "dtype must be int16"
    assert responses[2].ok is True
    assert (responses[0].session_id, responses[0].turn_id, responses[0].generation_epoch) == (
        "s1",
        "ok-1",
        1,
    )
    assert (responses[2].session_id, responses[2].turn_id, responses[2].generation_epoch) == (
        "s3",
        "ok-2",
        3,
    )


def test_serve_returns_nonzero_on_truncated_header_bytes() -> None:
    stdin = io.BytesIO(b"\x00\x00\x00")
    stdout = io.BytesIO()

    exit_code = serve(stdin, stdout)
    assert exit_code == 1
    assert stdout.getvalue() == b""
