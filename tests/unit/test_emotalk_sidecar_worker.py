from __future__ import annotations

import io
import os
import sys
from types import SimpleNamespace

import numpy as np

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
    num_samples: int = 4,
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
        audio=audio if audio is not None else np.array([0, 32767, -32768, 1024], dtype=np.int16).tobytes(),
    )


def _decode_stream(stdout: bytes):
    stream = io.BytesIO(stdout)
    while stream.tell() < len(stdout):
        header, body = read_message(stream)
        yield decode_response(encode_message(header, body))


def test_module_import_does_not_eagerly_import_real_model_dependencies() -> None:
    sys.modules.pop("bionic_head.emotalk_sidecar_worker", None)
    sys.modules.pop("torch", None)
    sys.modules.pop("librosa", None)
    sys.modules.pop("transformers", None)

    __import__("bionic_head.emotalk_sidecar_worker")

    assert "torch" not in sys.modules
    assert "librosa" not in sys.modules
    assert "transformers" not in sys.modules


def test_parse_args_exposes_required_defaults() -> None:
    from bionic_head.emotalk_sidecar_worker import parse_args

    args = parse_args([])

    assert args.device == "cpu"
    assert args.emotalk_root == "/home/user/code/EmoTalk_release"
    assert args.checkpoint == "/home/user/code/EmoTalk_release/pretrain_model/EmoTalk.pth"
    assert args.wav2vec_content_path is None
    assert args.wav2vec_emotion_path is None
    assert args.fps == 30
    assert args.person == 0
    assert args.level == 1
    assert args.torch_num_threads == 4
    assert args.torch_interop_threads == 1
    assert args.warmup is False


def test_pcm16_bytes_to_float32_waveform_scales_to_unit_range() -> None:
    from bionic_head.emotalk_sidecar_worker import pcm16_bytes_to_float32_waveform

    waveform = pcm16_bytes_to_float32_waveform(
        np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16).tobytes()
    )

    assert waveform.dtype == np.float32
    assert waveform.shape == (5,)
    np.testing.assert_allclose(
        waveform,
        np.array([-1.0, -0.5, 0.0, 0.5, 32767.0 / 32768.0], dtype=np.float32),
        atol=1e-6,
    )


def test_handle_request_payload_with_fake_runner_returns_valid_success_response() -> None:
    from bionic_head.emotalk_sidecar_worker import handle_request_payload

    calls = []

    def fake_runner(audio: np.ndarray, *, level: int, person: int) -> np.ndarray:
        calls.append((audio.copy(), level, person))
        return np.arange(104, dtype=np.float32).reshape(2, 52)

    response_payload = handle_request_payload(
        encode_request(_valid_request(num_samples=4, fps=24)),
        fake_runner,
        level=7,
        person=2,
    )
    response = decode_response(response_payload)

    assert response.ok is True
    assert response.frame_count == 2
    assert response.channel_count == 52
    assert response.fps == 24
    assert len(calls) == 1
    np.testing.assert_allclose(
        calls[0][0],
        np.array([0.0, 32767.0 / 32768.0, -1.0, 1024.0 / 32768.0], dtype=np.float32),
        atol=1e-6,
    )
    assert calls[0][1:] == (7, 2)


def test_handle_request_payload_returns_failure_response_for_invalid_request() -> None:
    from bionic_head.emotalk_sidecar_worker import handle_request_payload

    bad_request = encode_request(_valid_request(dtype="float32"))
    response_payload = handle_request_payload(bad_request, lambda *_args, **_kwargs: None, level=1, person=0)
    response = decode_response(response_payload)

    assert response.ok is False
    assert response.error_code == "invalid_request"
    assert response.error_message == "dtype must be int16"
    assert response.frames == b""


def test_serve_continues_after_error_and_loads_runner_only_once() -> None:
    from bionic_head.emotalk_sidecar_worker import serve

    load_calls: list[tuple[int, int]] = []
    predict_calls: list[np.ndarray] = []

    def runner_factory(args) -> object:
        load_calls.append((args.level, args.person))

        def fake_runner(audio: np.ndarray, *, level: int, person: int) -> np.ndarray:
            predict_calls.append(audio.copy())
            return np.zeros((1, 52), dtype=np.float32)

        return fake_runner

    stdin = io.BytesIO(
        encode_request(_valid_request(turn_id="ok-1"))
        + encode_request(_valid_request(turn_id="bad", dtype="float32"))
        + encode_request(_valid_request(turn_id="ok-2"))
    )
    stdout = io.BytesIO()
    stderr = io.StringIO()

    exit_code = serve(
        stdin,
        stdout,
        stderr,
        args=SimpleNamespace(level=5, person=9, warmup=False),
        runner_factory=runner_factory,
    )

    assert exit_code == 0
    responses = list(_decode_stream(stdout.getvalue()))
    assert [response.ok for response in responses] == [True, False, True]
    assert responses[1].error_code == "invalid_request"
    assert [response.turn_id for response in responses] == ["ok-1", "bad", "ok-2"]
    assert load_calls == [(5, 9)]
    assert len(predict_calls) == 2
    assert "Traceback" not in stderr.getvalue()


def test_serve_logs_request_error_to_stderr_without_writing_text_to_stdout() -> None:
    from bionic_head.emotalk_sidecar_worker import serve

    def exploding_runner(_audio: np.ndarray, *, level: int, person: int) -> np.ndarray:
        raise RuntimeError(f"boom level={level} person={person}")

    stdin = io.BytesIO(encode_request(_valid_request(turn_id="boom")))
    stdout = io.BytesIO()
    stderr = io.StringIO()

    exit_code = serve(
        stdin,
        stdout,
        stderr,
        args=SimpleNamespace(level=3, person=4, warmup=False),
        runner_factory=lambda _args: exploding_runner,
    )

    assert exit_code == 0
    response = next(_decode_stream(stdout.getvalue()))
    assert response.ok is False
    assert response.turn_id == "boom"
    assert response.error_code == "prediction_failed"
    assert "boom level=3 person=4" in response.error_message
    assert "prediction_failed" in stderr.getvalue()
    assert b"Traceback" not in stdout.getvalue()


def test_serve_exits_cleanly_on_eof() -> None:
    from bionic_head.emotalk_sidecar_worker import serve

    stdout = io.BytesIO()
    stderr = io.StringIO()

    exit_code = serve(
        io.BytesIO(),
        stdout,
        stderr,
        args=SimpleNamespace(level=1, person=0, warmup=False),
        runner_factory=lambda _args: lambda audio, *, level, person: np.zeros((1, 52), dtype=np.float32),
    )

    assert exit_code == 0
    assert stdout.getvalue() == b""
    assert stderr.getvalue() == ""
