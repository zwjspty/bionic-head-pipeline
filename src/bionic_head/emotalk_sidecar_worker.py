from __future__ import annotations

import argparse
import contextlib
import importlib
import struct
import sys
import traceback
from pathlib import Path
from typing import Callable, TextIO

import numpy as np

from .sidecar_protocol import (
    HEADER_PREFIX_SIZE,
    SidecarRequest,
    SidecarResponse,
    SidecarProtocolError,
    decode_message,
    decode_request,
    encode_message,
    encode_response,
)

DEFAULT_EMOTALK_ROOT = "/home/user/code/EmoTalk_release"
DEFAULT_CHECKPOINT = "/home/user/code/EmoTalk_release/pretrain_model/EmoTalk.pth"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNEL_COUNT = 52

Runner = Callable[[np.ndarray], object]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bionic_head.emotalk_sidecar_worker",
        description="Persistent real EmoTalk sidecar worker.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--emotalk-root", default=DEFAULT_EMOTALK_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--wav2vec-content-path", default=None)
    parser.add_argument("--wav2vec-emotion-path", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--person", type=int, default=0)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def pcm16_bytes_to_float32_waveform(audio: bytes) -> np.ndarray:
    samples = np.frombuffer(audio, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def _coerce_frames(prediction: object) -> np.ndarray:
    if hasattr(prediction, "detach"):
        prediction = prediction.detach()
    if hasattr(prediction, "cpu"):
        prediction = prediction.cpu()
    frames = np.asarray(prediction, dtype=np.float32)
    if frames.ndim == 3 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim != 2 or frames.shape[0] <= 0 or frames.shape[1] != DEFAULT_CHANNEL_COUNT:
        raise ValueError(f"expected [N,52] float32 prediction, got {tuple(frames.shape)}")
    if not np.isfinite(frames).all():
        raise ValueError("prediction contains non-finite values")
    return np.ascontiguousarray(frames, dtype=np.float32)


def _failure_response(
    error_code: str,
    error_message: str,
    request: SidecarRequest | None = None,
) -> bytes:
    return encode_response(
        SidecarResponse.failure(
            error_code=error_code,
            error_message=error_message,
            session_id=request.session_id if request is not None else None,
            turn_id=request.turn_id if request is not None else None,
            generation_epoch=request.generation_epoch if request is not None else None,
        )
    )


def _request_from_header_only(payload: bytes) -> SidecarRequest | None:
    try:
        header, audio = decode_message(payload)
    except Exception:
        return None

    session_id = header.get("session_id")
    turn_id = header.get("turn_id")
    generation_epoch = header.get("generation_epoch")
    sample_rate = header.get("sample_rate")
    channels = header.get("channels")
    dtype = header.get("dtype")
    num_samples = header.get("num_samples")
    fps = header.get("fps")
    if not isinstance(session_id, str) or not session_id:
        return None
    if not isinstance(turn_id, str) or not turn_id:
        return None
    if not isinstance(generation_epoch, int) or isinstance(generation_epoch, bool) or generation_epoch < 0:
        return None
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        return None
    if not isinstance(channels, int) or isinstance(channels, bool):
        return None
    if not isinstance(dtype, str) or not dtype:
        return None
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
        return None
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        return None
    return SidecarRequest(
        session_id=session_id,
        turn_id=turn_id,
        generation_epoch=generation_epoch,
        sample_rate=sample_rate,
        channels=channels,
        dtype=dtype,
        num_samples=num_samples,
        fps=fps,
        audio=audio,
    )


def _log(stderr: TextIO, message: str) -> None:
    stderr.write(f"{message}\n")
    stderr.flush()


def _log_exception(stderr: TextIO, prefix: str, exc: BaseException) -> None:
    _log(stderr, f"{prefix}: {exc}")
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=stderr)
    stderr.flush()


def handle_request_payload(
    payload: bytes,
    runner: Callable[[np.ndarray], object] | Callable[..., object],
    *,
    level: int,
    person: int,
    stderr: TextIO | None = None,
) -> bytes:
    request: SidecarRequest | None = None
    try:
        request = decode_request(payload)
        waveform = pcm16_bytes_to_float32_waveform(request.audio)
        prediction = runner(waveform, level=level, person=person)
        frames = _coerce_frames(prediction)
        response = SidecarResponse.success(
            session_id=request.session_id,
            turn_id=request.turn_id,
            generation_epoch=request.generation_epoch,
            frame_count=int(frames.shape[0]),
            frames=frames.tobytes(),
            fps=request.fps,
            channel_count=DEFAULT_CHANNEL_COUNT,
        )
        return encode_response(response)
    except SidecarProtocolError as exc:
        if request is None:
            request = _request_from_header_only(payload)
        if stderr is not None:
            _log(stderr, f"invalid_request: {exc}")
        return _failure_response("invalid_request", str(exc), request)
    except Exception as exc:
        if stderr is not None:
            _log_exception(stderr, "prediction_failed", exc)
        return _failure_response("prediction_failed", str(exc) or "prediction_failed", request)


def _load_real_runner(args: argparse.Namespace, stderr: TextIO) -> Callable[[np.ndarray], object]:
    emotalk_root = str(Path(args.emotalk_root).resolve())
    if emotalk_root not in sys.path:
        sys.path.insert(0, emotalk_root)

    with contextlib.redirect_stdout(stderr):
        torch = importlib.import_module("torch")
        export_module = importlib.import_module("scripts.export_blendshape_from_audio")
        if getattr(args, "torch_num_threads", None) is not None:
            torch.set_num_threads(args.torch_num_threads)
        if getattr(args, "torch_interop_threads", None) is not None:
            try:
                torch.set_num_interop_threads(args.torch_interop_threads)
            except RuntimeError:
                pass
        model_args = argparse.Namespace(
            wav_path="",
            out_path="",
            model_path=args.checkpoint,
            checkpoint=args.checkpoint,
            device=args.device,
            bs_dim=DEFAULT_CHANNEL_COUNT,
            feature_dim=832,
            period=args.fps,
            max_seq_len=5000,
            num_workers=0,
            batch_size=1,
            level=args.level,
            person=args.person,
            summary_csv=None,
            wav2vec_content_path=args.wav2vec_content_path,
            wav2vec_emotion_path=args.wav2vec_emotion_path,
        )
        model = export_module.load_model(model_args)

    inference_mode = getattr(torch, "inference_mode", torch.no_grad)

    def run(audio: np.ndarray, *, level: int, person: int) -> object:
        waveform = np.ascontiguousarray(audio, dtype=np.float32)
        with contextlib.redirect_stdout(stderr):
            with inference_mode():
                audio_tensor = torch.from_numpy(waveform).unsqueeze(0).to(args.device)
                level_tensor = torch.tensor([level]).to(args.device)
                person_tensor = torch.tensor([person]).to(args.device)
                return model.predict(audio_tensor, level_tensor, person_tensor)

    return run


def _warmup(runner: Callable[[np.ndarray], object], stderr: TextIO, *, level: int, person: int) -> None:
    _log(stderr, "warmup: start")
    silence = np.zeros(DEFAULT_SAMPLE_RATE, dtype=np.float32)
    frames = _coerce_frames(runner(silence, level=level, person=person))
    _log(stderr, f"warmup: ok frames={frames.shape[0]}")


def _read_next_request_payload(stdin) -> bytes | None:
    prefix = stdin.read(HEADER_PREFIX_SIZE)
    if prefix == b"":
        return None
    if len(prefix) < HEADER_PREFIX_SIZE:
        raise SidecarProtocolError("partial header prefix")

    header_len = struct.unpack(">I", prefix)[0]
    if header_len <= 0:
        raise SidecarProtocolError("header length must be positive")

    header_bytes = stdin.read(header_len)
    if len(header_bytes) < header_len:
        raise SidecarProtocolError("truncated header payload")

    header, _ = decode_message(prefix + header_bytes)
    num_samples = header.get("num_samples")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
        raise SidecarProtocolError("num_samples must be a positive integer")

    body_len = num_samples * 2
    body = bytearray()
    while len(body) < body_len:
        chunk = stdin.read(body_len - len(body))
        if chunk == b"":
            raise SidecarProtocolError("truncated message body")
        body.extend(chunk)

    return prefix + header_bytes + bytes(body)


def serve(
    stdin,
    stdout,
    stderr: TextIO,
    *,
    args: argparse.Namespace,
    runner_factory: Callable[[argparse.Namespace], Callable[[np.ndarray], object]] | None = None,
) -> int:
    factory = runner_factory or (lambda inner_args: _load_real_runner(inner_args, stderr))
    try:
        runner = factory(args)
        if args.warmup:
            _warmup(runner, stderr, level=args.level, person=args.person)
    except Exception as exc:
        _log_exception(stderr, "startup_failed", exc)
        return 1

    while True:
        try:
            payload = _read_next_request_payload(stdin)
        except SidecarProtocolError as exc:
            if str(exc) == "stream ended before header length":
                return 0
            _log_exception(stderr, "stream_failed", exc)
            return 1
        if payload is None:
            return 0

        response_payload = handle_request_payload(
            payload,
            runner,
            level=args.level,
            person=args.person,
            stderr=stderr,
        )
        stdout.write(response_payload)
        stdout.flush()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    exit_code = serve(
        sys.stdin.buffer,
        sys.stdout.buffer,
        sys.stderr,
        args=args,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
