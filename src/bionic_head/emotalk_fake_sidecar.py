from __future__ import annotations

import argparse
import math
import struct
import sys
from typing import BinaryIO

import numpy as np

from .sidecar_protocol import (
    HEADER_PREFIX_SIZE,
    SidecarProtocolError,
    SidecarResponse,
    decode_message,
    decode_request,
    encode_response,
)


def build_fake_frames(num_samples: int, sample_rate: int, fps: int) -> tuple[int, bytes]:
    """
    Return deterministic fake blendshape bytes for the requested duration.
    """
    duration_seconds = num_samples / sample_rate
    frame_count = int(math.ceil(duration_seconds * fps))
    frames = np.zeros((frame_count, 52), dtype=np.float32)
    return frame_count, frames.tobytes()


def handle_request_payload(payload: bytes) -> bytes:
    """Decode a request payload and return an encoded response payload."""
    try:
        request = decode_request(payload)
        frame_count, frames = build_fake_frames(
            request.num_samples,
            request.sample_rate,
            request.fps,
        )
        response = SidecarResponse.success(
            session_id=request.session_id,
            turn_id=request.turn_id,
            generation_epoch=request.generation_epoch,
            frame_count=frame_count,
            frames=frames,
            fps=request.fps,
            channel_count=52,
        )
    except Exception as exc:
        message = str(exc) if str(exc) else "invalid_request"
        response = SidecarResponse.failure(
            error_code="invalid_request",
            error_message=message,
        )

    return encode_response(response)


def _read_next_request_payload(stdin: BinaryIO) -> bytes | None:
    """
    Read one raw protocol payload, returning None on clean EOF.
    """
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
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        raise SidecarProtocolError("num_samples must be a positive integer")
    if num_samples <= 0:
        raise SidecarProtocolError("num_samples must be a positive integer")

    body_len = num_samples * 2
    body = bytearray()
    while len(body) < body_len:
        chunk = stdin.read(body_len - len(body))
        if chunk == b"":
            raise SidecarProtocolError("truncated message body")
        body.extend(chunk)
    return prefix + header_bytes + bytes(body)


def _stream_messages(
    stdin: BinaryIO,
    stdout: BinaryIO,
) -> int:
    while True:
        try:
            payload = _read_next_request_payload(stdin)
        except SidecarProtocolError as exc:
            message = str(exc)
            if message == "stream ended before header length":
                return 0
            raise

        if payload is None:
            return 0

        response = handle_request_payload(payload)
        stdout.write(response)
        stdout.flush()


def serve(stdin: BinaryIO, stdout: BinaryIO) -> int:
    try:
        return _stream_messages(stdin, stdout)
    except Exception:
        return 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m bionic_head.emotalk_fake_sidecar",
        description="Emulates an EmoTalk sidecar server using fake deterministic frames.",
    )
    return parser.parse_args(argv)


def main() -> None:
    _parse_args()
    exit_code = serve(sys.stdin.buffer, sys.stdout.buffer)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
