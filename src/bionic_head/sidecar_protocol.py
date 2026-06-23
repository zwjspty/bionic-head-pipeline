from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO

PROTOCOL_VERSION: str = "emotalk-sidecar-v1"
HEADER_PREFIX_SIZE: int = 4


class SidecarProtocolError(ValueError):
    """Error parsing or validating EmoTalk sidecar binary protocol messages."""


def _validate_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SidecarProtocolError(f"{name} must be a positive integer")
    return value


def _validate_non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SidecarProtocolError(f"{name} must be a non-negative integer")
    return value


def _validate_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SidecarProtocolError(f"{name} must be a non-empty string")
    return value


@dataclass
class SidecarRequest:
    session_id: str
    turn_id: str
    generation_epoch: int
    sample_rate: int
    channels: int
    dtype: str
    num_samples: int
    fps: int
    audio: bytes

    def to_header(self) -> dict[str, object]:
        return {
            "protocol": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "generation_epoch": self.generation_epoch,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "dtype": self.dtype,
            "num_samples": self.num_samples,
            "fps": self.fps,
        }


@dataclass
class SidecarResponse:
    ok: bool
    session_id: str | None
    turn_id: str | None
    generation_epoch: int | None
    frame_count: int
    channel_count: int
    dtype: str
    fps: int
    frames: bytes
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def success(
        cls,
        session_id: str,
        turn_id: str,
        generation_epoch: int,
        frame_count: int,
        frames: bytes,
        *,
        fps: int = 30,
        channel_count: int = 52,
    ) -> "SidecarResponse":
        return cls(
            ok=True,
            session_id=session_id,
            turn_id=turn_id,
            generation_epoch=generation_epoch,
            frame_count=frame_count,
            channel_count=channel_count,
            dtype="float32",
            fps=fps,
            frames=frames,
            error_code=None,
            error_message=None,
        )

    @classmethod
    def failure(
        cls,
        error_code: str,
        error_message: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        generation_epoch: int | None = None,
    ) -> "SidecarResponse":
        return cls(
            ok=False,
            session_id=session_id,
            turn_id=turn_id,
            generation_epoch=generation_epoch,
            frame_count=0,
            channel_count=0,
            dtype="float32",
            fps=0,
            frames=b"",
            error_code=error_code,
            error_message=error_message,
        )

    def to_header(self) -> dict[str, object]:
        if self.ok:
            return {
                "ok": True,
                "protocol": PROTOCOL_VERSION,
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "generation_epoch": self.generation_epoch,
                "frame_count": self.frame_count,
                "channel_count": self.channel_count,
                "dtype": self.dtype,
                "fps": self.fps,
            }
        return {
            "ok": False,
            "protocol": PROTOCOL_VERSION,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "generation_epoch": self.generation_epoch,
        }


def encode_message(header: dict[str, object], body: bytes) -> bytes:
    encoded_header = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(encoded_header)) + encoded_header + body


def decode_message(payload: bytes) -> tuple[dict[str, object], bytes]:
    if len(payload) < HEADER_PREFIX_SIZE:
        raise SidecarProtocolError("payload shorter than 4-byte header prefix")
    header_len = struct.unpack(">I", payload[:HEADER_PREFIX_SIZE])[0]
    if header_len <= 0:
        raise SidecarProtocolError("header length must be positive")
    if len(payload) < HEADER_PREFIX_SIZE + header_len:
        raise SidecarProtocolError("truncated header payload")
    header_bytes = payload[HEADER_PREFIX_SIZE : HEADER_PREFIX_SIZE + header_len]
    body = payload[HEADER_PREFIX_SIZE + header_len :]
    try:
        raw_header = header_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SidecarProtocolError("header is not valid UTF-8") from exc
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise SidecarProtocolError("invalid JSON header") from exc
    if not isinstance(header, dict):
        raise SidecarProtocolError("header must be a JSON object")
    return header, body


def _require_type(header: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in header:
        raise SidecarProtocolError(f"missing required header field: {key}")
    value = header[key]
    if not isinstance(value, expected_type):
        raise SidecarProtocolError(f"header field {key} must be of type {expected_type.__name__}")
    return value


def decode_request(payload: bytes) -> SidecarRequest:
    header, audio = decode_message(payload)
    if header.get("protocol") != PROTOCOL_VERSION:
        raise SidecarProtocolError("invalid protocol")

    session_id = _validate_str(header.get("session_id"), "session_id")
    turn_id = _validate_str(header.get("turn_id"), "turn_id")
    generation_epoch = _validate_non_negative_int(header.get("generation_epoch"), "generation_epoch")
    sample_rate = _validate_positive_int(header.get("sample_rate"), "sample_rate")
    if sample_rate != 16000:
        raise SidecarProtocolError("sample_rate must be 16000")
    channels = _validate_positive_int(header.get("channels"), "channels")
    if channels != 1:
        raise SidecarProtocolError("channels must be 1")
    dtype = _validate_str(header.get("dtype"), "dtype")
    if dtype != "int16":
        raise SidecarProtocolError("dtype must be int16")
    num_samples = _validate_positive_int(header.get("num_samples"), "num_samples")
    fps = _validate_positive_int(header.get("fps"), "fps")

    expected_len = num_samples * 2
    if len(audio) != expected_len:
        raise SidecarProtocolError("audio body length does not match num_samples * 2")

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


def decode_response(payload: bytes) -> SidecarResponse:
    header, frames = decode_message(payload)
    if header.get("protocol") != PROTOCOL_VERSION:
        raise SidecarProtocolError("invalid protocol")
    ok = _require_type(header, "ok", bool)
    session_id = header.get("session_id")
    turn_id = header.get("turn_id")
    generation_epoch = header.get("generation_epoch")

    if session_id is not None and not isinstance(session_id, str):
        raise SidecarProtocolError("session_id must be string when provided")
    if session_id is not None and not session_id:
        raise SidecarProtocolError("session_id must be non-empty")
    if turn_id is not None and not isinstance(turn_id, str):
        raise SidecarProtocolError("turn_id must be string when provided")
    if turn_id is not None and not turn_id:
        raise SidecarProtocolError("turn_id must be non-empty")
    if generation_epoch is not None and (
        not isinstance(generation_epoch, int)
        or isinstance(generation_epoch, bool)
        or generation_epoch < 0
    ):
        raise SidecarProtocolError("generation_epoch must be non-negative integer when provided")

    if ok:
        frame_count = _validate_positive_int(header.get("frame_count"), "frame_count")
        generation_epoch = _validate_non_negative_int(
            header.get("generation_epoch"), "generation_epoch"
        )
        channel_count = _validate_positive_int(header.get("channel_count"), "channel_count")
        if channel_count != 52:
            raise SidecarProtocolError("channel_count must be 52")
        dtype = _validate_str(header.get("dtype"), "dtype")
        if dtype != "float32":
            raise SidecarProtocolError("dtype must be float32")
        fps = _validate_positive_int(header.get("fps"), "fps")
        expected_len = frame_count * channel_count * 4
        if len(frames) != expected_len:
            raise SidecarProtocolError("body length does not match frame_count * 52 * 4")
        return SidecarResponse(
            ok=True,
            session_id=_validate_str(session_id, "session_id"),
            turn_id=_validate_str(turn_id, "turn_id"),
            generation_epoch=generation_epoch,
            frame_count=frame_count,
            channel_count=channel_count,
            dtype=dtype,
            fps=fps,
            frames=frames,
            error_code=None,
            error_message=None,
        )

    error_code = header.get("error_code")
    error_message = header.get("error_message")
    if not isinstance(error_code, str) or not error_code:
        raise SidecarProtocolError("error_code must be non-empty string")
    if not isinstance(error_message, str) or not error_message:
        raise SidecarProtocolError("error_message must be non-empty string")
    if frames:
        raise SidecarProtocolError("failure response body must be empty")

    return SidecarResponse(
        ok=False,
        session_id=session_id if session_id is None else _validate_str(session_id, "session_id"),
        turn_id=turn_id if turn_id is None else _validate_str(turn_id, "turn_id"),
        generation_epoch=generation_epoch
        if isinstance(generation_epoch, int) and not isinstance(generation_epoch, bool)
        else None,
        frame_count=0,
        channel_count=0,
        dtype="float32",
        fps=0,
        frames=frames,
        error_code=error_code,
        error_message=error_message,
    )


def encode_request(request: SidecarRequest) -> bytes:
    return encode_message(request.to_header(), request.audio)


def encode_response(response: SidecarResponse) -> bytes:
    return encode_message(response.to_header(), response.frames)


def read_message(stream: BinaryIO) -> tuple[dict[str, object], bytes]:
    raw_prefix = stream.read(HEADER_PREFIX_SIZE)
    if not raw_prefix:
        raise SidecarProtocolError("stream ended before header length")
    if len(raw_prefix) < HEADER_PREFIX_SIZE:
        raise SidecarProtocolError("truncated header length prefix")
    header_len = struct.unpack(">I", raw_prefix)[0]
    if header_len <= 0:
        raise SidecarProtocolError("header length must be positive")
    header_bytes = stream.read(header_len)
    if len(header_bytes) < header_len:
        raise SidecarProtocolError("truncated header payload")
    return decode_message(raw_prefix + header_bytes + stream.read())


def write_message(stream: BinaryIO, header: dict[str, object], body: bytes) -> None:
    stream.write(encode_message(header, body))
