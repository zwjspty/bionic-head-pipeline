import struct
import json

import numpy as np
import pytest

from bionic_head.sidecar_protocol import (
    PROTOCOL_VERSION,
    SidecarProtocolError,
    SidecarRequest,
    SidecarResponse,
    decode_message,
    decode_request,
    decode_response,
    encode_message,
    encode_request,
    encode_response,
    read_message,
    write_message,
)


def _request(
    *,
    session_id: str = "session",
    turn_id: str = "turn",
    generation_epoch: int = 3,
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: str = "int16",
    num_samples: int = 8,
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


class _ShortReadStream:
    def __init__(self, payload: bytes, read_size: int) -> None:
        self._payload = payload
        self._read_size = read_size
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""

        if size < 0:
            size = len(self._payload) - self._offset

        chunk = min(size, self._read_size)
        next_offset = self._offset + chunk
        chunk_data = self._payload[self._offset : next_offset]
        self._offset = next_offset
        return bytes(chunk_data)


def test_encode_decode_message_round_trip_preserves_large_raw_payload() -> None:
    header = {"protocol": PROTOCOL_VERSION, "session_id": "large"}
    body = b"\x00" * 70_000
    payload = encode_message(header, body)

    decoded_header, decoded_body = decode_message(payload)

    assert decoded_header == header
    assert decoded_body == body


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00\x00\x00",
        struct.pack(">I", 0),
        struct.pack(">I", 16) + b'{"protocol":"x"',
        struct.pack(">I", 1000) + b"{}",
        struct.pack(">I", 2) + b"[]",
        struct.pack(">I", 1) + b"{",
    ],
)
def test_decode_message_rejects_malformed(payload: bytes) -> None:
    with pytest.raises(SidecarProtocolError):
        decode_message(payload)


def test_message_prefix_is_big_endian_header_length() -> None:
    header = {"protocol": PROTOCOL_VERSION}
    body = b"abc"
    payload = encode_message(header, body)
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    expected_len = len(header_bytes)
    assert struct.unpack(">I", payload[:4])[0] == expected_len
    assert payload[4:] == header_bytes + body


def test_stream_helpers_round_trip() -> None:
    import io

    request = _request(num_samples=5)
    header = request.to_header()
    header["protocol"] = PROTOCOL_VERSION
    body = request.audio
    stream = io.BytesIO()

    write_message(stream, header, body)
    stream.seek(0)

    decoded_header, decoded_body = read_message(stream)
    assert decoded_header == header
    assert decoded_body == body


def test_read_message_reads_two_back_to_back_messages() -> None:
    import io

    request = _request(session_id="session-a", turn_id="turn-a", num_samples=4)
    response_frames = np.arange(52, dtype=np.float32).tobytes()
    response = SidecarResponse.success(
        session_id="session-b",
        turn_id="turn-b",
        generation_epoch=9,
        frame_count=1,
        frames=response_frames,
        fps=30,
        channel_count=52,
    )

    stream = io.BytesIO(encode_request(request) + encode_response(response))
    stream.seek(0)

    first_header, first_body = read_message(stream)
    second_header, second_body = read_message(stream)

    assert first_header == request.to_header()
    assert first_body == request.audio
    assert second_header == response.to_header()
    assert second_body == response_frames


def test_read_message_supports_short_reads_for_prefix_header_and_body() -> None:
    request = _request(session_id="short-read", turn_id="short-read", num_samples=4)
    response = SidecarResponse.success(
        session_id="resp-short-read",
        turn_id="resp-short-read",
        generation_epoch=12,
        frame_count=1,
        frames=np.arange(52, dtype=np.float32).tobytes(),
        fps=30,
        channel_count=52,
    )
    payload = encode_request(request) + encode_response(response)

    stream = _ShortReadStream(payload, read_size=2)

    first_header, first_body = read_message(stream)
    second_header, second_body = read_message(stream)

    assert first_header == request.to_header()
    assert first_body == request.audio
    assert second_header == response.to_header()
    assert second_body == response.frames


def test_request_round_trip_preserves_session_turn_epoch_and_pcm() -> None:
    expected = _request(
        session_id="sess",
        turn_id="turn",
        generation_epoch=42,
        sample_rate=16000,
        channels=1,
        dtype="int16",
        num_samples=6,
        fps=24,
        audio=b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c",
    )

    encoded = encode_request(expected)
    actual = decode_request(encoded)

    assert actual.session_id == "sess"
    assert actual.turn_id == "turn"
    assert actual.generation_epoch == 42
    assert actual.sample_rate == 16000
    assert actual.channels == 1
    assert actual.dtype == "int16"
    assert actual.num_samples == 6
    assert actual.fps == 24
    assert actual.audio == expected.audio


def test_response_success_round_trip_preserves_float32_frames() -> None:
    frames = np.arange(156, dtype=np.float32).reshape(3, 52)
    request = SidecarResponse.success(
        session_id="sess",
        turn_id="turn",
        generation_epoch=7,
        frame_count=3,
        frames=frames.tobytes(),
        fps=25,
        channel_count=52,
    )
    encoded = encode_response(request)
    decoded = decode_response(encoded)

    assert decoded.session_id == "sess"
    assert decoded.turn_id == "turn"
    assert decoded.generation_epoch == 7
    assert decoded.frame_count == 3
    assert decoded.fps == 25
    assert decoded.channel_count == 52
    assert decoded.frames == frames.tobytes()


def test_failure_response_round_trip_has_empty_body() -> None:
    request = SidecarResponse.failure(error_code="bad", error_message="not valid")
    encoded = encode_response(request)
    response = decode_response(encoded)

    assert response.ok is False
    assert response.error_code == "bad"
    assert response.error_message == "not valid"
    assert response.frames == b""
    assert response.to_header()["ok"] is False


@pytest.mark.parametrize(
    "header_overrides",
    [
        {"sample_rate": 8000},
        {"channels": 2},
        {"dtype": "float32"},
        {"num_samples": 8, "audio": b"\x00" * 2},
        {"fps": 0},
        {"generation_epoch": -1},
        {"session_id": ""},
        {"turn_id": ""},
    ],
)
def test_decode_request_rejects_invalid_fields(header_overrides: dict[str, object]) -> None:
    values = _request().__dict__.copy()
    values.update(header_overrides)
    request = SidecarRequest(
        session_id=values["session_id"],  # type: ignore[arg-type]
        turn_id=values["turn_id"],  # type: ignore[arg-type]
        generation_epoch=values["generation_epoch"],  # type: ignore[arg-type]
        sample_rate=values["sample_rate"],  # type: ignore[arg-type]
        channels=values["channels"],  # type: ignore[arg-type]
        dtype=values["dtype"],  # type: ignore[arg-type]
        num_samples=values["num_samples"],  # type: ignore[arg-type]
        fps=values["fps"],  # type: ignore[arg-type]
        audio=values["audio"],  # type: ignore[arg-type]
    )
    payload = encode_request(request)
    with pytest.raises(SidecarProtocolError):
        decode_request(payload)


def test_decode_request_rejects_bad_protocol() -> None:
    request = _request()
    header = request.to_header()
    header["protocol"] = "bad-protocol"
    payload = encode_message(header, request.audio)

    with pytest.raises(SidecarProtocolError):
        decode_request(payload)


@pytest.mark.parametrize(
    "header_overrides, body",
    [
        ({"dtype": "float64"}, b"\x00" * (3 * 52 * 4)),
        ({"channel_count": 51}, b"\x00" * (3 * 52 * 4)),
        ({"frame_count": 3}, b"\x00" * (2 * 52 * 4)),
    ],
)
def test_decode_response_rejects_invalid_success_payload(
    header_overrides: dict[str, object],
    body: bytes,
) -> None:
    header = {
        "ok": True,
        "protocol": PROTOCOL_VERSION,
        "session_id": "sess",
        "turn_id": "turn",
        "generation_epoch": 11,
        "frame_count": 3,
        "channel_count": 52,
        "dtype": "float32",
        "fps": 30,
    }
    header.update(header_overrides)
    payload = encode_message(header, body)
    with pytest.raises(SidecarProtocolError):
        decode_response(payload)


def test_decode_failure_response_rejects_missing_error_fields() -> None:
    for header in (
        {
            "ok": False,
            "protocol": PROTOCOL_VERSION,
            "error_code": "",
            "error_message": "not empty",
        },
        {
            "ok": False,
            "protocol": PROTOCOL_VERSION,
            "error_code": "bad",
            "error_message": "",
        },
    ):
        with pytest.raises(SidecarProtocolError):
            decode_response(encode_message(header, b""))
