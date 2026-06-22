from __future__ import annotations

from collections.abc import Iterator
import math

from pydantic import ValidationError

from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import UE5Frame, UE5Payload


CHANNELS = [f"morpheus_{index:02d}" for index in range(52)]


def _validation_error(message: str) -> PipelineException:
    return PipelineException(
        code=ErrorCode.OUTPUT_VALIDATION_FAILED,
        stage="ue5",
        provider=None,
        retryable=False,
        message=message,
    )


def build_ue5_payload(frames: list[list[float]], *, fps: int = 30) -> UE5Payload:
    if fps <= 0:
        raise _validation_error("UE5 fps must be positive")

    ue5_frames: list[UE5Frame] = []
    try:
        for frame_index, weights in enumerate(frames):
            if len(weights) != 52:
                raise ValueError("UE5 frame must contain exactly 52 weights")
            normalized = [float(weight) for weight in weights]
            if any(not math.isfinite(weight) for weight in normalized):
                raise ValueError("UE5 frame weights must be finite numbers")
            ue5_frames.append(
                UE5Frame(
                    frame_index=frame_index,
                    time_seconds=frame_index / float(fps),
                    weights=normalized,
                )
            )
        return UE5Payload(
            fps=fps,
            channels=CHANNELS.copy(),
            frame_count=len(ue5_frames),
            frames=ue5_frames,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise _validation_error(str(exc)) from exc


def chunk_ue5_frames(
    payload: UE5Payload,
    *,
    chunk_size: int = 30,
    chunk_id: str = "chunk",
) -> Iterator[dict[str, object]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    for chunk_index, start in enumerate(range(0, payload.frame_count, chunk_size)):
        end = min(start + chunk_size, payload.frame_count)
        frames = payload.frames[start:end]
        yield {
            "chunk_id": f"{chunk_id}-{chunk_index:04d}",
            "protocol": payload.protocol,
            "format": payload.format,
            "fps": payload.fps,
            "channel_count": payload.channel_count,
            "channels": payload.channels,
            "frame_count": len(frames),
            "start_frame_index": frames[0].frame_index if frames else None,
            "is_last": end >= payload.frame_count,
            "frames": [frame.model_dump(mode="json") for frame in frames],
        }
