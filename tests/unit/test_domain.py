from pathlib import Path
from uuid import uuid4

import pytest

from bionic_head.core.cancellation import CancellationToken
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import Emotion, FaceArtifact, TurnContext, UE5Frame, UE5Payload


def test_turn_context_keeps_identity_and_artifact_dir(tmp_path: Path) -> None:
    context = TurnContext(
        session_id=uuid4(),
        turn_id=uuid4(),
        artifact_dir=tmp_path,
        cancellation=CancellationToken(),
    )
    assert context.artifact_dir == tmp_path


def test_pipeline_exception_has_safe_shape() -> None:
    error = PipelineException(
        code=ErrorCode.PROVIDER_FAILED,
        stage="tts",
        provider="mock",
        retryable=False,
        message="TTS failed",
    )
    assert error.to_detail()["code"] == "provider_failed"
    assert Emotion.FRIENDLY.value == "friendly"


def test_error_code_uses_strenum_contract() -> None:
    assert ErrorCode.__bases__[0].__name__ == "StrEnum"
    assert ErrorCode.PROVIDER_FAILED == "provider_failed"


def test_face_artifact_rejects_non_52_value_frames() -> None:
    with pytest.raises(ValueError, match="52"):
        FaceArtifact(frames=[[0.0] * 51], frame_count=1)


def test_face_artifact_rejects_non_finite_weights() -> None:
    frame = [0.0] * 52
    frame[7] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        FaceArtifact(frames=[frame], frame_count=1)


def test_face_artifact_rejects_mismatched_frame_count() -> None:
    with pytest.raises(ValueError, match="frame_count"):
        FaceArtifact(frames=[[0.0] * 52], frame_count=2)


def test_ue5_payload_preserves_raw_52_channel_contract() -> None:
    payload = UE5Payload(
        channels=[f"morpheus_{index:02d}" for index in range(52)],
        frame_count=1,
        frames=[UE5Frame(frame_index=0, time_seconds=0.0, weights=[0.0] * 52)],
    )

    assert payload.protocol == "bionic-head-ue5-v1"
    assert payload.format == "morpheus_52_raw"
    assert payload.channel_count == 52


def test_ue5_payload_rejects_channel_count_mismatch() -> None:
    with pytest.raises(ValueError, match="52"):
        UE5Payload(
            channels=["morpheus_00"],
            frame_count=1,
            frames=[UE5Frame(frame_index=0, time_seconds=0.0, weights=[0.0] * 52)],
        )


def test_ue5_payload_rejects_mismatched_frame_count() -> None:
    with pytest.raises(ValueError, match="frame_count"):
        UE5Payload(
            channels=[f"morpheus_{index:02d}" for index in range(52)],
            frame_count=2,
            frames=[UE5Frame(frame_index=0, time_seconds=0.0, weights=[0.0] * 52)],
        )
