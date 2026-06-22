from pathlib import Path
from uuid import uuid4

from bionic_head.core.cancellation import CancellationToken
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import Emotion, TurnContext


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
