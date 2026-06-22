from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import json

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse

from bionic_head.api.dependencies import AppContainer, get_container
from bionic_head.core.cancellation import CancellationToken
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import TurnContext


router = APIRouter()


@router.post("/pipeline/audio")
async def run_audio_pipeline(
    audio: UploadFile = File(...),
    container: AppContainer = Depends(get_container),
) -> dict[str, object]:
    session_id = uuid4()
    turn_id = uuid4()
    async with container.sessions.activate(session_id, turn_id):
        turn_dir = container.store.create_turn(session_id, turn_id)
        upload_path = turn_dir / "upload.wav"
        payload = await audio.read()
        if not payload:
            raise PipelineException(
                code=ErrorCode.INVALID_REQUEST,
                stage="request",
                provider=None,
                retryable=False,
                message="Uploaded audio is empty",
            )
        upload_path.write_bytes(payload)

        context = TurnContext(
            session_id=session_id,
            turn_id=turn_id,
            artifact_dir=turn_dir,
            cancellation=CancellationToken(),
        )
        orchestrator = container.make_offline_orchestrator()
        result = await orchestrator.run(upload_path, context)
    return result.model_dump(mode="json")


@router.get("/pipeline/latest")
def latest_pipeline(container: AppContainer = Depends(get_container)) -> object:
    return _read_latest(container.store.latest / "latest_pipeline.json", stage="latest")


@router.get("/ue5/latest")
def latest_ue5(container: AppContainer = Depends(get_container)) -> object:
    return _read_latest(container.store.latest / "latest_ue5_blendshape.json", stage="ue5")


def _read_latest(path: Path, *, stage: str) -> object:
    if not path.exists():
        return _not_found(stage, "Latest result does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def _not_found(stage: str, message: str) -> JSONResponse:
    error = PipelineException(
        code=ErrorCode.INVALID_REQUEST,
        stage=stage,
        provider=None,
        retryable=False,
        message=message,
    )
    return JSONResponse(status_code=404, content={"error": error.to_detail()})
