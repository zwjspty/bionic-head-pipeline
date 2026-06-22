from __future__ import annotations

from pathlib import Path
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from bionic_head.api.dependencies import AppContainer
from bionic_head.api.routes import health, pipeline, stream
from bionic_head.config import AppSettings, load_settings
from bionic_head.domain.errors import ErrorCode, PipelineException


ERROR_STATUS = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.INVALID_AUDIO_FORMAT: 415,
    ErrorCode.NO_SPEECH_DETECTED: 422,
    ErrorCode.SESSION_LIMIT_REACHED: 429,
    ErrorCode.PROVIDER_UNAVAILABLE: 503,
    ErrorCode.PROVIDER_TIMEOUT: 504,
    ErrorCode.PROVIDER_FAILED: 502,
    ErrorCode.OUTPUT_VALIDATION_FAILED: 502,
    ErrorCode.TURN_CANCELLED: 499,
    ErrorCode.INTERNAL_ERROR: 500,
}


def create_app(settings: AppSettings | None = None) -> FastAPI:
    config_path = Path(os.environ.get("BIONIC_CONFIG", "config/mock.json"))
    resolved_settings = settings or load_settings(config_path)

    app = FastAPI(title="Bionic Head Pipeline", version="0.1.0")
    app.state.container = AppContainer.create(resolved_settings)
    app.add_exception_handler(PipelineException, pipeline_exception_handler)
    app.include_router(health.router)
    app.include_router(pipeline.router)
    app.include_router(stream.router)
    return app


async def pipeline_exception_handler(request: Request, exc: PipelineException) -> JSONResponse:
    return JSONResponse(
        status_code=ERROR_STATUS.get(exc.code, 500),
        content={"error": exc.to_detail()},
    )
