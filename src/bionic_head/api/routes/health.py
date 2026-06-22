from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from bionic_head.adapters.protocols import (
    ASRAdapter,
    Audio2FaceAdapter,
    LLMAdapter,
    TTSAdapter,
    UE5Adapter,
)
from bionic_head.api.dependencies import AppContainer, get_container
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import DiagnosticResult


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/diagnostics")
async def diagnostics(container: AppContainer = Depends(get_container)) -> dict[str, object]:
    adapters = _adapters(container)
    results = await asyncio.gather(
        *(_safe_diagnostics(name, adapter) for name, adapter in adapters.items()),
    )
    return dict(results)


@router.get("/diagnostics/{adapter_name}")
async def adapter_diagnostics(
    adapter_name: str,
    container: AppContainer = Depends(get_container),
) -> object:
    adapters = _adapters(container)
    adapter = adapters.get(adapter_name)
    if adapter is None:
        return _not_found("diagnostics", f"Unknown adapter: {adapter_name}")
    _, result = await _safe_diagnostics(adapter_name, adapter)
    return result


def _adapters(
    container: AppContainer,
) -> dict[str, ASRAdapter | LLMAdapter | TTSAdapter | Audio2FaceAdapter | UE5Adapter]:
    return {
        "asr": container.registry.asr,
        "llm": container.registry.llm,
        "tts": container.registry.tts,
        "audio2face": container.registry.audio2face,
        "ue5": container.registry.ue5,
    }


async def _safe_diagnostics(
    name: str,
    adapter: ASRAdapter | LLMAdapter | TTSAdapter | Audio2FaceAdapter | UE5Adapter,
) -> tuple[str, dict[str, object]]:
    try:
        result = await adapter.diagnostics()
        return name, result.model_dump(mode="json")
    except PipelineException as exc:
        return name, _unavailable_diagnostic(name, adapter.name, exc.safe_message)
    except Exception:
        return name, _unavailable_diagnostic(name, adapter.name, "Provider diagnostics failed")


def _unavailable_diagnostic(adapter: str, provider: str, message: str) -> dict[str, object]:
    return DiagnosticResult(
        adapter=adapter,
        provider=provider,
        available=False,
        latency_ms=0,
        message=message,
    ).model_dump(mode="json")


def _not_found(stage: str, message: str) -> JSONResponse:
    error = PipelineException(
        code=ErrorCode.INVALID_REQUEST,
        stage=stage,
        provider=None,
        retryable=False,
        message=message,
    )
    return JSONResponse(status_code=404, content={"error": error.to_detail()})
