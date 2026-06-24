from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bionic_head.api.app import create_app
from bionic_head.domain.errors import ErrorCode, PipelineException


class _FailingStartupContainer:
    def __init__(self) -> None:
        self.closed = False

    async def prewarm(self):
        raise PipelineException(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            stage="audio2face.prewarm",
            provider="emotalk_sidecar",
            retryable=True,
            message="startup prewarm failed",
        )

    async def close(self) -> None:
        self.closed = True


def test_lifespan_closes_container_when_startup_prewarm_fails(
    mock_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _FailingStartupContainer()
    monkeypatch.setattr(
        "bionic_head.api.app.AppContainer.create",
        lambda _settings: container,
    )

    with pytest.raises(PipelineException):
        with TestClient(create_app(mock_settings)):
            pass

    assert container.closed is True
