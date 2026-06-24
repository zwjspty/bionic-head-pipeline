from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID
import asyncio
import inspect

from fastapi import Request

from bionic_head.adapters.registry import AdapterRegistry, build_registry
from bionic_head.config import AppSettings
from bionic_head.core.artifacts import ArtifactStore
from bionic_head.domain.errors import ErrorCode, PipelineException
from bionic_head.domain.models import DiagnosticResult
from bionic_head.orchestrators.offline import CommitCallback, OfflineOrchestrator
from bionic_head.orchestrators.stream import StreamOrchestrator


class SessionManager:
    def __init__(self, *, max_active_sessions: int) -> None:
        self.max_active_sessions = max_active_sessions
        self._active_sessions: set[UUID] = set()
        self._latest_turns: dict[UUID, UUID] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def admit(self, session_id: UUID) -> AsyncIterator[None]:
        async with self._lock:
            if session_id not in self._active_sessions and len(self._active_sessions) >= self.max_active_sessions:
                raise PipelineException(
                    code=ErrorCode.SESSION_LIMIT_REACHED,
                    stage="session",
                    provider=None,
                    retryable=True,
                    message="Session limit reached",
                )
            self._active_sessions.add(session_id)

        try:
            yield
        finally:
            async with self._lock:
                self._active_sessions.discard(session_id)

    @asynccontextmanager
    async def activate(self, session_id: UUID, turn_id: UUID) -> AsyncIterator[None]:
        async with self._lock:
            if session_id not in self._active_sessions and len(self._active_sessions) >= self.max_active_sessions:
                raise PipelineException(
                    code=ErrorCode.SESSION_LIMIT_REACHED,
                    stage="session",
                    provider=None,
                    retryable=True,
                    message="Session limit reached",
                )
            self._active_sessions.add(session_id)
            self._latest_turns[session_id] = turn_id

        try:
            yield
        finally:
            async with self._lock:
                self._active_sessions.discard(session_id)

    async def commit_if_current(
        self,
        session_id: UUID,
        turn_id: UUID,
        callback: CommitCallback,
    ) -> bool:
        async with self._lock:
            if self._latest_turns.get(session_id) != turn_id:
                return False
            callback()
            return True


@dataclass
class AppContainer:
    settings: AppSettings
    registry: AdapterRegistry
    store: ArtifactStore
    sessions: SessionManager

    @classmethod
    def create(cls, settings: AppSettings) -> "AppContainer":
        return cls(
            settings=settings,
            registry=build_registry(settings),
            store=ArtifactStore(settings.storage.root),
            sessions=SessionManager(max_active_sessions=settings.server.max_active_sessions),
        )

    def make_offline_orchestrator(self) -> OfflineOrchestrator:
        return OfflineOrchestrator(
            settings=self.settings,
            registry=self.registry,
            store=self.store,
            commit_if_current=self.sessions.commit_if_current,
        )

    def make_stream_orchestrator(self) -> StreamOrchestrator:
        return StreamOrchestrator(
            settings=self.settings,
            registry=self.registry,
            store=self.store,
        )

    async def prewarm(self) -> list[DiagnosticResult]:
        results: list[DiagnosticResult] = []
        if (
            self.settings.adapters.llm.provider == "ollama"
            and self.settings.providers.ollama.prewarm
        ):
            started = perf_counter()
            try:
                results.append(await self.registry.llm.prewarm())  # type: ignore[attr-defined]
            except PipelineException as exc:
                results.append(
                    DiagnosticResult(
                        adapter="llm",
                        provider=exc.provider or self.registry.llm.name,
                        available=False,
                        latency_ms=(perf_counter() - started) * 1000.0,
                        message=exc.safe_message,
                    )
                )

        if (
            self.settings.adapters.audio2face.provider == "emotalk_sidecar"
            and self.settings.providers.emotalk_sidecar.prewarm_on_startup
        ):
            result = await self.prewarm_audio2face()
            if result is not None:
                results.append(result)

        return results

    async def prewarm_audio2face(self) -> DiagnosticResult | None:
        if self.settings.adapters.audio2face.provider != "emotalk_sidecar":
            return None

        started = perf_counter()
        try:
            return await self.registry.audio2face.prewarm()  # type: ignore[attr-defined]
        except PipelineException as exc:
            if self.settings.providers.emotalk_sidecar.prewarm_required:
                raise
            return DiagnosticResult(
                adapter="audio2face",
                provider=exc.provider or self.registry.audio2face.name,
                available=False,
                latency_ms=(perf_counter() - started) * 1000.0,
                message=exc.safe_message,
            )

    async def close(self) -> None:
        for name in ("asr", "llm", "tts", "audio2face", "ue5"):
            adapter = getattr(self.registry, name, None)
            if adapter is None:
                continue
            close = getattr(adapter, "close", None)
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await result


def get_container(request: Request) -> AppContainer:
    return request.app.state.container
