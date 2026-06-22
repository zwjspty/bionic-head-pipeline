from __future__ import annotations

import asyncio
from time import perf_counter
from uuid import UUID

from bionic_head.core.ue5 import build_ue5_payload
from bionic_head.domain.models import DiagnosticResult, FaceArtifact, TurnContext, UE5Payload


class MorpheusRawUE5Adapter:
    name = "morpheus-raw"
    call_count = 0

    async def format(self, face: FaceArtifact, context: TurnContext) -> UE5Payload:
        self.call_count += 1
        context.cancellation.raise_if_cancelled()
        return build_ue5_payload(face.frames, fps=face.fps)

    async def diagnostics(self) -> DiagnosticResult:
        started = perf_counter()
        return DiagnosticResult(
            adapter="ue5",
            provider=self.name,
            available=True,
            latency_ms=(perf_counter() - started) * 1000.0,
            message="morpheus_52_raw formatter ready",
        )

    async def cancel(self, turn_id: UUID) -> None:
        await asyncio.sleep(0)
