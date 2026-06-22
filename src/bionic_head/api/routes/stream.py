from __future__ import annotations

from fastapi import APIRouter, WebSocket

from bionic_head.protocol.connection import StreamConnection


router = APIRouter()


@router.websocket("/pipeline/stream")
async def pipeline_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    container = websocket.app.state.container
    await StreamConnection(websocket, container).run()
