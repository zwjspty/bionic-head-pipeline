from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID
import json
import os
import tempfile


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs = root / "runs"
        self.latest = root / "latest"

    def create_turn(self, session_id: UUID, turn_id: UUID) -> Path:
        turn_dir = self.runs / str(session_id) / str(turn_id)
        for relative in ("tts", "face", "ue5"):
            (turn_dir / relative).mkdir(parents=True, exist_ok=True)
        return turn_dir

    def write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._atomic_write(path, encoded)

    async def publish_latest(
        self,
        *,
        pipeline: object,
        ue5: object,
        commit_if_current: Callable[[Callable[[], None]], Awaitable[bool]],
    ) -> bool:
        self.latest.mkdir(parents=True, exist_ok=True)
        pipeline_path = self.latest / "latest_pipeline.json"
        ue5_path = self.latest / "latest_ue5_blendshape.json"
        pipeline_tmp = self._write_temp(pipeline_path, pipeline)
        ue5_tmp = self._write_temp(ue5_path, ue5)

        def commit() -> None:
            os.replace(pipeline_tmp, pipeline_path)
            os.replace(ue5_tmp, ue5_path)

        published = await commit_if_current(commit)
        if not published:
            pipeline_tmp.unlink(missing_ok=True)
            ue5_tmp.unlink(missing_ok=True)
        return published

    def _atomic_write(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)

    def _write_temp(self, target: Path, payload: object) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            handle.write(encoded)
            return Path(handle.name)
