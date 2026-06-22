import json
from uuid import uuid4

import pytest

from bionic_head.core.artifacts import ArtifactStore


@pytest.mark.asyncio
async def test_stale_turn_cannot_publish_latest(tmp_path) -> None:
    store = ArtifactStore(tmp_path)

    async def reject(_action) -> bool:
        return False

    await store.publish_latest(
        pipeline={"turn": "old"},
        ue5={"frames": []},
        commit_if_current=reject,
    )
    assert not (tmp_path / "latest/latest_pipeline.json").exists()


def test_create_turn_creates_expected_subdirectories(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    session_id = uuid4()
    turn_id = uuid4()
    turn_dir = store.create_turn(session_id, turn_id)
    assert turn_dir == tmp_path / "runs" / str(session_id) / str(turn_id)
    assert (turn_dir / "tts").is_dir()
    assert (turn_dir / "face").is_dir()
    assert (turn_dir / "ue5").is_dir()


def test_write_json_uses_utf8_json(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    path = tmp_path / "nested" / "result.json"
    store.write_json(path, {"message": "你好"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"message": "你好"}


@pytest.mark.asyncio
async def test_current_turn_publishes_latest_atomically(tmp_path) -> None:
    store = ArtifactStore(tmp_path)

    async def accept(action) -> bool:
        action()
        return True

    published = await store.publish_latest(
        pipeline={"turn": "new"},
        ue5={"frames": [1]},
        commit_if_current=accept,
    )
    assert published is True
    assert json.loads((tmp_path / "latest/latest_pipeline.json").read_text()) == {"turn": "new"}
    assert json.loads((tmp_path / "latest/latest_ue5_blendshape.json").read_text()) == {"frames": [1]}
