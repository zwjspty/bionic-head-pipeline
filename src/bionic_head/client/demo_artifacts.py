from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from bionic_head.client.demo_acceptance import write_json


def http_get_json(url: str, timeout_sec: float = 5.0) -> tuple[bool, object | None, str | None]:
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return True, payload, None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return False, None, str(exc)


def collect_latest_artifacts(
    *,
    output_dir: Path,
    http_base_url: str | None,
    data_latest_dir: Path | None,
    timeout_sec: float = 5.0,
) -> dict[str, str]:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    if data_latest_dir is not None:
        local_sources = {
            "latest_pipeline": data_latest_dir / "latest_pipeline.json",
            "latest_ue5": data_latest_dir / "latest_ue5_blendshape.json",
        }
        for name, source in local_sources.items():
            if source.exists():
                destination = artifact_dir / source.name
                shutil.copy2(source, destination)
                artifacts[name] = destination.relative_to(output_dir).as_posix()

    if http_base_url:
        http_sources = {
            "latest_pipeline": {
                "path": "/pipeline/latest",
                "destination": artifact_dir / "latest_pipeline.json",
            },
            "latest_ue5": {
                "path": "/ue5/latest",
                "destination": artifact_dir / "latest_ue5_blendshape.json",
            },
        }
        for name, config in http_sources.items():
            ok, payload, _ = http_get_json(http_base_url.rstrip("/") + config["path"], timeout_sec=timeout_sec)
            if ok and payload is not None:
                destination = config["destination"]
                write_json(destination, payload)
                artifacts[name] = destination.relative_to(output_dir).as_posix()
    return artifacts


def collect_existing_artifacts(output_dir: Path, names: Mapping[str, Path]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for name, path in names.items():
        if path.exists() and path.is_file():
            artifacts[name] = path.relative_to(output_dir).as_posix()
    return artifacts
