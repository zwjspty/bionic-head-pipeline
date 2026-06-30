from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bionic_head.ue5_playback_contract import (
    UE5PlaybackContractError,
    validate_playback_stop,
    validate_ue5_frame_chunk,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate UE5 playback contract JSON payloads or event logs."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="JSON files to validate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_files(args.paths)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["success"] else 1)


def validate_files(paths: Iterable[Path]) -> dict[str, object]:
    validated_count = 0
    failures: list[dict[str, object]] = []

    for path in paths:
        try:
            document = _read_json(path)
            for index, payload in enumerate(_iter_payloads(document)):
                try:
                    _validate_payload(payload)
                    validated_count += 1
                except UE5PlaybackContractError as exc:
                    failures.append({"path": str(path), "index": index, "error": str(exc)})
        except Exception as exc:
            failures.append({"path": str(path), "index": None, "error": str(exc)})

    return {
        "success": not failures,
        "validated_count": validated_count,
        "failure_count": len(failures),
        "failures": failures,
    }


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_payloads(document: object) -> Iterable[Mapping[str, object]]:
    if isinstance(document, list):
        for item in document:
            yield _payload_from_document(item)
    else:
        yield _payload_from_document(document)


def _payload_from_document(document: object) -> Mapping[str, object]:
    if not isinstance(document, Mapping):
        raise UE5PlaybackContractError("JSON document must be an object or list of objects")
    payload = document.get("payload", document)
    if not isinstance(payload, Mapping):
        raise UE5PlaybackContractError("event payload must be an object")
    if "type" in document and "type" not in payload:
        return {"type": document["type"], **payload}
    return payload


def _validate_payload(payload: Mapping[str, object]) -> None:
    event_type = payload.get("type")
    if event_type == "server.playback.stop":
        validate_playback_stop(payload)
        return
    validate_ue5_frame_chunk(payload)


if __name__ == "__main__":
    main()
