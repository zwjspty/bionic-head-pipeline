from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bionic_head.ue5_playback_contract import UE5PlaybackContractError, replay_ue5_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay UE5 frame events and report playback receiver metrics."
    )
    parser.add_argument("path", type=Path, help="JSON event list or single event")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = replay_file(args.path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["success"] else 1)


def replay_file(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        events = document if isinstance(document, list) else [document]
        metrics = replay_ue5_events(events)
        return {"success": True, "metrics": metrics}
    except (OSError, json.JSONDecodeError, UE5PlaybackContractError) as exc:
        return {"success": False, "error": str(exc), "metrics": {}}


if __name__ == "__main__":
    main()
