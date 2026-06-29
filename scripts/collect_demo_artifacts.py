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

from bionic_head.client.demo_artifacts import collect_latest_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect demo acceptance artifacts.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--http-base-url")
    parser.add_argument("--data-latest-dir", type=Path, default=Path("data/latest"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = collect_latest_artifacts(
        output_dir=args.output_dir,
        http_base_url=args.http_base_url,
        data_latest_dir=args.data_latest_dir,
    )
    print(json.dumps({"artifacts": artifacts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
