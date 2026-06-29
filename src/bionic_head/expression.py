from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ExpressionChannelMapping:
    format: str
    channel_count: int
    verified: bool
    channels: dict[str, int | None]
    groups: dict[str, list[int]]
    notes: dict[str, object]


def load_expression_channel_mapping(path: Path) -> ExpressionChannelMapping:
    body = json.loads(path.read_text(encoding="utf-8"))
    channel_count = int(body.get("channel_count", 52))
    channels = dict(body.get("channels") or {})
    groups = dict(body.get("groups") or {})

    for value in channels.values():
        if value is None:
            continue
        if not isinstance(value, int) or value < 0 or value >= channel_count:
            raise ValueError(f"channel index must be null or in [0, {channel_count - 1}]")

    normalized_groups: dict[str, list[int]] = {}
    for group_name, values in groups.items():
        if not isinstance(values, list):
            raise ValueError("group values must be lists")
        normalized: list[int] = []
        for value in values:
            if not isinstance(value, int) or value < 0 or value >= channel_count:
                raise ValueError(f"group channel index must be in [0, {channel_count - 1}]")
            normalized.append(value)
        normalized_groups[str(group_name)] = normalized

    return ExpressionChannelMapping(
        format=str(body.get("format", "")),
        channel_count=channel_count,
        verified=bool(body.get("verified", False)),
        channels={str(key): value for key, value in channels.items()},
        groups=normalized_groups,
        notes=dict(body.get("notes") or {}),
    )
