from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json

from bionic_head.domain.models import Emotion


@dataclass(frozen=True)
class ExpressionChannelMapping:
    format: str
    channel_count: int
    verified: bool
    channels: dict[str, int | None]
    groups: dict[str, list[int]]
    notes: dict[str, object]


@dataclass(frozen=True)
class ExpressionMetrics:
    enabled: bool
    applied: bool
    expression_emotion: str
    expression_intensity: float
    profile_channel_count: int
    expression_max_delta: float
    warning_count: int = 0

    def to_timing_payload(self) -> dict[str, bool | float | str]:
        return {
            "expression_enabled": self.enabled,
            "expression_applied": self.applied,
            "expression_emotion": self.expression_emotion,
            "expression_intensity": self.expression_intensity,
            "expression_profile_channel_count": float(self.profile_channel_count),
            "expression_max_delta": self.expression_max_delta,
            "expression_warning_count": float(self.warning_count),
        }


class ExpressionPostProcessor:
    def __init__(
        self,
        *,
        enabled: bool = False,
        mapping: ExpressionChannelMapping | None = None,
        profiles: dict[str, dict[str, float]] | None = None,
        max_delta: float = 0.3,
    ) -> None:
        if max_delta < 0.0:
            raise ValueError("max_delta must be non-negative")
        self.enabled = enabled
        self.mapping = mapping
        self.profiles = profiles or {}
        self.max_delta = float(max_delta)

    def process(
        self,
        frames: list[list[float]],
        *,
        emotion: Emotion | str,
        intensity: float,
    ) -> tuple[list[list[float]], ExpressionMetrics]:
        copied = copy.deepcopy(frames)
        emotion_value = str(emotion.value if isinstance(emotion, Emotion) else emotion)
        intensity_value = max(0.0, min(1.0, float(intensity)))

        if not self.enabled or self.mapping is None or not self.mapping.verified:
            return copied, ExpressionMetrics(
                enabled=self.enabled,
                applied=False,
                expression_emotion=emotion_value,
                expression_intensity=intensity_value,
                profile_channel_count=0,
                expression_max_delta=0.0,
                warning_count=0,
            )

        profile = self.profiles.get(emotion_value, {})
        channel_deltas: dict[int, float] = {}
        warning_count = 0
        for semantic_name, configured_delta in profile.items():
            channel_index = self.mapping.channels.get(semantic_name)
            if channel_index is None:
                warning_count += 1
                continue
            delta = max(-self.max_delta, min(self.max_delta, float(configured_delta) * intensity_value))
            if delta == 0.0:
                continue
            channel_deltas[channel_index] = channel_deltas.get(channel_index, 0.0) + delta

        for channel_index, delta in list(channel_deltas.items()):
            channel_deltas[channel_index] = max(-self.max_delta, min(self.max_delta, delta))

        max_applied_delta = 0.0
        if channel_deltas:
            _validate_frames(copied, self.mapping.channel_count)
            for frame in copied:
                for channel_index, delta in channel_deltas.items():
                    frame[channel_index] = float(frame[channel_index]) + delta
                    max_applied_delta = max(max_applied_delta, abs(delta))

        return copied, ExpressionMetrics(
            enabled=True,
            applied=bool(channel_deltas),
            expression_emotion=emotion_value,
            expression_intensity=intensity_value,
            profile_channel_count=len(channel_deltas),
            expression_max_delta=max_applied_delta,
            warning_count=warning_count,
        )


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


def _validate_frames(frames: list[list[float]], channel_count: int) -> None:
    for frame in frames:
        if len(frame) != channel_count:
            raise ValueError(f"expression frames must contain exactly {channel_count} channels")
