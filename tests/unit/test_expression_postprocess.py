from pathlib import Path

import pytest

from bionic_head.domain.models import Emotion
from bionic_head.expression import ExpressionPostProcessor, load_expression_channel_mapping


def _frames(value: float = 0.0, *, frame_count: int = 3) -> list[list[float]]:
    return [[value for _ in range(52)] for _ in range(frame_count)]


def test_expression_postprocessor_is_noop_when_disabled() -> None:
    mapping = load_expression_channel_mapping(Path("config/expression_channels.example.json"))
    processor = ExpressionPostProcessor(
        enabled=False,
        mapping=mapping,
        profiles={"happy": {"mouth_smile_left": 0.2}},
        max_delta=0.3,
    )

    output, metrics = processor.process(_frames(), emotion=Emotion.HAPPY, intensity=1.0)

    assert output == _frames()
    assert metrics.enabled is False
    assert metrics.applied is False
    assert metrics.profile_channel_count == 0


def test_expression_postprocessor_is_noop_when_mapping_is_unverified() -> None:
    verified_mapping = load_expression_channel_mapping(Path("config/expression_channels.example.json"))
    unverified_mapping = verified_mapping.__class__(
        format=verified_mapping.format,
        channel_count=verified_mapping.channel_count,
        verified=False,
        channels=verified_mapping.channels,
        groups=verified_mapping.groups,
        notes=verified_mapping.notes,
    )
    processor = ExpressionPostProcessor(
        enabled=True,
        mapping=unverified_mapping,
        profiles={"happy": {"mouth_smile_left": 0.2}},
        max_delta=0.3,
    )

    output, metrics = processor.process(_frames(), emotion=Emotion.HAPPY, intensity=1.0)

    assert output == _frames()
    assert metrics.enabled is True
    assert metrics.applied is False
    assert metrics.warning_count == 0


def test_expression_postprocessor_applies_verified_profile_to_mapped_channels() -> None:
    mapping = load_expression_channel_mapping(Path("config/expression_channels.example.json"))
    processor = ExpressionPostProcessor(
        enabled=True,
        mapping=mapping,
        profiles={"happy": {"mouth_smile_left": 0.2, "mouth_smile_right": 0.1}},
        max_delta=0.3,
    )

    output, metrics = processor.process(_frames(), emotion=Emotion.HAPPY, intensity=0.5)

    assert len(output) == 3
    assert len(output[0]) == 52
    assert output[0][43] == pytest.approx(0.1)
    assert output[0][44] == pytest.approx(0.05)
    assert output[1][43] == pytest.approx(0.1)
    assert metrics.applied is True
    assert metrics.expression_emotion == "happy"
    assert metrics.expression_intensity == pytest.approx(0.5)
    assert metrics.profile_channel_count == 2
    assert metrics.expression_max_delta == pytest.approx(0.1)
    assert metrics.warning_count == 0


def test_expression_postprocessor_caps_per_channel_delta() -> None:
    mapping = load_expression_channel_mapping(Path("config/expression_channels.example.json"))
    processor = ExpressionPostProcessor(
        enabled=True,
        mapping=mapping,
        profiles={"surprised": {"jaw_open": 0.9}},
        max_delta=0.25,
    )

    output, metrics = processor.process(_frames(), emotion=Emotion.SURPRISED, intensity=1.0)

    assert output[0][24] == pytest.approx(0.25)
    assert metrics.expression_max_delta == pytest.approx(0.25)


def test_expression_postprocessor_warns_and_skips_unknown_or_unmapped_channels() -> None:
    mapping = load_expression_channel_mapping(Path("config/expression_channels.example.json"))
    processor = ExpressionPostProcessor(
        enabled=True,
        mapping=mapping,
        profiles={"friendly": {"unknown_channel": 0.2, "eye_blink_left": 0.2}},
        max_delta=0.3,
    )

    output, metrics = processor.process(_frames(), emotion=Emotion.FRIENDLY, intensity=1.0)

    assert output == _frames()
    assert metrics.applied is False
    assert metrics.profile_channel_count == 0
    assert metrics.warning_count == 2
