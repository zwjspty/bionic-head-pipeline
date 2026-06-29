from pathlib import Path
import json

import pytest

from bionic_head.expression import load_expression_channel_mapping


def test_expression_channel_example_is_parseable_and_unverified() -> None:
    mapping = load_expression_channel_mapping(Path("config/expression_channels.example.json"))

    assert mapping.format == "morpheus_52_raw"
    assert mapping.channel_count == 52
    assert mapping.verified is False
    assert mapping.channels["jaw_open"] is None
    assert mapping.groups["mouth"] == []


def test_expression_channel_mapping_rejects_channel_index_outside_52(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "format": "morpheus_52_raw",
                "channel_count": 52,
                "verified": True,
                "channels": {"jaw_open": 52},
                "groups": {},
                "notes": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"channel index must be null or in \[0, 51\]"):
        load_expression_channel_mapping(path)


def test_expression_channel_mapping_rejects_group_index_outside_52(tmp_path: Path) -> None:
    path = tmp_path / "bad-group.json"
    path.write_text(
        json.dumps(
            {
                "format": "morpheus_52_raw",
                "channel_count": 52,
                "verified": True,
                "channels": {"jaw_open": 1},
                "groups": {"jaw": [1, 99]},
                "notes": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"group channel index must be in \[0, 51\]"):
        load_expression_channel_mapping(path)
