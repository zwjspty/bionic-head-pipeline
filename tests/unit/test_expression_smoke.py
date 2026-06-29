import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.expression_smoke import DEFAULT_EXPRESSION_PROFILES, write_expression_variants


def test_expression_smoke_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/expression_smoke.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "expression visual smoke" in result.stdout
    assert "--base-face" in result.stdout


def test_write_expression_variants_writes_npy_and_report_without_render(tmp_path: Path) -> None:
    base_face = tmp_path / "base-face.npy"
    np.save(base_face, np.zeros((2, 52), dtype=np.float32))

    report = write_expression_variants(
        base_face=base_face,
        output_dir=tmp_path / "smoke",
        emotions=["neutral", "happy"],
        mapping_path=Path("config/expression_channels.example.json"),
        profiles=DEFAULT_EXPRESSION_PROFILES,
        max_delta=0.3,
    )

    neutral = np.load(tmp_path / "smoke" / "neutral.npy")
    happy = np.load(tmp_path / "smoke" / "happy.npy")
    saved_report = json.loads((tmp_path / "smoke" / "report.json").read_text(encoding="utf-8"))

    assert neutral.shape == (2, 52)
    assert happy.shape == (2, 52)
    assert neutral[0, 43] == 0.0
    assert happy[0, 43] > neutral[0, 43]
    assert report["variants"]["happy"]["metrics"]["expression_applied"] is True
    assert saved_report["variants"]["happy"]["face_npy"].endswith("happy.npy")
