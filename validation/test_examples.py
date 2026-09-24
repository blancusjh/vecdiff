"""Execute the supported scientific examples using their public APIs."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "package,script,result_file",
    [
        ("general", "planar_interface.py", "transmitted.npz"),
        ("stigmatic", "focal_field.py", "focal_field.npz"),
    ],
)
def test_complete_example(package, script, result_file, tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / package / "examples" / script),
            "--output",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((tmp_path / "report.json").read_text())
    if package == "general":
        assert report["relative_power_balance"] < 1e-11
        assert max(report["boundary_residuals"].values()) < 1e-11
    else:
        assert report["relative_ray_power_balance"] < 1e-12
        assert report["relative_optical_path_variation"] < 1e-12
    with np.load(tmp_path / result_file) as data:
        assert data["electric_V_per_m"].shape == data["points_m"].shape
        assert data["magnetic_A_per_m"].shape == data["points_m"].shape
        assert np.isfinite(data["electric_V_per_m"]).all()
