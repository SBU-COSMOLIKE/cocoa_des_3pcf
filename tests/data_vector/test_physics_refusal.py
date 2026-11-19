"""Refuse unsupported geometry and labels before lossy conversions."""

from des3pcf.variance import reference_physics_model

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def invalid_physics_input(case):
    """Call a fatal boundary in a child process, where exit(1) is isolated."""
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "interface"))
    import cosmolike_des_3pcf_interface as compiled
    from test_physics_model import physical_inputs

    compiled.set_omp_threads(nthreads=1)
    inputs = physical_inputs()
    if case == "missing_variance":
        compiled.PhysicsModel(**inputs)
    if case == "curved":
        inputs["omega_de"] = 0.65
    model = reference_physics_model(factory=compiled.PhysicsModel, **inputs)
    sides = np.asarray([[10.0, 15.0, 20.0]], dtype=np.float64)
    if case == "underflow":
        sides[0] = [np.nextafter(0.0, 1.0), 1.0e10, 1.0e10]
        model.matter(sides=sides, redshifts=np.asarray([0.0]))
    elif case == "fractional":
        model.projected_notebook(
            sides=sides, bins=np.asarray([0.9, 1.1, 0.2])
        )
    elif case == "wide_signed":
        model.projected_notebook(
            sides=sides, bins=np.asarray([-(2**32), 1, 0], dtype=np.int64)
        )
    elif case == "wide_unsigned":
        model.projected_notebook(
            sides=sides, bins=np.asarray([2**32, 1, 0], dtype=np.uint64)
        )
    elif case == "fractional_matrix":
        model.projection_grids_notebook(
            bins=np.asarray([[0.9, 1.1, 0.2]])
        )
    print("Invalid physical input returned")


@pytest.mark.parametrize(
    "case, expected",
    [
        ("missing_variance", "internal BiHalofit requires supplied variance arrays"),
        (
            "curved",
            "flat geometry requires abs(omega_m + omega_de - 1) <= 1e-8",
        ),
        ("underflow", "triangle side ratio underflows double precision"),
        ("fractional", "notebook source bins must have integer dtype"),
        ("fractional_matrix", "notebook source bins must have integer dtype"),
        ("wide_signed", "notebook source bins exceed int32 range"),
        ("wide_unsigned", "notebook source bins exceed int32 range"),
    ],
)
def test_invalid_physics_is_refused(case, expected):
    """Require the intended C diagnostic, not an import or loader failure."""
    result = subprocess.run(
        args=[sys.executable, "-B", str(Path(__file__)), "--case", case],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    message = result.stdout + result.stderr
    assert result.returncode == 1
    assert expected in message
    assert "Traceback" not in message
    assert "runtime error:" not in message
    assert "Invalid physical input returned" not in message


def test_notebook_keeps_strided_int64_labels(compiled):
    """Integer dtype conversion preserves labels and negative strides."""
    from test_physics_model import physical_inputs

    model = reference_physics_model(factory=compiled.PhysicsModel, **physical_inputs())
    sides = np.asarray([[10.0, 15.0, 20.0]], dtype=np.float64)
    original = np.asarray([0, 1, 1], dtype=np.int64)
    bins = original[::-1]
    assert bins.strides[0] < 0
    expected = model.projected(sides=sides, bins=bins.astype(np.int32))
    actual = model.projected_notebook(sides=sides, bins=bins)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(original, np.asarray([0, 1, 1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    arguments = parser.parse_args()
    invalid_physics_input(case=arguments.case)
