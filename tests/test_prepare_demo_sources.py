"""Test the source-processing cells students run in EXAMPLE_EVALUATE2.

The small calculation now lives in the notebook. These checks execute its
actual cells, compare hand-integrated areas and reproduce the bundled source
samples. They do not run its Cobaya model, Fisher derivatives or any maps.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "EXAMPLE_EVALUATE2.ipynb"


def run_cell(identity, namespace):
    """Execute one identified teaching cell with explicitly supplied arrays.

    Arguments:
      identity = persistent cell ID in the notebook.
      namespace = test inputs and prior cell results.
    Returns:
      nothing; results remain in namespace as in a notebook kernel.
    """
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("id") == identity:
            source = "".join(cell["source"])
            exec(compile(source, str(NOTEBOOK) + ":" + identity, "exec"), namespace)
            return
    raise AssertionError("Missing source-processing cell: " + identity)


def toy_inputs():
    """Return an irregular midpoint grid whose first bin has area 1.325.

    Its positive trapezoidal area is 1.375 and its negative magnitude is
    0.05. The sample at z=0 is negative, making clipping observable.
    """
    return {
        "np": np,
        "redshift": np.array([0.0, 0.5, 1.5, 3.0]),
        "densities": np.array([[-0.2, 1.0, 0.5, 0.0]]),
    }


def test_notebook_clipping_accounting_and_input_preservation():
    """Check all three areas independently and retain the original arrays."""
    state = toy_inputs()
    original = state["densities"].copy()
    run_cell(identity="source-normalize", namespace=state)
    np.testing.assert_allclose(state["signed_area"], [1.325])
    np.testing.assert_allclose(state["positive_area"], [1.375])
    np.testing.assert_allclose(state["negative_area"], [0.05])
    np.testing.assert_allclose(state["normalized"], [[0.0, 1.0/1.375, 0.5/1.375, 0.0]])
    np.testing.assert_array_equal(state["densities"], original)


@pytest.mark.parametrize("problem", [
    "nonfinite_z", "negative_z", "repeated_z", "reversed_z",
    "nonfinite_density", "zero_area", "negative_area", "wrong_shape",
])
def test_notebook_refuses_invalid_source_inputs(problem):
    """Clipping must not conceal invalid coordinates or unusable curves."""
    state = toy_inputs()
    if problem == "nonfinite_z":
        state["redshift"][1] = np.inf
    elif problem == "negative_z":
        state["redshift"][0] = -0.1
    elif problem == "repeated_z":
        state["redshift"][1] = 0.0
    elif problem == "reversed_z":
        state["redshift"] = state["redshift"][::-1]
    elif problem == "nonfinite_density":
        state["densities"][0, 1] = np.nan
    elif problem == "zero_area":
        state["densities"][:] = 0.0
    elif problem == "negative_area":
        state["densities"][:] = -1.0
    else:
        state["densities"] = np.zeros(shape=(1, 3))
    with pytest.raises(ValueError):
        run_cell(identity="source-normalize", namespace=state)


def test_notebook_reproduces_bundled_source_samples():
    """Run the read, calculation and comparison cells on the included FITS."""
    state = {"np": np, "fits": fits, "project": PROJECT}
    run_cell(identity="source-read", namespace=state)
    original = state["source_path"].read_bytes()
    run_cell(identity="source-normalize", namespace=state)
    run_cell(identity="source-verify", namespace=state)
    assert np.count_nonzero(state["densities"][0] < 0.0) == 12
    assert state["source_path"].read_bytes() == original
