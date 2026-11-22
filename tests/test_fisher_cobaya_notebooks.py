"""Check the direct Cobaya notebook path without running a physical forecast.

The notebooks must use the same likelihood prediction as the numbered examples.
A small polynomial stand-in supplies that prediction here, so the tests can
exercise the actual notebook function and the shared derivative operations.
No CAMB, emulator, native projection or notebook calculation is launched.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cosmolike_notebook_utils import fisher as cnu


PROJECT = Path(__file__).resolve().parents[1]
# These notebooks teach different physics with the same ordinary prediction call.
NOTEBOOKS = (
    "EXAMPLE_FISHER_IA.ipynb",
    "EXAMPLE_FISHER_ACCURACY.ipynb",
    "EXAMPLE_FISHER_BACCO.ipynb",
)


def notebook_cells(filename):
    """Read code cells, removing only the IPython display directives.

    Arguments:
      filename = notebook basename in the project directory.
    Returns:
      List of executable Python source strings; no cell is evaluated here.
    """
    notebook = json.loads((PROJECT / filename).read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = []
        for line in "".join(cell["source"]).splitlines():
            if not line.startswith("%"):
                lines.append(line)
        cells.append("\n".join(lines))
    return cells


@pytest.mark.parametrize("filename", NOTEBOOKS)
def test_cells_use_the_numbered_cobaya_path(filename):
    """Reject a return to a separate forecast driver or workflow class.

    Arguments:
      filename = one of the public Fisher notebook examples.
    Returns:
      Nothing; assertions check the code without initializing any theory.
    """
    cells = notebook_cells(filename=filename)
    source = "\n".join(cells)
    for cell in cells:
        ast.parse(source=cell)
    for old_layer in (
        "FisherWorkflow",
        "AccuracyStudy",
        "run_3pcf",
        "run_fisher_accuracy",
        "cosmolike_des_3pcf_fisher_wrappers",
        "cosmolike_des_3pcf_bacco_wrappers",
        "cosmolike_des_3pcf_accuracy_wrappers",
    ):
        assert old_layer not in source
    assert "yaml_load_file" in source
    assert "get_model(info_or_yaml_or_file=" in source
    assert "cnu.differentiate(" in source
    assert "model.close()" in source


@pytest.mark.parametrize("filename", NOTEBOOKS)
def test_notebook_prediction_and_absolute_derivatives(filename):
    """Differentiate a negative and zero fiducial through the actual notebook call.

    The expected derivative follows analytically from a cubic polynomial.
    The fake interface has a nontrivial row selector, so this also checks that
    derivatives use the selected vector rather than the likelihood's full array.

    Arguments:
      filename = notebook containing the get_dv function under test.
    Returns:
      Nothing; checks both finite-difference steps and Cobaya cache arguments.
    """
    likelihood = SimpleNamespace(last_datavector=None)
    calls = []

    class Model:
        """Supply deterministic likelihood values without any physical solver."""

        def loglike(self, params_values, return_derived, cached):
            """Record the exact Cobaya arguments and update the retained vector.

            Arguments:
              params_values = two physical coordinates, a1 and wa.
              return_derived, cached = flags passed by the notebook.
            Returns:
              Finite scalar likelihood, independent of the stand-in data units.
            """
            calls.append((return_derived, cached))
            x = params_values["a1"]
            y = params_values["wa"]
            likelihood.last_datavector = np.array([
                x + 3.0 * y,
                123.0,
                x**3 + 2.0 * y,
            ])
            return -1.0

    class Interface:
        """Select two rows, as the native dataset selector does in production."""

        @staticmethod
        def select_joint_data_vector(prediction):
            """Return the selected two-row vector from the three-row input.

            Arguments:
              prediction = full mean retained by the likelihood.
            Returns:
              First and third mean entries, in covariance order.
            """
            return prediction[[0, 2]]

    namespace = {
        "np": np,
        "model": Model(),
        "likelihood": likelihood,
        "ci": Interface,
    }
    functions = []
    for cell in notebook_cells(filename=filename):
        for node in ast.parse(source=cell).body:
            if isinstance(node, ast.FunctionDef) and node.name == "get_dv":
                functions.append(node)
    assert len(functions) == 1
    definition = ast.Module(body=functions, type_ignores=[])
    exec(compile(definition, filename=filename, mode="exec"), namespace)
    predict = namespace["get_dv"]
    point = {"a1": -0.33, "wa": 0.0}
    steps = {"a1": 0.01, "wa": 0.04}
    names, center, first, second = cnu.differentiate(
        predict=predict,
        parameters=point,
        steps=steps,
    )
    expected = np.array([[1.0, 3.0], [3.0 * point["a1"]**2, 2.0]])
    assert names == ["a1", "wa"]
    np.testing.assert_allclose(actual=first, desired=expected, atol=2.0e-14)
    np.testing.assert_allclose(actual=second, desired=expected, atol=2.0e-14)
    np.testing.assert_allclose(actual=center, desired=[-0.33, -0.33**3])
    assert all(flags == (False, False) for flags in calls)
    assert point == {"a1": -0.33, "wa": 0.0}

    # The accuracy notebook supplies offsets about zero to the shared noise
    # function. Its absolute displacement must remain 0.01 even at a1=-0.33.
    offsets = cnu.make_offset_predictor(
        predict=predict,
        reference=point,
        names=names,
    )
    noise = cnu.get_stencil_noise(
        dv=offsets,
        index=0,
        h=steps["a1"],
        CV=np.zeros(2),
        invcov=np.eye(2),
    )
    assert noise["eps"] == steps["a1"]
    np.testing.assert_allclose(actual=noise["ddv"], desired=expected[:, 0])
    assert noise["sigma"] < 1.0e-14


def test_joint_information_and_prior_are_counted_once():
    """Verify the notebook's shared Schur decomposition with an explicit matrix.

    A nonzero cross covariance changes the information supplied by the third
    row. Adding its raw information independently would fail this test.

    Returns:
      Nothing; checks likelihood addition and the single parameter prior.
    """
    covariance = np.array([
        [2.0, 0.1, 0.6],
        [0.1, 1.0, -0.2],
        [0.6, -0.2, 1.5],
    ])
    derivatives = np.array([[1.0, 0.0], [0.2, 1.0], [0.5, 1.2]])
    result = cnu.compare_information(
        derivatives=derivatives,
        names=["a1", "m1"],
        kind=np.array(["xip", "xim", "map3"]),
        covariance=covariance,
        priors={"m1": 0.2},
    )
    expected = derivatives.T @ np.linalg.solve(covariance, derivatives)
    np.testing.assert_allclose(actual=result["fisher_joint_likelihood"], desired=expected)
    np.testing.assert_allclose(
        actual=result["fisher_joint_likelihood"],
        desired=result["fisher_two_point_likelihood"]
        + result["fisher_conditional_three_point"],
    )
    np.testing.assert_allclose(
        actual=result["fisher_joint"],
        desired=expected + np.diag([0.0, 25.0]),
    )
