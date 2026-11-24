"""Test the quarter-step cells of the ordinary Fisher accuracy notebook.

Polynomial means supply exact derivatives without a physical solver. The tests
execute the notebook's actual array operations, including retention of already
computed stencil nodes and the two new predictions per parameter. No study
launcher, configuration adapter, native calculation or notebook run is needed.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from cosmolike_notebook_utils import fisher as cnu


# The teaching notebook owns these calculations; tests read its code directly.
NOTEBOOK = Path(__file__).resolve().parents[1] / "EXAMPLE_FISHER_ACCURACY.ipynb"


def notebook_cell(identifier):
    """Return the named code cell without executing other notebook sections.

    Arguments:
      identifier = stable notebook cell ID identifying the operation under test.
    Returns:
      Python source of that cell.
    Raises:
      AssertionError if the cell is absent or is not Python code.
    """
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["id"] == identifier:
            assert cell["cell_type"] == "code"
            return "".join(cell["source"])
    raise AssertionError("The accuracy notebook lacks cell " + identifier)


def polynomial_mean(parameters):
    """Return three observables with known cubic through quintic derivatives.

    Arguments:
      parameters = physical coordinate dictionary, including a negative and zero value.
    Returns:
      Three-element vector, with no files or physical model state changed.
    """
    x = parameters["omega_m"]
    y = parameters["h"]
    z = parameters["offset"]
    return np.array([
        x**3 + 2.0 * y + z,
        x**4 + y**3 + 3.0 * z,
        x**5 + y + z**3,
    ])


def retained_inputs():
    """Build the same cached half-step samples that the notebook retains.

    Binary-exact steps make reuse comparisons independent of decimal rounding.
    The returned cache has already evaluated every half-stencil coordinate.

    Returns:
      Namespace containing fiducial, steps, derivatives, center and mean cache.
    """
    fiducial = {"omega_m": 0.25, "h": -0.75, "offset": -0.0}
    steps = {"omega_m": 0.125, "h": 0.0625, "offset": 0.125}
    predict = cnu.ExactMeanCache(predict=polynomial_mean)
    names, center, first, second = cnu.differentiate(
        predict=predict,
        parameters=fiducial,
        steps=steps,
    )
    namespace = {
        "np": np,
        "cnu": cnu,
        "fiducial": fiducial,
        "steps": steps,
        "names": names,
        "predict": predict,
        "refined_center": center,
        "refined_h": first,
        "refined_h2": second,
    }
    # Execute only the retention assignments and loop from the full-refinement
    # cell. The surrounding Cobaya creation and physical derivatives stay paused.
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    retention = []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or "refined_stencils_h2 =" not in source:
            continue
        tree = ast.parse(source=source)
        for node in tree.body:
            if not isinstance(node, ast.Try):
                continue
            recording = False
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    name = ast.unparse(statement.targets[0])
                    if name == "refined_stencils_h2":
                        recording = True
                    if name == "projection_settings":
                        recording = False
                if recording:
                    retention.append(statement)
    assert retention  # Both original and half-step nodes are retained.
    before = len(predict.values)
    module = ast.Module(body=retention, type_ignores=[])
    exec(compile(module, filename=str(NOTEBOOK), mode="exec"), namespace)
    assert len(predict.values) == before
    return namespace


class Model:
    """Record closing of the model without opening any theory component."""

    def __init__(self):
        """Start with an open marker and no resources or native state."""
        self.closed = False

    def close(self):
        """Mark the test model closed after successful or failed stencil work."""
        self.closed = True


def test_quarter_step_uses_only_two_new_means_per_parameter():
    """Check exact reuse, signed coordinates and the known quintic stencil error.

    Returns:
      Nothing; all assertions use the notebook's actual quarter-step code.
    """
    namespace = retained_inputs()
    old_stencils = namespace["refined_stencils_h2"].copy()
    old_coordinates = namespace["refined_coordinates_h2"].copy()
    calls = []

    def get_dv(parameters):
        """Record only the new inner coordinates requested by the notebook.

        Arguments:
          parameters = one physical displacement about the fixed fiducial.
        Returns:
          Polynomial data vector at that point.
        """
        calls.append(dict(parameters))
        return polynomial_mean(parameters=parameters)

    namespace["get_dv"] = get_dv
    namespace["model"] = Model()
    exec(notebook_cell(identifier="fisher-quarter-compute"), namespace)
    assert namespace["model"].closed
    assert namespace["new_predictions"] == 2 * len(namespace["names"])
    assert len(calls) == namespace["new_predictions"]
    np.testing.assert_array_equal(x=namespace["refined_stencils_h2"], y=old_stencils)
    np.testing.assert_array_equal(x=namespace["refined_coordinates_h2"], y=old_coordinates)
    np.testing.assert_array_equal(
        x=namespace["quarter_stencils"][:, 0],
        y=old_stencils[:, 1],
    )
    np.testing.assert_array_equal(
        x=namespace["quarter_stencils"][:, 4],
        y=old_stencils[:, 3],
    )
    point = namespace["fiducial"]
    expected = np.array([
        [3.0 * point["omega_m"]**2, 2.0, 1.0],
        [4.0 * point["omega_m"]**3, 3.0 * point["h"]**2, 3.0],
        [5.0 * point["omega_m"]**4, 1.0, 0.0],
    ])
    # The five-point stencil for x**5 has error -4*s**4 at spacing s. The
    # shared stencil (cnu.five_point) sums in its own order, so the match
    # is exact up to round-off, amplified by the division by the small
    # quarter step (about 1e-13 here).
    expected[2, 0] -= 4.0 * (namespace["steps"]["omega_m"] / 4.0)**4
    np.testing.assert_allclose(actual=namespace["quarter_derivatives"], desired=expected,
                               rtol=1.0e-12, atol=1.0e-12 * np.max(np.abs(expected)))
    assert np.signbit(namespace["fiducial"]["offset"])
    for column, name in enumerate(namespace["names"]):
        step = namespace["steps"][name] / 4.0
        assert calls[2 * column][name] == point[name] - step
        assert calls[2 * column + 1][name] == point[name] + step


@pytest.mark.parametrize("failure", ["coordinates", "derivative", "center", "collapsed"])
def test_quarter_step_refuses_inconsistent_retained_values(failure):
    """A mismatch in the first parameter stops before either new prediction.

    Arguments:
      failure = retained quantity or too-small displacement deliberately changed.
    Returns:
      Nothing; checks refusal and model cleanup without a native calculation.
    """
    namespace = retained_inputs()
    if failure == "coordinates":
        namespace["refined_coordinates_h2"][0, 0] += 0.001
    elif failure == "derivative":
        namespace["refined_h2"][0, 0] += 0.001
    elif failure == "center":
        namespace["refined_stencils_h2"][0, 2, 0] += 0.001
    else:
        namespace["steps"]["omega_m"] = 1.0e-20
    calls = []

    def get_dv(parameters):
        """Record an unwanted evaluation if an invalid stencil reaches it.

        Arguments:
          parameters = coordinates that must not be requested in this test.
        Returns:
          Polynomial vector so an unexpected call is independently countable.
        """
        calls.append(dict(parameters))
        return polynomial_mean(parameters=parameters)

    namespace["get_dv"] = get_dv
    namespace["model"] = Model()
    with pytest.raises((AssertionError, ValueError)):
        exec(notebook_cell(identifier="fisher-quarter-compute"), namespace)
    assert calls == []
    assert namespace["model"].closed


def test_full_comparison_reconstructs_cross_information_and_single_prior():
    """The array-only notebook comparison agrees with an independent contraction.

    Nonzero cross covariance makes the conditional information necessary. The
    two-parameter example names dark energy so the same FoM display also runs.

    Returns:
      Nothing; checks each matrix case and preserves the input derivative arrays.
    """
    covariance = np.array([[2.0, 0.1, 0.6], [0.1, 1.0, -0.2], [0.6, -0.2, 1.5]])
    derivatives = np.array([[1.0, 0.0], [0.2, 1.0], [0.5, 1.2]])
    namespace = {
        "np": np,
        "cnu": cnu,
        "names": ["w0", "wa"],
        "row_kind": np.array(["xip", "xim", "map3"]),
        "covariance": covariance,
        "priors": {"wa": 0.2},
        "dh2": derivatives.copy(),
        "refined_h": 1.1 * derivatives,
        "refined_h2": 1.2 * derivatives,
        "selected": [0],
    }
    exec(notebook_cell(identifier="fisher-full-array-comparison"), namespace)
    for label, matrix in namespace["derivative_cases"].items():
        expected = matrix.T @ np.linalg.solve(covariance, matrix)
        result = namespace["full_comparisons"][label]
        np.testing.assert_allclose(
            actual=result["fisher_joint"],
            desired=expected + np.diag([0.0, 25.0]),
        )
    np.testing.assert_array_equal(x=namespace["dh2"], y=derivatives)
    np.testing.assert_array_equal(x=namespace["partial"][:, 1], y=derivatives[:, 1])


def test_wider_step_reuses_original_outer_nodes():
    """The two selected wider stencils need only their new outer pair.

    The quintic term has a known fourth-order error. Its larger-step result
    distinguishes correct spacing 2h from accidentally reusing the h divisor.
    """
    namespace = retained_inputs()
    calls = []

    def get_dv(parameters):
        """Evaluate the polynomial and count only the new physical coordinates."""
        calls.append(parameters.copy())
        return polynomial_mean(parameters=parameters)

    namespace["get_dv"] = get_dv
    namespace["model"] = Model()
    exec(notebook_cell(identifier="fisher-wide-compute"), namespace)
    assert len(calls) == 4
    assert namespace["wide_predictions"] == 4
    assert namespace["model"].closed
    x = namespace["fiducial"]["omega_m"]
    y = namespace["fiducial"]["h"]
    wider_step = 2 * namespace["steps"]["omega_m"]
    expected = np.array([
        [3 * x**2, 2.0],
        [4 * x**3, 3 * y**2],
        [5 * x**4 - 4 * wider_step**4, 1.0],
    ])
    np.testing.assert_allclose(namespace["wide_derivatives"][:, :2], expected)
    # The unselected column stays at the fine reference spacing.
    np.testing.assert_array_equal(
        namespace["wide_derivatives"][:, 2],
        namespace["refined_h2"][:, 2],
    )
    for name in ("omega_m", "h"):
        column = namespace["names"].index(name)
        np.testing.assert_array_equal(
            namespace["wide_stencils"][name][[1, 2, 3]],
            namespace["refined_stencils_h"][column][[0, 2, 4]],
        )


def test_projection_and_block_cells_keep_the_full_covariance():
    """Independent inverses verify QR, SVD, weak directions and row substitutions.

    This small matrix includes cross covariance and two nuisance parameters.
    The same full covariance is used even when just one observable block's
    derivative entries change; treating those blocks as independent would fail.
    """
    from scipy.linalg import solve_triangular

    random = np.random.default_rng(seed=2026)
    root = random.normal(size=(8, 8))
    covariance = root @ root.T + np.eye(8)
    derivatives = random.normal(size=(8, 4))
    namespace = {
        "np": np,
        "cnu": cnu,
        "solve_triangular": solve_triangular,
        "covariance": covariance,
        "names": ["w0", "wa", "omega_m", "h"],
        "priors": {"omega_m": 0.3},
        "row_kind": np.array(["xip", "xim"] * 2 + ["map3"] * 4),
        "dh2": derivatives.copy(),
        "refined_h2": derivatives + 0.03 * random.normal(size=(8, 4)),
        "refined_h": derivatives + 0.05 * random.normal(size=(8, 4)),
    }
    exec(notebook_cell(identifier="fisher-projection-metrics"), namespace)
    for label, response in namespace["augmented"].items():
        full_inverse = np.linalg.inv(response.T @ response)
        expected = np.linalg.inv(full_inverse[:2, :2])
        np.testing.assert_allclose(
            namespace["effective_information"][label], expected,
            rtol=1e-12, atol=1e-12,
        )
        metrics = namespace["projection_metrics"][label]
        np.testing.assert_allclose(metrics["fom_qr"], metrics["fom_inverse"])
        np.testing.assert_allclose(metrics["fom_svd"], metrics["fom_inverse"])
    exec(notebook_cell(identifier="fisher-weak-direction"), namespace)
    old = namespace["effective_information"]["default_half"]
    new = namespace["effective_information"]["refined_half"]
    np.testing.assert_allclose(
        new @ namespace["directions"],
        old @ namespace["directions"] * namespace["ratios"][None, :],
    )
    # The fitted nuisance shifts must leave a response orthogonal to their span.
    np.testing.assert_allclose(
        namespace["current"][:, 2:].T @ namespace["response"],
        np.zeros(2), atol=1e-14,
    )
    exec(notebook_cell(identifier="fisher-block-substitutions"), namespace)
    fine = namespace["refined_h2"]
    changed = fine.copy()
    changed[:4, 2] = namespace["refined_h"][:4, 2]
    prior = np.diag([0.0, 0.0, 1 / 0.3**2, 0.0])
    baseline = fine.T @ np.linalg.solve(covariance, fine) + prior
    replacement = changed.T @ np.linalg.solve(covariance, changed) + prior
    ratio = np.sqrt(
        np.linalg.det(np.linalg.inv(baseline)[:2, :2])
        / np.linalg.det(np.linalg.inv(replacement)[:2, :2])
    )
    np.testing.assert_allclose(
        namespace["block_substitutions"]["step:omega_m:xi"], ratio,
    )


def test_study_cells_bind_to_shared_public_apis():
    """Bind every shared Fisher and Cobaya call without evaluating its arguments.

    This detects an obsolete keyword in a newly added expensive cell while
    keeping physical predictions and complete notebook execution paused.
    """
    import inspect
    from cobaya.model import get_model

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(
            line for line in "".join(cell["source"]).splitlines()
            if not line.startswith("%")
        )
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            function = None
            if isinstance(node.func, ast.Name) and node.func.id == "get_model":
                function = get_model
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "cnu"):
                function = getattr(cnu, node.func.attr)
            if function is not None:
                assert all(keyword.arg is not None for keyword in node.keywords)
                inspect.signature(function).bind(
                    *[None for _ in node.args],
                    **{keyword.arg: None for keyword in node.keywords},
                )
