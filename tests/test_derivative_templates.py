"""Check covariance-weighted degeneracy without a physical model call."""

import ast
import importlib

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pytest

from test_fisher_cobaya_notebooks import notebook_cells


shared = importlib.import_module("cosmolike_notebook_utils.3pcf.derivative_templates")


def test_exact_two_point_degeneracy_has_measurable_aperture_residual():
    """The first two rows are proportional; the third distinguishes them."""
    result = shared.compare_templates(
        target=[1.0, 2.0, 3.0], mimic=[2.0, 4.0, 5.0],
        covariance=np.eye(3), two_point=np.array([True, True, False]),
    )
    assert result["two_point"]["coefficient"] == 0.5
    np.testing.assert_allclose(result["two_point"]["residual"], [0.0, 0.0, 0.5])
    assert result["two_point"]["residual_conditional_norm2"] == 0.25
    np.testing.assert_allclose(result["joint"]["coefficient"], 5.0 / 9.0)
    np.testing.assert_allclose(result["joint"]["residual_joint_norm2"], 1.0 / 9.0)
    np.testing.assert_allclose(result["joint"]["reproduced_fraction"], 125.0 / 126.0)


def test_correlations_units_and_row_permutations():
    """A covariance fit differs from a diagonal-error fit and survives units."""
    covariance = np.array([
        [2.0, 0.4, 0.3],
        [0.4, 1.0, -0.2],
        [0.3, -0.2, 1.5],
    ])
    target = np.array([1.0, 2.0, 3.0])
    mimic = np.array([2.0, 3.0, -1.0])
    second = np.array([True, True, False])
    baseline = shared.compare_templates(
        target=target, mimic=mimic, covariance=covariance, two_point=second,
    )
    inverse = np.linalg.inv(covariance)
    coefficient = (mimic @ inverse @ target) / (mimic @ inverse @ mimic)
    np.testing.assert_allclose(baseline["joint"]["coefficient"], coefficient)
    diagonal_coefficient = np.sum(mimic * target / np.diag(covariance))
    diagonal_coefficient /= np.sum(mimic**2 / np.diag(covariance))
    assert abs(coefficient - diagonal_coefficient) > 0.01
    units = np.array([1.0, 1.0, 1.e-8])
    order = np.array([2, 0, 1])
    scaled = covariance * units[:, None] * units[None, :]
    changed = shared.compare_templates(
        target=(target * units)[order], mimic=(mimic * units)[order],
        covariance=scaled[np.ix_(order, order)], two_point=second[order],
    )
    for name in ("two_point", "joint"):
        for key in ("cosine", "coefficient", "residual_joint_norm2"):
            np.testing.assert_allclose(changed[name][key], baseline[name][key])
        np.testing.assert_allclose(
            baseline[name]["residual_joint_norm2"],
            baseline[name]["residual_two_point_norm2"]
            + baseline[name]["residual_conditional_norm2"],
        )
    np.testing.assert_allclose(changed["templates"], baseline["templates"][:, order])


def test_opposite_signs_remain_degenerate():
    """A mirror image has negative overlap but reproduced fraction one."""
    result = shared.compare_templates(
        target=[1.0, -2.0, 3.0], mimic=[-2.0, 4.0, -6.0],
        covariance=np.eye(3), two_point=np.array([True, True, False]),
    )
    np.testing.assert_allclose(result["joint"]["cosine"], -1.0)
    np.testing.assert_allclose(result["joint"]["residual"], 0.0, atol=1.e-14)


@pytest.mark.parametrize("fault", ["zero", "nonpositive", "nonfinite", "mask"])
def test_bad_scientific_inputs_are_refused(fault):
    target = np.array([1.0, 2.0, 3.0])
    covariance = np.eye(3)
    second = np.array([True, True, False])
    if fault == "zero":
        target[:] = 0.0
    elif fault == "nonpositive":
        covariance[0, 1] = covariance[1, 0] = 2.0
    elif fault == "nonfinite":
        target[0] = np.nan
    else:
        second[:] = True
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        shared.compare_templates(
            target=target, mimic=[2.0, 4.0, 5.0],
            covariance=covariance, two_point=second,
        )


def test_named_columns_scales_and_display_are_preserved():
    """Stored column order and display intervals determine the drawn response."""
    result = {
        "parameter_names": np.array(["DES_A2_1", "DES_A1_1"]),
        "derivatives_h2": np.array([[2., 1.], [4., 2.], [5., 3.]]),
        "row_kind": np.array(["xip", "xim", "map3"]),
        "row_labels": np.array(["first", "second", "third"]),
        "fiducial_covariance": np.eye(3),
    }
    # Use the notebook's actual named-column selection. Its A1 display
    # interval is one; the mimic interval is an explicit physical displacement.
    source = next(cell for cell in notebook_cells("notes/intrinsic_alignments/EXAMPLE_FISHER_IA.ipynb")
                  if "shapes = response_plots.compare_templates" in cell)
    assignment = next(node for node in ast.walk(ast.parse(source))
                      if isinstance(node, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == "shapes"
                          for t in node.targets))
    namespace = dict(response_plots=shared, derivatives=result["derivatives_h2"],
                     names=list(result["parameter_names"]), mimic="DES_A2_1", mimic_scale=.2,
                     forecast={"covariance": result["fiducial_covariance"],
                               "row_kind": result["row_kind"]})
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "notebook", "exec"),
         namespace)
    report = namespace["shapes"]
    np.testing.assert_allclose(report["templates"][0], [1., 2., 3.])
    np.testing.assert_allclose(report["templates"][1], [.4, .8, 1.])
    panel = {
        "coordinate": np.array([1.0, 2.0, 3.0]),
        "templates": report["templates"],
        "residuals": np.array([
            report["two_point"]["residual"], report["joint"]["residual"],
        ]),
        "title": "Controlled responses",
        "coordinate_label": "Angle [arcmin]",
    }
    figure, axes = shared.plot_derivative_templates(
        panels=[panel], parameter_labels=["A1", "A2"], show=None,
    )
    np.testing.assert_array_equal(axes[0, 0].lines[0].get_ydata(), panel["templates"][0])
    np.testing.assert_array_equal(axes[1, 0].lines[1].get_ydata(), panel["residuals"][1])
    plt.close(figure)
    panel["coordinate"][0] = -1.0
    before = plt.get_fignums()
    assert shared.plot_derivative_templates(
        panels=[panel], parameter_labels=["A1", "A2"], show=None,
    ) == 0
    assert plt.get_fignums() == before
