"""Check information accounting independently of any survey prediction.

Correlated two/three-point fluctuations are deliberately large in these
small matrices. A mistaken zero cross block or a selected precision block
therefore cannot masquerade as the correct two-point-only experiment.
"""

import numpy as np
import pytest

from des3pcf.fisher_comparison import compare_information, marginalized_errors


def test_schur_information_and_prior_count_with_interleaved_rows():
    """The new information plus xi equals the joint likelihood, before priors."""
    factor = np.array([
        [2.0, 0.0, 0.0, 0.0],
        [1.1, 1.3, 0.0, 0.0],
        [-0.4, 0.6, 1.7, 0.0],
        [0.9, -0.3, 0.7, 1.2],
    ])
    covariance = factor @ factor.T
    derivative = np.array([[1.0, 2.0], [-0.3, 1.0], [2.0, -0.7], [0.2, 1.4]])
    names = ["amplitude", "alignment"]
    kind = np.array(["xip", "map3", "xim", "map3"])
    comparison = compare_information(
        derivatives=derivative, names=names, kind=kind,
        covariance=covariance, priors={"alignment": 0.5},
    )
    expected_joint = derivative.T @ np.linalg.solve(covariance, derivative)
    indices = np.array([0, 2])
    selected = covariance[np.ix_(indices, indices)]
    expected_two = derivative[indices].T @ np.linalg.solve(selected, derivative[indices])
    np.testing.assert_allclose(comparison["fisher_joint_likelihood"], expected_joint)
    np.testing.assert_allclose(comparison["fisher_two_point_likelihood"], expected_two)
    np.testing.assert_allclose(
        comparison["fisher_two_point_likelihood"]
        + comparison["fisher_conditional_three_point"], expected_joint,
    )
    np.testing.assert_array_equal(comparison["fisher_prior"], [[0.0, 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(
        comparison["fisher_joint"], expected_joint + [[0, 0], [0, 4]],
    )
    np.testing.assert_allclose(
        comparison["fisher_two_point"], expected_two + [[0, 0], [0, 4]],
    )
    # Selecting an inverse block conditions on the discarded data; it does
    # not describe observing xi alone. This counterexample separates them.
    wrong = np.linalg.inv(covariance)[np.ix_(indices, indices)]
    assert not np.allclose(wrong, np.linalg.inv(selected))
    errors, statuses = marginalized_errors(comparison=comparison, names=names)
    assert statuses == {"two_point": "resolved", "joint": "resolved"}
    assert np.all(errors["sigma_ratio_two_point_over_joint"] >= 1.0)
    from_precision = compare_information(
        derivatives=derivative, names=names, kind=kind,
        precision=np.linalg.inv(covariance), priors={"alignment": 0.5},
    )
    for name in ("fisher_joint", "fisher_two_point", "fisher_conditional_three_point"):
        np.testing.assert_allclose(from_precision[name], comparison[name])


def test_perfectly_predicted_three_point_response_adds_no_information():
    """A nonzero D3 can carry zero new information after correlated xi is known."""
    covariance = np.array([[4.0, 2.0], [2.0, 3.0]])
    derivative = np.array([[2.0], [1.0]])
    comparison = compare_information(
        derivatives=derivative, names=["amplitude"], kind=["xip", "map3"],
        covariance=covariance,
    )
    np.testing.assert_array_equal(comparison["conditional_derivatives"], 0.0)
    np.testing.assert_array_equal(comparison["fisher_conditional_three_point"], 0.0)
    np.testing.assert_allclose(comparison["fisher_joint"], comparison["fisher_two_point"])


def test_rank_failure_is_reported_separately_for_each_experiment():
    """A parameter constrained only by Map3 stays unresolved in xi alone."""
    comparison = compare_information(
        derivatives=np.eye(2), names=["first", "second"],
        kind=["xip", "map3"], covariance=np.eye(2),
    )
    arrays, statuses = marginalized_errors(
        comparison=comparison, names=["first", "second"],
    )
    assert statuses["joint"] == "resolved"
    assert "Unconstrained" in statuses["two_point"]
    assert "sigma_two_point" not in arrays
    assert "sigma_ratio_two_point_over_joint" not in arrays
    with pytest.raises(ValueError, match="both xi and Map3"):
        compare_information(derivatives=np.eye(2), names=["a", "b"],
                            kind=["xip", "xim"], covariance=np.eye(2))
