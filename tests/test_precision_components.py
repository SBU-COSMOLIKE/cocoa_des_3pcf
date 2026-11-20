"""Check the known-component PME without requiring a cosmological simulation.

Noncommuting covariances separate the two ordered cross products in Eq12.
The stochastic check compares an ensemble mean with the analytic truncated
series; it never mistakes a finite Taylor polynomial for the exact inverse.
"""

import numpy as np
import pytest

from cosmolike_notebook_utils.covariance.precision import component_precision_expansion
from cosmolike_notebook_utils.covariance.precision import precision_expansion


def component_example():
    """Return positive 3D known/model/true components with different modes."""
    known = np.diag([1.8, 1.4, 1.1])
    factor = np.array([
        [0.6, 0.0, 0.0],
        [0.2, 0.7, 0.0],
        [-0.1, 0.3, 0.5],
    ])
    component = factor @ factor.T
    baseline = np.array([
        [0.5, 0.06, -0.02],
        [0.06, 0.6, 0.04],
        [-0.02, 0.04, 0.4],
    ])
    return known, baseline, component


def test_component_pme_reduces_to_whole_covariance_estimator():
    """A=0 and Bm=M must give the existing full-Wishart polynomial."""
    known, baseline, component = component_example()
    model = known + baseline
    for order in (1, 2):
        actual, _ = component_precision_expansion(
            analytic=model, component_model=model, component_sample=component,
            independent_count=7, order=order,
        )
        expected, _ = precision_expansion(
            analytic=model, sample=component, independent_count=7, order=order,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-15)


def test_component_pme_noncommuting_eq12_and_infinite_sample_limit():
    """Compute Eq12 in original units, without reusing the whitened formula."""
    known, baseline, component = component_example()
    model = known + baseline
    inverse = np.linalg.inv(model)
    assert not np.allclose(component @ inverse @ baseline,
                           baseline @ inverse @ component)
    for count in (5, 1000000000000):
        nu = count - 1.0
        correction = nu**2 * component @ inverse @ component
        correction -= nu * component * np.trace(inverse @ component)
        correction /= nu**2 + nu - 2.0
        expected = inverse - inverse @ (component - baseline) @ inverse
        expected += inverse @ baseline @ inverse @ baseline @ inverse
        expected -= inverse @ component @ inverse @ baseline @ inverse
        expected -= inverse @ baseline @ inverse @ component @ inverse
        expected += inverse @ correction @ inverse
        actual, _ = component_precision_expansion(
            analytic=model, component_model=baseline,
            component_sample=component, independent_count=count,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-15)
    residual = component - baseline
    limit = inverse - inverse @ residual @ inverse
    limit += inverse @ residual @ inverse @ residual @ inverse
    np.testing.assert_allclose(actual, limit, rtol=2e-12, atol=1e-13)
    assert not np.allclose(limit, np.linalg.inv(known + component))


def test_gaussian_component_ensemble_targets_truncation_not_exact_inverse():
    """Small groups test the Wishart bias subtraction in expectation."""
    known, baseline, component = component_example()
    model = known + baseline
    inverse = np.linalg.inv(model)
    residual = component - baseline
    target = inverse - inverse @ residual @ inverse
    target += inverse @ residual @ inverse @ residual @ inverse
    rng = np.random.default_rng(seed=4231)
    estimates = []
    # N=3 equals the vector dimension: individual centered covariances are
    # singular. The mean-of-estimates test never inverts those covariances.
    for repeat in range(2400):
        draws = rng.multivariate_normal(
            mean=np.zeros(3), cov=component, size=3,
        )
        measured = np.cov(draws, rowvar=False, ddof=1)
        precision, _ = component_precision_expansion(
            analytic=model, component_model=baseline,
            component_sample=measured, independent_count=3,
        )
        estimates.append(precision)
    estimates = np.asarray(estimates)
    error = np.std(estimates, axis=0, ddof=1) / np.sqrt(len(estimates))
    difference = np.abs(np.mean(estimates, axis=0) - target)
    assert np.all(difference < 6.0 * error)


@pytest.mark.parametrize("component", ["known", "modeled", "sample"])
def test_component_pme_refuses_indefinite_components(component):
    """A cross block with zero variance cannot serve as a covariance B."""
    model = np.eye(2)
    baseline = 0.3 * model
    sample = baseline.copy()
    if component == "known":
        baseline = 1.3 * model
    elif component == "modeled":
        baseline = np.array([[0.0, 0.2], [0.2, 0.0]])
    else:
        sample = np.array([[0.0, 0.2], [0.2, 0.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        component_precision_expansion(
            analytic=model, component_model=baseline,
            component_sample=sample, independent_count=5,
        )
