"""Check fixed quadrature measures, fine rules and oscillating integrals."""

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest

from des3pcf.covariance import quadrature


@pytest.mark.parametrize("count", [8, 64, 256])
def test_rule_agrees_with_independent_companion_matrix(count):
    """Independent node algorithms implement the same integration rule."""
    reference_nodes, reference_weights = leggauss(deg=count)
    nodes, weights = quadrature(lower=-1.0, upper=1.0, count=count)
    np.testing.assert_allclose(nodes, reference_nodes, rtol=0.0, atol=3.0e-15)
    np.testing.assert_allclose(weights, reference_weights, rtol=0.0, atol=6.0e-14)


@pytest.mark.parametrize("count", [4096, 8192])
def test_fine_rule_resolves_oscillations_and_low_moments(count):
    """A resolved oscillation and polynomial moments have analytic integrals."""
    nodes, weights = quadrature(lower=-1.0, upper=1.0, count=count)
    assert np.all(weights > 0.0)
    assert np.all(np.diff(nodes) > 0.0)
    for degree in (0, 2, 8, 20):
        value = weights @ nodes**degree
        np.testing.assert_allclose(value, 2.0 / (degree + 1), rtol=0.0, atol=2e-11)
    frequency = 1000.0
    integral = weights @ np.cos(frequency * nodes)
    exact = 2.0 * np.sin(frequency) / frequency
    np.testing.assert_allclose(integral, exact, rtol=0.0, atol=2e-11)


def test_logarithmic_rule_includes_one_jacobian():
    """Integrating x and 1/x distinguishes dx from an accidental dlnx measure."""
    nodes, weights = quadrature(lower=0.01, upper=100.0, count=128, logarithmic=True)
    np.testing.assert_allclose(weights @ nodes, (100.0**2 - 0.01**2) / 2,
                               rtol=2e-13)
    np.testing.assert_allclose(weights @ (1.0 / nodes), np.log(100.0 / 0.01),
                               rtol=2e-13)
