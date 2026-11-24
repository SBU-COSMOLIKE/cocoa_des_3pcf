"""Check fixed quadrature measures, fine rules and oscillating integrals."""

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest

from cobaya.likelihoods.des_3pcf.survey import survey_rows
from cosmolike_des_3pcf_interface import quadrature


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


def test_xi_quadrature_changes_and_restores_without_retained_state(compiled):
    """The native integral consumes each supplied grid, including A/B/A edits."""
    rows = survey_rows(nbin=2, aperture_radii=[], xi_edges=[0.001, 0.002])
    ell = np.array([30.0, 100.0, 500.0])
    weights = np.array([20.0, 80.0, 300.0])
    theta, theta_weight = compiled.xi_bin_weights(
        kind=rows.kind.tolist(),
        angles=np.ascontiguousarray(rows.angles),
        count=3,
        pair_weight="uniform_area",
    )
    spectra = np.ones((rows.values.size, ell.size)) * 1.0e-8
    arguments = dict(
        ell=ell,
        dell=weights,
        theta=theta,
        theta_weight=theta_weight,
        sign=np.where(rows.kind == "xim", -1, 1).astype(np.int32),
        spectra_e=spectra,
        spectra_b=np.zeros_like(spectra),
    )
    baseline = compiled.production.xi_from_power(**arguments)
    changed = compiled.production.xi_from_power(**dict(arguments, ell=2.0*ell))
    assert not np.array_equal(changed, baseline)
    np.testing.assert_array_equal(
        compiled.production.xi_from_power(**arguments), baseline,
    )
