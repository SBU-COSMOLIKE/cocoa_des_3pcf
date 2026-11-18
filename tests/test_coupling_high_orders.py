"""Check the geometric coupling at the larger angular orders used in studies.

The native routine uses finite Fourier sums. This independent reference
instead integrates the defining angular function with a fixed Gauss rule.
It tests geometry only: no cosmology, bispectrum or aperture mean is built.
The tolerance uses the natural 4*pi**2 scale, including coefficients that
are exactly zero; it is not a fractional error bound for tiny coefficients.
"""

import numpy as np
import pytest
from scipy.special import eval_legendre, roots_legendre


@pytest.mark.parametrize("node_count", [512, 1024])
def test_high_order_coupling_against_direct_integral(compiled, node_count):
    """Check all L=0..50 and M=0..100 at seven side-ratio angles.

    For z=exp(i*delta) and t=tan(psi), the geometric phase is
    (t+z)/(1+t*z). The desired real coupling is 4*pi times the
    integral of P_L(cos(delta))*exp(i*M*delta)*phase over [0, pi].
    Equal sides have phase one away from a measure-zero endpoint;
    psi=pi/2 has phase exp(-i*delta). Using those exact limiting
    expressions avoids introducing an artificial tan(pi/2) singularity.
    This finite set of angles does not resolve every near-equal-side
    boundary region, or test the subsequent shear/aperture transforms.
    """
    psi = np.array(object=[
        0.0,
        0.1,
        0.4,
        np.pi / 4.0,
        1.1,
        1.4,
        np.pi / 2.0,
    ])
    actual = compiled.production.legendre_coupling(
        psi=psi, lmax=50, mmax=100, npsi=0,
    )
    nodes, weights = roots_legendre(n=node_count)
    angle = (nodes + 1.0) * np.pi / 2.0
    weights = weights * np.pi / 2.0
    # These SciPy ufunc inputs are positional. The broadcast axes are
    # [Legendre degree, quadrature node] and [Fourier mode, node].
    legendre = eval_legendre(
        np.arange(stop=51)[:, None], np.cos(angle)[None, :],
    )
    fourier = np.exp(1j * np.arange(stop=101)[:, None] * angle[None, :])
    direct = np.empty_like(prototype=actual)
    for index, value in enumerate(psi):
        if value == np.pi / 4.0:
            phase = np.ones_like(a=angle, dtype=complex)
        elif value == np.pi / 2.0:
            phase = np.exp(-1j * angle)
        else:
            ratio = np.tan(value)
            phase = (ratio + np.exp(1j * angle)) / (
                1.0 + ratio * np.exp(1j * angle)
            )
        weighted_phase = (fourier * phase[None, :]).real * weights[None, :]
        direct[:, :, index] = 4.0 * np.pi * (weighted_phase @ legendre.T)
    assert actual.shape == (
        101,
        51,
        psi.size,
    )
    np.testing.assert_allclose(
        actual=actual, desired=direct, rtol=0.0,
        atol=4.0 * np.pi**2 * 1.0e-11,
    )
