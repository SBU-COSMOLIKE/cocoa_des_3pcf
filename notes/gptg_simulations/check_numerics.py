"""Check the map transform and sixth-order algebra without a large simulation run.

The checks isolate deterministic numerical operations from the uncertain
physical calibration: Hermite correlation integrals, harmonic conventions,
noise subtraction, and products of distinct random factors. They do not
establish agreement with a DES catalogue estimator or N-body covariance.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'

from itertools import combinations
import sys
import ducc0
import numpy as np
from scipy.special import roots_hermitenorm
import yaml
from generate_maps import PROJECT, OUTPUT, normalized_transform, harmonic_indices

sys.path.insert(0, str(PROJECT / 'notes/covariance_from_simulations'))
from precision_order_tests import distinct_orders


def main():
    """Print measured residuals and stop if a numerical identity fails."""
    record = yaml.safe_load((OUTPUT / 'calibration.yaml').read_text())
    parameters = np.asarray(record['point_parameters_a1_a2_b_x0_t'])
    nodes, weights = roots_hermitenorm(n=160)
    weights /= np.sqrt(2*np.pi)
    fine_nodes, fine_weights = roots_hermitenorm(n=240)
    fine_weights /= np.sqrt(2*np.pi)
    basis = np.ones((97, len(nodes)))
    basis[1] = nodes
    for degree in range(1, 96):
        basis[degree+1] = (
            nodes*basis[degree] - np.sqrt(degree)*basis[degree-1]
        ) / np.sqrt(degree+1)
    coefficients = []
    normalization_change = 0.
    for values in parameters:
        coarse = normalized_transform(
            gaussian=nodes,
            parameters=values,
            nodes=nodes,
            weights=weights,
        )
        fine = normalized_transform(
            gaussian=nodes,
            parameters=values,
            nodes=fine_nodes,
            weights=fine_weights,
        )
        # Weighted RMS measures the difference over Gaussian probability,
        # rather than making a negligible far-tail value set the scale.
        change = np.sqrt(weights @ (coarse-fine)**2)
        normalization_change = max(normalization_change, change)
        coefficients.append(basis @ (weights*coarse))
    coefficients = np.asarray(coefficients)
    integration_error = 0.
    truncation_error = 0.
    for i in range(4):
        first = normalized_transform(
            gaussian=nodes,
            parameters=parameters[i],
            nodes=nodes,
            weights=weights,
        )
        for j in range(i, 4):
            for rho in (-0.8, 0., 0.5, 0.95):
                # Conditional construction converts the correlated 2D
                # Gaussian integral to a product quadrature in two IID axes.
                second_input = rho*nodes[:, None]
                second_input = second_input + np.sqrt(1-rho*rho)*nodes[None, :]
                second = normalized_transform(
                    gaussian=second_input,
                    parameters=parameters[j],
                    nodes=nodes,
                    weights=weights,
                )
                direct = (weights*first) @ second @ weights
                series = coefficients[i]*coefficients[j]
                series[0] = 0.
                coarse = np.polynomial.polynomial.polyval(rho, series[:65])
                fine = np.polynomial.polynomial.polyval(rho, series)
                integration_error = max(integration_error, abs(coarse-direct))
                truncation_error = max(truncation_error, abs(coarse-fine))
    assert normalization_change < 1.e-5
    assert integration_error < 1.e-5
    assert truncation_error < 1.e-5
    print('Normalization 160 -> 240 nodes, weighted RMS:', normalization_change)
    print('Hermite 64 versus direct 2D quadrature:', integration_error)
    print('Hermite 64 -> 96, maximum correlation change:', truncation_error)

    # A low-resolution band-limited sky isolates transform conventions.
    # Here there is no nonlinear high-ell signal for a finite projection
    # to discard, so the harmonic coefficients should round-trip.
    lmax = 24
    ell = harmonic_indices(lmax=lmax)
    rng = np.random.default_rng(seed=114759)
    alm = rng.normal(size=(1, len(ell))).astype(complex)
    alm += 1j*rng.normal(size=alm.shape)
    alm[:, :lmax+1] = alm[:, :lmax+1].real
    alm[:, ell < 2] = 0.
    geometry = ducc0.healpix.Healpix_Base(32, 'RING').sht_info()
    sky = ducc0.sht.synthesis(
        alm=alm,
        spin=0,
        lmax=lmax,
        nthreads=4,
        **geometry,
    )
    recovered = ducc0.sht.pseudo_analysis(
        map=sky,
        spin=0,
        lmax=lmax,
        maxiter=30,
        epsilon=1.e-10,
        nthreads=4,
        **geometry,
    )[0]
    roundtrip = np.linalg.norm(recovered-alm) / np.linalg.norm(alm)
    assert roundtrip < 1.e-8
    print('Band-limited scalar harmonic round trip:', roundtrip)

    # Enumerating all subsets is cheap for seven pairs. Factors deliberately
    # do not commute, so a scalar-only check would not test their order.
    vectors = rng.normal(size=(1, 14, 3))
    pair = (vectors[0, 0::2]-vectors[0, 1::2]) / np.sqrt(2.)
    factors = np.eye(3)-pair[:, :, None]*pair[:, None, :]
    terms = [np.eye(3)]
    for degree in range(1, 7):
        products = []
        for subset in combinations(range(7), degree):
            product = np.eye(3)
            for index in subset:
                product = product @ factors[index]
            products.append(product)
        terms.append(np.mean(
            a=products,
            axis=0,
        ))
    expected = []
    for order in (2, 4, 6):
        value = sum(terms[:order+1])
        expected.append((value+value.T)/2.)
    actual = distinct_orders(vectors=vectors)[:, 0]
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.e-12,
        atol=1.e-12,
    )
    print('Distinct-pair recurrence agrees with every subset through degree 6.')

    # Gaussian quadrature integrates these low-degree polynomials exactly.
    # No Monte Carlo fluctuation can hide an incorrect coefficient of 3.
    signal = .2
    noise_variance = .07
    observed = signal + np.sqrt(noise_variance)*nodes
    second = weights @ (observed**2-noise_variance)
    third = weights @ (observed**3-3*noise_variance*observed)
    np.testing.assert_allclose(second, signal**2, atol=1.e-14)
    np.testing.assert_allclose(third, signal**3, atol=1.e-14)
    print('Noise-subtracted square and cube recover', second, third)


if __name__ == '__main__':
    main()
