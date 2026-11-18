"""Test filtered pairs and contractions of a stated covariance model.

The two-point function of filtered Gaussian maps determines their six-field
moment through Wick's theorem: every field is paired with one other field,
giving fifteen products of three pair correlations. This test enumerates
those pairings independently. The connected response calculation has a
different interpretation: positive halo weights form a Gram matrix, which
is positive semidefinite even when individual responses have either sign.
Neither calculation supplies a complete survey covariance by itself.
"""

import numpy as np


def pairings(labels):
    """Enumerate each perfect matching of distinct field positions once.

    Arguments:
        labels = tuple of an even number of distinct position indices.

    Returns:
        list of pair lists. With six positions the list has fifteen entries.

    Choose the partner of the first remaining position, then pair the
    unchosen positions recursively. Repeated source labels stay distinct
    positions; otherwise their combinatorial factors would be lost.
    """
    if not labels:
        return [[]]
    results = []
    first = labels[0]
    for index in range(1, len(labels)):
        second = labels[index]
        remaining = labels[1:index] + labels[index + 1:]
        for tail in pairings(labels=remaining):
            results.append([(first, second)] + tail)
    return results


def test_filtered_white_noise(compiled):
    """Compare a Hankel integral with its analytic Gaussian-filter result.

    A white spectrum N has no ell dependence. For aperture radii R and S,
    a=(R^2+S^2)/2 and x=r^2/(4a), the filtered correlation is
    N R^2 S^2 exp(-x)(1-2x+x^2/2)/(8 pi a^3). The quadrature integrates
    d ell; the C routine supplies ell/(2 pi) exactly once.
    """
    nodes, weights = np.polynomial.legendre.leggauss(deg=192)
    ell_max = 12000.0  # The smallest filter suppresses the omitted tail.
    ell = np.ascontiguousarray(0.5 * ell_max * (nodes + 1.0))
    ell_weights = np.ascontiguousarray(0.5 * ell_max * weights)
    radii = np.array([0.002, 0.003, 0.005])
    separation = np.array([0.0, 0.0007, 0.002, 0.004, 0.008])
    argument = radii[:, None] * ell[None, :]
    filters = 0.5 * argument**2 * np.exp(-0.5 * argument**2)
    noise = np.eye(3) + 0.2 * np.ones((3, 3))
    power = np.repeat(noise[:, :, None], ell.size, axis=2)
    expected = np.empty((3, 3, separation.size))
    for first in range(3):
        for second in range(3):
            radius_a = radii[first]
            radius_b = radii[second]
            width = (radius_a**2 + radius_b**2) / 2.0
            scaled_separation = separation**2 / (4.0 * width)
            polynomial = 1.0 - 2.0 * scaled_separation
            polynomial += 0.5 * scaled_separation**2
            normalization = noise[first, second] * radius_a**2 * radius_b**2
            normalization /= 8.0 * np.pi * width**3
            expected[first, second] = (
                normalization * np.exp(-scaled_separation) * polynomial
            )

    for adapter in (compiled.production, compiled.notebook):
        actual = adapter.filtered_pair(
            ell=ell, ell_weights=ell_weights, r=separation,
            filters=filters, power=power
        )
        np.testing.assert_allclose(actual, expected, rtol=3.0e-10, atol=1.0e-8)


def test_all_wick_pairings(compiled):
    """Check unequal field labels and repeated labels against all matchings."""
    rng = np.random.default_rng(seed=4710)
    basis = rng.normal(size=(5, 5))
    pair_zero = basis @ basis.T
    radial_shape = np.array([0.9, 0.7, 0.4, 0.2, 0.05])
    pair = pair_zero[:, :, None] * radial_shape[None, None, :]
    radial_weights = np.array([0.1, 0.25, 0.3, 0.2, 0.15])
    triplets = np.array([[0, 1, 2], [1, 3, 4], [0, 0, 0]], dtype=np.int32)
    matchings = pairings(labels=tuple(range(6)))
    assert len(matchings) == 15
    expected = np.zeros((3, 3))

    # Positions 0..2 belong to the left aperture, 3..5 to the right.
    # A pair at one centre uses K(0); a pair spanning centres uses K(r).
    for first_row in range(3):
        for second_row in range(3):
            labels = np.concatenate((triplets[first_row], triplets[second_row]))
            for radial in range(5):
                moment = 0.0
                for matching in matchings:
                    product = 1.0
                    for first, second in matching:
                        same_centre = (first < 3) == (second < 3)
                        if same_centre:
                            product *= pair_zero[labels[first], labels[second]]
                        else:
                            product *= pair[labels[first], labels[second], radial]
                    moment += product
                expected[first_row, second_row] += radial_weights[radial] * moment

    for adapter in (compiled.production, compiled.notebook):
        actual = adapter.gaussian_aperture_covariance(
            triplets=triplets, pair=pair, pair_zero=pair_zero,
            radial_weights=radial_weights
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=1.0e-12)
        np.testing.assert_array_equal(actual, actual.T)


def test_identical_field_gaussian_moment(compiled):
    """For one field the integrand is 6 K(r)^3 + 9 K(0)^2 K(r)."""
    correlation = np.array([0.7, 0.2, -0.1, 0.05, 0.01])
    variance = 1.3
    weights = np.array([0.4, 0.3, 0.2, 0.08, 0.02])
    expected = np.dot(weights, 6.0 * correlation**3 + 9.0 * variance**2 * correlation)
    actual = compiled.production.gaussian_aperture_covariance(
        triplets=np.array([[0, 0, 0]], dtype=np.int32),
        pair=correlation.reshape(1, 1, -1),
        pair_zero=np.array([[variance]]), radial_weights=weights
    )
    np.testing.assert_allclose(actual[0, 0], expected, rtol=2.0e-14)


def test_connected_gram_rank_and_sign(compiled):
    """Check rank two, signed cross entries and nonnegative variances."""
    responses = np.array([
        [1.0, -2.0, 0.5, 1.0, -0.7],
        [-1.0, 2.0, -0.5, -1.0, 0.7],
        [0.2, 0.3, -0.7, 1.1, 0.4],
    ])
    weights = np.array([0.2, 0.4, 0.3, 0.0, 0.1])
    expected = (responses * weights[None, :]) @ responses.T
    for adapter in (compiled.production, compiled.notebook):
        actual = adapter.connected_gram(responses=responses, weights=weights)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=1.0e-15)
        np.testing.assert_array_equal(actual, actual.T)
        assert actual[0, 1] < 0.0
        assert np.linalg.matrix_rank(actual, tol=1.0e-12) == 2
        assert np.linalg.eigvalsh(actual)[0] >= -1.0e-12


def test_covariance_threads_and_reuse(compiled):
    """Changing workers preserves ordered pair, Wick and response sums."""
    rng = np.random.default_rng(seed=4711)
    responses = rng.normal(size=(7, 13))
    weights = rng.uniform(size=13)
    expected = compiled.production.connected_gram(responses=responses, weights=weights)
    pair_inputs = {
        "ell": np.linspace(1.0, 7.0, num=7),
        "ell_weights": np.ones(7),
        "r": np.linspace(0.0, 0.4, num=5),
        "filters": rng.uniform(size=(3, 7)),
        "power": np.ones((3, 3, 7)),
    }
    expected_pair = compiled.production.filtered_pair(**pair_inputs)
    gaussian_inputs = {
        "triplets": np.array([[0, 1, 2], [1, 1, 0]], dtype=np.int32),
        "pair": expected_pair,
        "pair_zero": np.ascontiguousarray(expected_pair[:, :, 0]),
        "radial_weights": np.ones(5) / 5.0,
    }
    expected_gaussian = compiled.production.gaussian_aperture_covariance(
        **gaussian_inputs
    )
    for nthreads in (2, 3, 4, 1):
        compiled.set_omp_threads(nthreads=nthreads)
        actual = compiled.production.connected_gram(responses=responses, weights=weights)
        np.testing.assert_array_equal(actual, expected)
        compiled.production.connected_gram(
            responses=responses[:3].copy(), weights=weights * 2.0
        )
        restored = compiled.production.connected_gram(
            responses=responses, weights=weights
        )
        np.testing.assert_array_equal(restored, expected)
        actual_pair = compiled.production.filtered_pair(**pair_inputs)
        actual_gaussian = compiled.production.gaussian_aperture_covariance(
            **gaussian_inputs
        )
        np.testing.assert_array_equal(actual_pair, expected_pair)
        np.testing.assert_array_equal(actual_gaussian, expected_gaussian)
