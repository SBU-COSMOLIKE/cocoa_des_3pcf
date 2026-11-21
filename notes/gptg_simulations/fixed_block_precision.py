"""Research precision expansion with a supplied, exact two-point block.

Distinct independent map pairs supply cross and three-point entries only.
The finite polynomial is unbiased for its deterministic polynomial target.
The final conditional reparameterization retains A exactly, but is nonlinear
and is NOT claimed to preserve finite-sample unbiasedness.
"""
import numpy as np


def estimates(vectors, reference, two_point_count):
    """Return second/fourth/sixth conditional precision and covariance pairs.

    reference must be fixed independently of vectors; its leading block is
    the exact supplied A. No Gaussian/Wishart sampling assumption is used.
    Odd final vectors are omitted from pairing, explicitly returned as used.
    """
    n = len(vectors)//2*2
    if n < 12:
        raise ValueError('Sixth order needs at least six independent pairs.')
    p = two_point_count
    dimension = vectors.shape[1]
    a = reference[:p, :p]
    factor = np.linalg.cholesky(reference)
    inverse = np.linalg.solve(factor, np.eye(dimension))
    difference = (vectors[:n:2]-vectors[1:n:2])/np.sqrt(2.)
    means = [np.eye(dimension)] + [np.zeros_like(reference) for _ in range(6)]
    for count, row in enumerate(difference, 1):
        draw = np.outer(row, row)
        # This replacement is in physical coordinates, before whitening.
        # Every pair has the same exact A; no simulation estimates that block.
        draw[:p, :p] = a
        residual = np.eye(dimension)-inverse @ draw @ inverse.T
        for degree in range(min(count, 6), 0, -1):
            means[degree] = ((count-degree)/count*means[degree]
                            + degree/count*(means[degree-1] @ residual))
    results = []
    for order in (2, 4, 6):
        polynomial = sum(means[:order+1])
        raw = inverse.T @ ((polynomial+polynomial.T)/2) @ inverse
        k = raw[p:, p:]
        if np.linalg.eigvalsh(k)[0] <= 0:
            results.append((order, raw, None, None))
            continue
        # H predicts y from x, and K is the estimated residual precision.
        # Building the covariance from these two quantities keeps A exact.
        h = -np.linalg.solve(k, raw[p:, :p])
        conditional = np.linalg.solve(k, np.eye(dimension-p))
        covariance = np.block([[a, a @ h.T],
                               [h @ a, conditional+h @ a @ h.T]])
        precision = np.block([[np.linalg.solve(a, np.eye(p))+h.T @ k @ h,
                               -h.T @ k], [-k @ h, k]])
        np.testing.assert_array_equal(covariance[:p, :p], a)
        np.testing.assert_allclose((factor.T @ precision @ factor)
                                   @ (inverse @ covariance @ inverse.T), np.eye(dimension),
                                   atol=2.e-7, rtol=2.e-7)
        results.append((order, raw, precision, covariance))
    return results, n
