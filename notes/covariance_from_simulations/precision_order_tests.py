"""Controlled second-, fourth- and sixth-order precision experiments.

Run beside the teaching notes with Cocoa's Python. The covariance is known
analytically; no DES map or fitted cosmological model is used. BLAS is one
worker. All generated results go to the project's ignored chains directory.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import numpy as np
import sympy as sp
from derive_wishart_powers import BASES, moment, nu


def coefficients(degrees_of_freedom):
    """Solve exact Gaussian moment equations at a specified integer nu.

    A column gives the expectation of one observed matrix/trace expression.
    Solving against the first coordinate selects the pure population power.
    Rational arithmetic precedes conversion to floating point for sampling.
    """
    result = {1: np.ones(1)}
    for degree in range(2, 7):
        basis = BASES[degree]
        columns = [moment(*entry) for entry in basis]
        matrix = sp.Matrix([
            [column.get(entry, 0) for column in columns] for entry in basis
        ]).subs(nu, degrees_of_freedom)
        target = sp.eye(len(basis))[:, 0]
        answer = matrix.inv(method='DM')[:, 0]
        assert matrix * answer == target
        scalar = sp.Rational(degrees_of_freedom)**degree
        scalar /= sp.prod(degrees_of_freedom + 2*j for j in range(degree))
        assert sum(answer) == scalar
        result[degree] = np.array(answer, dtype=float).ravel()
    return result


def gaussian_orders(sample, weights):
    """Build corrected powers and the three inverse polynomials in order.

    sample has shape (..., p, p). Multiplication uses matrix @; trace
    factors are scalars. The coefficients are valid for Gaussian vectors.
    Non-Gaussian experiments deliberately test failure of that assumption.
    """
    identity = np.eye(sample.shape[-1])
    powers = [np.broadcast_to(identity, sample.shape), sample]
    for degree in range(2, 7):
        powers.append(powers[-1] @ sample)
    traces = [np.trace(a=value, axis1=-2, axis2=-1) for value in powers]
    corrected = [identity, sample]
    for degree in range(2, 7):
        value = np.zeros_like(sample)
        for (power, trace_powers), weight in zip(BASES[degree], weights[degree]):
            multiplier = np.full(sample.shape[:-2], weight)
            for trace_power in trace_powers:
                multiplier *= traces[trace_power]
            value += multiplier[..., None, None] * powers[power]
        corrected.append(value)
    second = 3*identity - 3*sample + corrected[2]
    fourth = (5*identity - 10*sample + 10*corrected[2]
              - 5*corrected[3] + corrected[4])
    sixth = (7*identity - 21*sample + 35*corrected[2] - 35*corrected[3]
             + 21*corrected[4] - 7*corrected[5] + corrected[6])
    return np.stack([second, fourth, sixth])


def distinct_orders(vectors):
    """Average products of distinct pair factors through degree six.

    Every input vector occurs in a fixed pair. Descending updates prevent
    the newest random factor from appearing twice in any product. This is
    unbiased for each truncated target under independent identical sampling
    and a reference fixed independently of these vectors. It need not be PD.
    """
    assert vectors.shape[1] >= 12 and vectors.shape[1] % 2 == 0
    identity = np.eye(vectors.shape[-1])
    difference = (vectors[:, 0::2] - vectors[:, 1::2]) / np.sqrt(2.)
    shape = (len(vectors), vectors.shape[-1], vectors.shape[-1])
    means = [np.broadcast_to(identity, shape).copy()]
    means += [np.zeros(shape) for _ in range(6)]
    for count in range(1, difference.shape[1]+1):
        value = difference[:, count-1]
        factor = identity - value[..., :, None] * value[..., None, :]
        for degree in range(min(count, 6), 0, -1):
            means[degree] = ((count-degree)/count * means[degree]
                            + degree/count * (means[degree-1] @ factor))
    estimates = np.stack([sum(means[:order+1]) for order in (2, 4, 6)])
    return (estimates + estimates.swapaxes(-1, -2)) / 2.


def targets(covariance):
    """Return true finite polynomials and the exact inverse separately."""
    identity = np.eye(len(covariance))
    residual = identity - covariance
    term = identity.copy()
    total = identity.copy()
    result = []
    for degree in range(1, 7):
        term = term @ residual
        total = total + term
        if degree in (2, 4, 6):
            result.append(total.copy())
    return np.asarray(result), np.linalg.solve(a=covariance, b=identity)


def summarize(estimates, truth, exact):
    """Separate mean bias, individual scatter, exact-inverse error and PD."""
    count = estimates.shape[1]
    mean = estimates.mean(axis=1)
    se = estimates.std(axis=1, ddof=1) / np.sqrt(count)
    difference = estimates - truth[:, None]
    rms = np.sqrt(np.mean(np.sum(difference**2, axis=(-2, -1)), axis=1))
    rms /= np.linalg.norm(truth, axis=(-2, -1))
    exact_rms = np.sqrt(np.mean(np.sum((estimates-exact)**2,
                                      axis=(-2, -1)), axis=1))
    exact_rms /= np.linalg.norm(exact)
    nonpositive = np.mean(np.linalg.eigvalsh(estimates)[..., 0] <= 0., axis=1)
    maximum_z = np.max(np.abs(mean-truth)/se, axis=(-2, -1))
    return np.column_stack([maximum_z, rms, exact_rms, nonpositive,
                            mean[:, 0, 0]-truth[:, 0, 0], se[:, 0, 0]])


def main():
    """Repeat three bounded populations, preserving earlier random seeds."""
    project = Path(__file__).resolve().parents[2]
    output = project / 'chains' / 'covariance_sixth_order'
    output.mkdir(parents=True, exist_ok=True)
    weights = coefficients(degrees_of_freedom=127)
    covariance = np.array([[1., .3], [.3, 1.]])
    model = np.array([[1.4, .1], [.1, .9]])
    inverse = np.linalg.solve(a=np.linalg.cholesky(model), b=np.eye(2))
    truth, exact = targets(covariance=inverse@covariance@inverse.T)
    all_truth = np.concatenate([truth, truth])
    rows = []
    for case, sigma in enumerate((0., .25, .5)):
        rng = np.random.default_rng(seed=4815)
        if case == 0:
            factor = np.linalg.cholesky(covariance)
        else:
            variance = np.expm1(sigma*sigma)
            latent = np.log1p(variance*covariance)
            factor = np.linalg.cholesky(latent)
        batches = []
        for start in range(0, 32000, 1000):
            draws = rng.normal(size=(1000, 128, 2)) @ factor.T
            if case:
                draws = np.expm1(draws-np.diag(latent)/2.)/np.sqrt(variance)
            white = draws @ inverse.T
            centered = white-white.mean(axis=1, keepdims=True)
            sample = centered.swapaxes(-1, -2) @ centered / 127.
            batches.append(np.concatenate([
                gaussian_orders(sample=sample, weights=weights),
                distinct_orders(vectors=white),
            ]))
        estimates = np.concatenate(batches, axis=1)
        summary = summarize(estimates=estimates, truth=all_truth, exact=exact)
        for method, row in enumerate(summary):
            rows.append([case, method, *row])
        print('case', case, 'rows Wishart 2,4,6; distinct 2,4,6\n', summary,
              flush=True)
    np.savetxt(
        fname=output/'toy_summary.txt',
        X=rows,
        header=('case method max_bias_over_MCSE relative_RMS_truncated '
                'relative_RMS_exact nonpositive_fraction bias00 MCSE00; '
                'cases Gaussian,lognormal_s0.25,s0.5; '
                'methods Wishart2,4,6,distinct2,4,6; 32000x128'),
    )


if __name__ == '__main__':
    main()
