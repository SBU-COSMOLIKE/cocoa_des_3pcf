"""Compare all 10000 maps, then score fixed estimates on 5000 new maps.

The validation score never inverts the validation covariance and does not
assume the direct-sample inverse is true. All choices are fixed in advance:
16 raw moment rows, reference=2 times the archived Gaussian moment reference,
orders 2/4/6, Gaussian-power and distinct-pair constructions. This evaluates
the initial fitted GPTG model, not a validated DES physical covariance.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'notes/covariance_from_simulations'))
from precision_order_tests import coefficients, gaussian_orders, distinct_orders


def main():
    """Produce the same-sample comparison and one independent validation table.

    Validation residuals are centered by their own sample mean. The quadratic
    mean uses denominator Nval-1, giving an unbiased covariance contraction.
    Its reported standard error is an asymptotic estimate from per-map
    quadratic differences; centering makes them weakly dependent. An
    indefinite estimate is recorded and receives no Gaussian precision score.
    """
    folder = PROJECT / 'chains/gptg_ensemble_15000'
    comparison = np.loadtxt(fname=folder / 'comparison_measurements.txt')
    validation = np.loadtxt(fname=folder / 'validation_measurements.txt')
    if comparison.shape != (10000, 20) or validation.shape != (5000, 20):
        raise ValueError('Finish all 10000 comparison and 5000 validation maps before scoring.')
    np.testing.assert_array_equal(comparison[:, 0], np.arange(10000))
    np.testing.assert_array_equal(validation[:, 0], np.arange(5000))
    if np.intersect1d(comparison[:, 1], validation[:, 1]).size:
        raise ValueError('Comparison and validation seed ranges overlap.')
    reference = np.loadtxt(fname=folder / 'fixed_reference.txt')
    factor = np.linalg.cholesky(reference)
    inverse = np.linalg.solve(
        a=factor,
        b=np.eye(16),
    )
    # Work in fixed reference coordinates so numerical units do not dominate
    # matrix norms. Determinant differences are invariant to this common
    # coordinate change; no validation observation chooses these scales.
    draws = comparison[:, 4:] @ inverse.T
    check_draws = validation[:, 4:] @ inverse.T
    sample = np.cov(
        m=draws,
        rowvar=False,
        ddof=1,
    )
    direct = np.linalg.solve(
        a=sample,
        b=np.eye(16),
    )
    weights = coefficients(degrees_of_freedom=9999)
    gaussian = gaussian_orders(
        sample=sample,
        weights=weights,
    )
    distinct = distinct_orders(vectors=draws[None, :, :])[:, 0]
    estimates = [direct]
    for estimate in gaussian:
        estimates.append(estimate)
    for estimate in distinct:
        estimates.append(estimate)
    labels = ['direct', 'gaussian2', 'gaussian4', 'gaussian6',
              'distinct2', 'distinct4', 'distinct6']
    validation_centered = check_draws-check_draws.mean(axis=0)
    validation_covariance = np.cov(
        m=check_draws,
        rowvar=False,
        ddof=1,
    )
    direct_logdet = np.linalg.slogdet(direct)[1]
    table = []
    for index, (label, precision) in enumerate(zip(labels, estimates)):
        minimum = np.linalg.eigvalsh(precision)[0]
        np.savetxt(
            fname=folder / f'precision_{label}_whitened.txt',
            X=precision,
        )
        if minimum <= 0.:
            table.append([index, minimum, np.nan, np.nan, np.nan])
            print(label, 'indefinite; no likelihood score', flush=True)
            continue
        covariance = np.linalg.solve(
            a=precision,
            b=np.eye(16),
        )
        physical_covariance = factor @ covariance @ factor.T
        np.savetxt(
            fname=folder / f'covariance_{label}.txt',
            X=physical_covariance,
        )
        difference = precision-direct
        quadratic = np.einsum(
            'ni,ij,nj->n', validation_centered, difference, validation_centered,
        )
        score_difference = quadratic.sum() / (len(validation)-1)
        score_difference -= np.linalg.slogdet(precision)[1]-direct_logdet
        # Only the quadratic term fluctuates on validation maps. The fitted
        # determinant term is fixed conditional on the comparison ensemble.
        error = quadratic.std(ddof=1) / np.sqrt(len(validation))
        error *= len(validation)/(len(validation)-1)
        calibration = np.trace(precision @ validation_covariance)/16.
        table.append([index, minimum, score_difference, error, calibration])
        print(label, 'validation score difference, SE, mean quadratic/16:',
              score_difference, error, calibration, flush=True)
    np.savetxt(
        fname=folder / 'validation_scores.txt',
        X=table,
        header=('method_index min_eigenvalue score_minus_direct approximate_SE '
                'validation_quadratic_per_row; labels=' + ','.join(labels)
                + '; negative score_difference favors candidate; initial fitted model only'),
    )
    np.savetxt(
        fname=folder / 'comparison_covariance_whitened.txt',
        X=sample,
    )
    np.savetxt(
        fname=folder / 'validation_covariance_whitened.txt',
        X=validation_covariance,
    )
    (folder / 'ANALYSIS_COMPLETE.txt').write_text(
        'All-map estimates and independent validation scores written.\n'
        'No validation-tuned reference or expansion order.\n',
    )


if __name__ == '__main__':
    main()
