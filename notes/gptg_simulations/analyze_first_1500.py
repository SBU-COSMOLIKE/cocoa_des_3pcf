"""Freeze estimates at 1000 maps, then score them on the next 500 maps.

Run alongside generate_ensemble.py. Only the small measurement journal is
read; map generation and its final 10000/5000 analysis are unchanged.
This preliminary validation is independent of these first-1000 estimates.
Its 500 maps later enter the full comparison ensemble, not its holdout.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import time
import numpy as np
from analyze_ensemble import coefficients, gaussian_orders, distinct_orders

PROJECT = Path(__file__).resolve().parents[2]


def read_rows(path):
    """Ignore an incomplete trailing write while the generator is running."""
    rows = []
    for line in path.read_text().splitlines():
        values = line.split()
        if values and not line.startswith('#') and len(values) == 20:
            rows.append([float(value) for value in values])
    return np.asarray(rows)


def main():
    folder = PROJECT / 'chains/gptg_ensemble_15000'
    output = folder / 'preliminary_1000_500'
    output.mkdir(exist_ok=True)
    journal = folder / 'comparison_measurements.txt'
    while len(read_rows(journal)) < 1000:
        time.sleep(30)
    comparison = read_rows(journal)[:1000]
    np.testing.assert_array_equal(comparison[:, 0], np.arange(1000))
    reference = np.loadtxt(folder / 'fixed_reference.txt')
    factor = np.linalg.cholesky(reference)
    inverse = np.linalg.solve(factor, np.eye(16))
    draws = comparison[:, 4:] @ inverse.T
    sample = np.cov(draws, rowvar=False, ddof=1)
    direct = np.linalg.solve(sample, np.eye(16))
    estimates = [direct, *gaussian_orders(sample, coefficients(999)),
                 *distinct_orders(draws[None, :, :])[:, 0]]
    labels = ['direct', 'gaussian2', 'gaussian4', 'gaussian6',
              'distinct2', 'distinct4', 'distinct6']
    # Compare precision weights in units where the direct covariance is I.
    # This is a same-sample comparison, not an error relative to truth.
    sample_factor = np.linalg.cholesky(sample)
    table = []
    for index, (label, precision) in enumerate(zip(labels, estimates)):
        eigenvalues = np.linalg.eigvalsh(sample_factor.T @ precision @ sample_factor)
        np.savetxt(output / f'precision_{label}_whitened.txt', precision)
        table.append([index, eigenvalues[0], eigenvalues[-1]])
        if eigenvalues[0] > 0:
            covariance = factor @ np.linalg.solve(precision, np.eye(16)) @ factor.T
            np.savetxt(output / f'covariance_{label}.txt', covariance)
        print(label, 'relative precision eigenvalue range', eigenvalues[[0, -1]], flush=True)
    np.savetxt(output / 'comparison.txt', table,
               header='method min_relative_precision max_relative_precision; ' + ','.join(labels))
    np.savetxt(output / 'comparison_measurements.txt', comparison)
    print('First 1000 frozen; waiting for 500 independent maps.', flush=True)
    while len(read_rows(journal)) < 1500:
        time.sleep(30)
    validation = read_rows(journal)[1000:1500]
    np.testing.assert_array_equal(validation[:, 0], np.arange(1000, 1500))
    assert not np.intersect1d(comparison[:, 1], validation[:, 1]).size
    check = validation[:, 4:] @ inverse.T
    check -= check.mean(axis=0)
    table = []
    for index, (label, precision) in enumerate(zip(labels, estimates)):
        minimum = np.linalg.eigvalsh(precision)[0]
        if minimum <= 0:
            table.append([index, minimum, np.nan, np.nan])
            continue
        # A lower held-out quadratic-minus-logdet score is better. No inverse
        # of the 500-map validation covariance is needed or treated as truth.
        quadratic = np.einsum('ni,ij,nj->n', check, precision-direct, check)
        score = quadratic.sum()/499
        score -= np.linalg.slogdet(precision)[1]-np.linalg.slogdet(direct)[1]
        error = quadratic.std(ddof=1)/np.sqrt(500)*500/499
        table.append([index, minimum, score, error])
        print(label, 'score minus direct, approximate SE:', score, error, flush=True)
    np.savetxt(output / 'validation_scores.txt', table,
               header='method min_eigenvalue score_minus_direct approximate_SE; ' + ','.join(labels))
    np.savetxt(output / 'validation_measurements.txt', validation)
    (output / 'COMPLETE.txt').write_text('1000 comparison + 500 independent validation; fixed initial GPTG model.\n')


if __name__ == '__main__':
    main()
