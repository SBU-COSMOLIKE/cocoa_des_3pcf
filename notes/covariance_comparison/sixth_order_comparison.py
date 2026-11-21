"""Extend the same all-map comparison to nominal Gaussian sixth order.

No new maps or split groups enter. The input covariance and fixed reference
are identical to the second/fourth-order comparison. The nominal N=15997
is a file/vector count, not an established count of independent skies.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'notes/covariance_from_simulations'))
from precision_order_tests import coefficients, gaussian_orders


def main():
    """Save the three corrected covariances and their measured differences."""
    folder = PROJECT / 'tests/covariance_map_comparison'
    covariance = np.loadtxt(fname=folder / 'all_maps_covariance.txt')
    reference = np.loadtxt(fname=folder / 'aperture_reference.txt')
    factor = np.linalg.cholesky(reference)
    inverse = np.linalg.solve(a=factor, b=np.eye(16))
    sample = inverse @ covariance @ inverse.T
    weights = coefficients(degrees_of_freedom=15996)
    polynomials = gaussian_orders(sample=sample, weights=weights)
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / scale[:, None] / scale[None, :]
    sample_factor = np.linalg.cholesky(covariance)
    output = PROJECT / 'chains/covariance_sixth_order'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for order, polynomial in zip((2, 4, 6), polynomials):
        assert np.linalg.eigvalsh(polynomial)[0] > 0.
        precision = inverse.T @ polynomial @ inverse
        corrected = np.linalg.solve(a=precision, b=np.eye(16))
        corrected_scale = np.sqrt(np.diag(corrected))
        corrected_correlation = corrected / corrected_scale[:, None]
        corrected_correlation /= corrected_scale[None, :]
        deviation = corrected_scale / scale - 1.
        relative_precision = sample_factor.T @ precision @ sample_factor
        rows.append([
            order,
            deviation[:8].min(),
            deviation[:8].max(),
            np.max(np.abs(corrected_correlation-correlation)),
            np.linalg.eigvalsh(relative_precision)[0],
            np.trace(precision @ covariance) / 16.,
        ])
        np.savetxt(
            fname=output / f'all_covariance_order{order}.txt',
            X=corrected,
        )
    np.savetxt(
        fname=output / 'all_summary.txt',
        X=rows,
        header=('order min_second_std_shift max_second_std_shift '
                'max_correlation_difference min_relative_precision trace_per_row'),
    )
    print(np.asarray(rows))
    print('Nominal Gaussian count only; neither independence nor exact truth established.')


if __name__ == '__main__':
    main()
