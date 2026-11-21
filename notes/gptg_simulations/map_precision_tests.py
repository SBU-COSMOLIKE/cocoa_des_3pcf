"""Test precision algebra on a fixed empirical population of GPTG measurements.

These are independent resampling experiments CONDITIONAL on a frozen list
of 64 measured map vectors. Their true discrete-population covariance is
known exactly. Resampling is not additional physical skies, and this test
does not establish that GPTG has the same covariance as Rafael's simulations.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'notes/covariance_from_simulations'))
from precision_order_tests import coefficients, gaussian_orders, distinct_orders, targets, summarize


def main():
    """Use all 64 population members, with replacement, in each experiment."""
    population = np.loadtxt(
        fname=PROJECT/'tests/gptg_pilot/measurement_population.txt',
    )[:, [0, 6, 8, 14]]
    # Two squared and two cubed aperture moments retain a joint cross block.
    # Fixed unit conversion keeps tiny lensing units from obscuring the algebra.
    population = (population-population.mean(axis=0))/population.std(axis=0)
    truth = population.T @ population / len(population)
    # Denominator 64 is deliberate: the list IS the stipulated population,
    # not an unbiased estimate of an unknown continuous population covariance.
    model = 2.*truth + .1*np.eye(4)
    inverse = np.linalg.solve(
        a=np.linalg.cholesky(model),
        b=np.eye(4),
    )
    white_population = population @ inverse.T
    order_targets, exact = targets(covariance=inverse@truth@inverse.T)
    all_targets = np.concatenate([order_targets, order_targets])
    output = PROJECT/'chains/covariance_sixth_order'
    output.mkdir(
        exist_ok=True,
        parents=True,
    )
    rows = []
    for sample_count in (128, 512):
        weights = coefficients(degrees_of_freedom=sample_count-1)
        rng = np.random.default_rng(seed=618274)
        batches = []
        for start in range(0, 8192, 256):
            indices = rng.integers(
                low=0,
                high=len(population),
                size=(256, sample_count),
            )
            white = white_population[indices]
            centered = white-white.mean(
                axis=1,
                keepdims=True,
            )
            sample = centered.swapaxes(-1, -2)@centered/(sample_count-1)
            batches.append(np.concatenate([
                gaussian_orders(
                    sample=sample,
                    weights=weights,
                ),
                distinct_orders(vectors=white),
            ]))
        estimates = np.concatenate(batches, axis=1)
        summary = summarize(
            estimates=estimates,
            truth=all_targets,
            exact=exact,
        )
        print('sample_count', sample_count, 'Wishart2,4,6; distinct2,4,6\n',
              summary, flush=True)
        for method, row in enumerate(summary):
            rows.append([sample_count, method, *row])
    np.savetxt(
        fname=output/'gptg_empirical_summary.txt',
        X=rows,
        header=('sample_count method max_bias_over_MCSE relative_RMS_truncated '
                'relative_RMS_exact nonpositive_fraction bias00 MCSE00; '
                '8192 independent resamplings of a fixed 64-member population; '
                'NOT new independent N-body or full-sky maps'),
    )


if __name__ == '__main__':
    main()
