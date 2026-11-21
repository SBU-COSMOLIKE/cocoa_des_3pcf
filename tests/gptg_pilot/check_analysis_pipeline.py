"""Exercise the ensemble-analysis script with known synthetic Gaussian vectors.

This bounded check needs no map files. It creates an isolated temporary copy
of the three numerical scripts and a 10000+5000-vector fixture, then checks
that per-realization quadratic scores equal direct covariance-trace scores.
Its numerical results are not measurements from the running GPTG ensemble.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def main():
    """Run all seven methods and remove only the temporary fixture afterwards."""
    with tempfile.TemporaryDirectory(prefix='gptg_analysis_check_') as name:
        temporary = Path(name)
        for relative in (
            'notes/gptg_simulations/analyze_ensemble.py',
            'notes/covariance_from_simulations/precision_order_tests.py',
            'notes/covariance_from_simulations/derive_wishart_powers.py',
        ):
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src=PROJECT / relative, dst=destination)
        folder = temporary / 'chains/gptg_ensemble_15000'
        folder.mkdir(parents=True, exist_ok=True)
        np.savetxt(fname=folder / 'fixed_reference.txt', X=2*np.eye(16))
        rng = np.random.default_rng(seed=181126)
        for group, count, seed in (
            ('comparison', 10000, 341104759),
            ('validation', 5000, 441104759),
        ):
            values = rng.normal(size=(count, 16))
            table = np.column_stack([
                np.arange(count),
                seed+np.arange(count),
                np.zeros((count, 2)),
                values,
            ])
            np.savetxt(fname=folder / f'{group}_measurements.txt', X=table)
        subprocess.run(
            args=[sys.executable, str(temporary / 'notes/gptg_simulations/analyze_ensemble.py')],
            check=True,
        )
        results = np.loadtxt(fname=folder / 'validation_scores.txt')
        values = np.loadtxt(fname=folder / 'validation_measurements.txt')[:, 4:]
        values = values / np.sqrt(2.)
        centered = values-values.mean(axis=0)
        covariance = centered.T @ centered / (len(values)-1)
        direct = np.loadtxt(fname=folder / 'precision_direct_whitened.txt')
        labels = ['direct', 'gaussian2', 'gaussian4', 'gaussian6',
                  'distinct2', 'distinct4', 'distinct6']
        for index, label in enumerate(labels):
            candidate = np.loadtxt(fname=folder / f'precision_{label}_whitened.txt')
            difference = np.trace((candidate-direct) @ covariance)
            difference -= np.linalg.slogdet(candidate)[1]
            difference += np.linalg.slogdet(direct)[1]
            np.testing.assert_allclose(results[index, 2], difference, atol=1.e-12)
        print('All seven methods completed; quadratic and trace scores agree.')


if __name__ == '__main__':
    main()
