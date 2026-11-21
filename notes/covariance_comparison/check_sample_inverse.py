"""Measure stability of the direct inverse using whole run-label deletions.

The run label is a grouping convention, not independently confirmed parent
metadata. All cuts/draws of a label are deleted together. The fixed direction
is selected from the full-sample sixth-order comparison, so its scatter is
a descriptive sensitivity diagnostic, not an independent confidence test.
Only the existing compact 16-moment table is read; raw maps are untouched.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def main():
    """Keep the reported estimate on all files; delete groups only for checks."""
    table = np.loadtxt(fname=PROJECT / 'chains/covariance_maps_full/moments.txt')
    labels = table[:, 0].astype(int)
    vectors = table[:, 3:]
    covariance = np.cov(m=vectors, rowvar=False, ddof=1)
    frozen = np.loadtxt(
        fname=PROJECT / 'tests/covariance_map_comparison/all_maps_covariance.txt',
    )
    np.testing.assert_allclose(covariance, frozen, rtol=1.e-11, atol=0.)
    # Fixed coordinate scales improve conditioning without changing any
    # relative quadratic weight or introducing a fitted covariance model.
    scale = np.sqrt(np.diag(covariance))
    vectors = vectors / scale
    covariance = covariance / scale[:, None] / scale[None, :]
    sixth = np.loadtxt(
        fname=PROJECT / 'chains/covariance_sixth_order/all_covariance_order6.txt',
    )
    sixth = sixth / scale[:, None] / scale[None, :]
    factor = np.linalg.cholesky(covariance)
    precision_sixth = np.linalg.solve(a=sixth, b=np.eye(16))
    eigenvalues, eigenvectors = np.linalg.eigh(factor.T @ precision_sixth @ factor)
    direction = factor @ eigenvectors[:, 0]
    full_weight = direction @ np.linalg.solve(a=covariance, b=direction)
    groups = np.unique(labels)
    weights = []
    counts = []
    for label in groups:
        keep = labels != label
        leave_covariance = np.cov(m=vectors[keep], rowvar=False, ddof=1)
        weights.append(direction @ np.linalg.solve(a=leave_covariance, b=direction))
        counts.append(np.count_nonzero(labels == label))
    weights = np.asarray(weights)
    # The usual equal-group jackknife prefactor is an approximation when
    # group sizes differ. Here sizes are recorded rather than concealed.
    jackknife_se = np.sqrt((len(groups)-1)/len(groups)
                          * np.sum((weights-weights.mean())**2))
    output = PROJECT / 'chains/covariance_sixth_order'
    np.savetxt(
        fname=output / 'direct_inverse_deletions.txt',
        X=np.column_stack([groups, counts, weights/full_weight]),
        header='run_label files_in_group direct_inverse_weight_after_deletion / full_weight',
    )
    print('Groups:', len(groups), 'size range:', min(counts), max(counts))
    print('Full-sample direction weight:', full_weight)
    print('Sixth-order relative weight:', eigenvalues[0])
    print('Single-group deletion range:', weights.min()/full_weight, weights.max()/full_weight)
    print('Approximate group jackknife relative SE:', jackknife_se/full_weight)
    # Nested prefixes in one reproducible random group order describe how
    # the same fixed direction evolves. Their errors are correlated and the
    # largest prefix defines the denominator: equality at G=200 is automatic.
    rng = np.random.default_rng(seed=211126)
    order = rng.permutation(groups)
    rows = []
    for count in (25, 50, 100, 150, len(groups)):
        take = np.isin(labels, order[:count])
        partial = np.cov(m=vectors[take], rowvar=False, ddof=1)
        weight = direction @ np.linalg.solve(a=partial, b=direction)
        rows.append([count, take.sum(), weight/full_weight])
    np.savetxt(
        fname=output / 'direct_inverse_growth.txt',
        X=rows,
        header='run_labels measured_vectors fixed_direction_weight / all_sample_weight; nested correlated prefixes, seed211126',
    )
    print('Nested group counts, vector counts, relative weights:')
    print(np.asarray(rows))
    print('Neither stability nor the all-sample normalization establishes true survey precision.')


if __name__ == '__main__':
    main()
