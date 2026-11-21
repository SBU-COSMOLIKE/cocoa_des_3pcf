"""Full 480-entry fixed-A comparison, followed by 500-map validation.

The 400x400 A is production G+SSC+cNG and is never calibrated on these maps.
The starting 80x80 D is the archived, simulation-estimated published matrix,
fixed independently of these newly generated maps. B starts at zero. This
reference is not an analytical three-point covariance. No tuning on holdout.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from fixed_block_precision import estimates

PROJECT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    folder = PROJECT / 'chains/gptg_ensemble_15000'
    measurements = folder / 'full_joint_480'
    output = folder / 'full_joint_comparison'
    output.mkdir(exist_ok=True)
    production = np.load(folder / 'des_y3_full_covariance.npz')
    rows = np.flatnonzero(production['rows'][:, 0] < 2)
    indices = (20*rows[:, None]+np.arange(20)).ravel()
    assert len(indices) == 400
    a = production['total'][np.ix_(indices, indices)]
    archived = fits.getdata(PROJECT / 'data/upstream_y3/data/covariance/xip_xim_map3_covariance_8ARCMINCUT_25Feb25.fits', extname='COVMAT')
    reference = np.zeros((480, 480))
    reference[:400, :400] = a
    reference[400:, 400:] = archived[400:, 400:]
    l = np.linalg.cholesky(reference)
    w = np.linalg.solve(l, np.eye(480))
    labels = ['direct', 'fixed_sample', 'order2', 'order4', 'order6']
    if not args.validate:
        vectors = np.array([np.loadtxt(measurements / f'map_{i:06d}.txt') for i in range(1000)])
        if vectors.shape != (1000, 480) or not np.isfinite(vectors).all():
            raise ValueError('Need 1000 completed 480-entry measurements.')
        sample = np.cov(vectors, rowvar=False, ddof=1)
        fixed_sample = sample.copy()
        fixed_sample[:400, :400] = a
        result, used = estimates(vectors, reference, 400)
        covariances = [sample, fixed_sample] + [item[3] for item in result]
        scales = np.sqrt(np.diag(sample))
        table = []
        for index, (label, covariance) in enumerate(zip(labels, covariances)):
            if covariance is None:
                table.append([index, np.nan, np.nan, np.nan, np.nan])
                continue
            white = w @ covariance @ w.T
            minimum = np.linalg.eigvalsh(white)[0]
            delta = (covariance-sample)/np.outer(scales, scales)
            table.append([index, minimum, np.max(np.abs(delta[:400, 400:])),
                          np.max(np.abs(delta[400:, 400:])),
                          np.max(np.abs(covariance[:400, :400]-a))])
            np.savetxt(output / f'covariance_{label}.txt', covariance)
            if minimum > 0:
                np.savetxt(output / f'precision_{label}_whitened.txt', np.linalg.solve(white, np.eye(480)))
        np.savetxt(output / 'reference.txt', reference)
        np.savetxt(output / 'comparison.txt', table,
                   header='method min_white_cov_eigenvalue max_normalized_B_difference max_normalized_D_difference max_A_change; '+','.join(labels))
        (output / 'FIRST1000_COMPLETE.txt').write_text('Full 480 dimensions; fixed reference; no validation used.\n')
        print('Full first-1000 comparison:', table, flush=True)
        return
    if not (output / 'FIRST1000_COMPLETE.txt').exists():
        raise ValueError('Freeze the first-1000 comparison before validating.')
    values = np.array([np.loadtxt(measurements / f'map_{i:06d}.txt') for i in range(1000, 1500)])
    if values.shape != (500, 480) or not np.isfinite(values).all():
        raise ValueError('Need 500 completed independent measurements.')
    centered = (values-values.mean(axis=0)) @ w.T
    direct = np.loadtxt(output / 'precision_direct_whitened.txt')
    table = []
    for index, label in enumerate(labels):
        path = output / f'precision_{label}_whitened.txt'
        if not path.exists():
            table.append([index, np.nan, np.nan])
            continue
        precision = np.loadtxt(path)
        quadratic = np.einsum('ni,ij,nj->n', centered, precision-direct, centered)
        score = quadratic.sum()/499-np.linalg.slogdet(precision)[1]+np.linalg.slogdet(direct)[1]
        error = quadratic.std(ddof=1)/np.sqrt(500)*500/499
        table.append([index, score, error])
    np.savetxt(output / 'validation_scores.txt', table,
               header='method score_minus_direct approximate_SE; '+','.join(labels))
    print('Full 500-map independent validation:', table, flush=True)


if __name__ == '__main__':
    main()
