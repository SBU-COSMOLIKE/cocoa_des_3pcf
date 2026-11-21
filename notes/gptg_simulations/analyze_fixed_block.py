"""Test the fixed production A on already measured Rafael map vectors.

The bounded vector has four source-bin-four xi-plus rows and eight cubic
aperture moments. A is the unmodified G+SSC+cNG production submatrix.
The distinct-pair construction needs independent primary run labels.
The producer's parent-label interpretation remains a condition of this test.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
from pathlib import Path
import argparse
import time
import numpy as np
from fixed_block_precision import estimates

PROJECT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gptg', action='store_true')
    args = parser.parse_args()
    folder = PROJECT / 'chains/gptg_ensemble_15000'
    output = folder / ('fixed_block_gptg1000' if args.gptg else 'fixed_block_rafael32')
    output.mkdir(exist_ok=True)
    production = np.load(folder / 'des_y3_full_covariance.npz')
    row = np.flatnonzero(np.all(production['rows'] == [0, 8, 8], axis=1))
    assert len(row) == 1
    indices = 20*row[0]+np.arange(12, 16)
    a = production['total'][np.ix_(indices, indices)]
    np.testing.assert_allclose(a, sum(production[key][np.ix_(indices, indices)]
                                     for key in ('gaussian', 'ssc', 'cng')))
    if args.gptg:
        while True:
            xi = np.atleast_2d(np.loadtxt(folder / 'joint_xip_measurements.txt'))
            if len(xi) >= 1000:
                break
            time.sleep(30)
        xi = xi[:1000]
        moments = np.loadtxt(folder / 'comparison_measurements.txt')
        vectors = []
        for record in xi:
            match = moments[moments[:, 0] == record[0]]
            assert len(match) == 1
            vectors.append(np.r_[record[2:], match[0, 12:20]])
    else:
        xi = np.loadtxt(PROJECT / 'tests/covariance_map_comparison/xip.txt')
        xi = xi[xi[:, 1] == 1]
        moments = np.loadtxt(PROJECT / 'chains/covariance_maps_full/moments.txt')
        moments = moments[(moments[:, 1] == 0) & (moments[:, 2] == 0)]
        vectors = []
        for record in xi:
            match = moments[moments[:, 0] == record[0]]
            assert len(match) == 1
            vectors.append(np.r_[record[2:], match[0, 11:19]])
    vectors = np.asarray(vectors)
    # The archived cubic reference is fixed before this test. Its quality
    # affects expansion convergence; it never changes the exact input A.
    reference = np.zeros((12, 12))
    reference[:4, :4] = a
    reference[4:, 4:] = 2*np.loadtxt(PROJECT / 'tests/covariance_map_comparison/aperture_reference.txt')[8:, 8:]
    sample = np.cov(vectors, rowvar=False)
    results, used = estimates(vectors, reference, 4)
    np.savetxt(output / 'joint_vectors.txt', vectors)
    np.savetxt(output / 'fixed_A.txt', a)
    np.savetxt(output / 'reference.txt', reference)
    np.savetxt(output / 'direct_covariance.txt', sample)
    scale = np.sqrt(np.diag(sample))
    records = []
    for order, raw, precision, covariance in results:
        np.savetxt(output / f'raw_precision_order{order}.txt', raw)
        if covariance is None:
            print(order, 'residual precision not positive; no constrained covariance')
            records.append([order, np.nan, np.nan, np.nan])
            continue
        np.savetxt(output / f'covariance_order{order}.txt', covariance)
        delta = (covariance-sample)/np.outer(scale, scale)
        record = [order, np.max(np.abs(delta[:4, 4:])),
                  np.max(np.abs(delta[4:, 4:])),
                  np.max(np.abs(covariance[:4, :4]-a))]
        records.append(record)
        print('order, max cross difference, max cubic difference, A change:', record)
    np.savetxt(output / 'comparison.txt', records,
               header=f'{used} primary maps; order max_normalized_cross_difference max_normalized_cubic_difference absolute_A_change; no independent validation yet')
    if args.gptg:
        print('First 1000 fixed; awaiting 500 matching independent measurements.', flush=True)
        while True:
            xi_check = np.atleast_2d(np.loadtxt(folder / 'joint_xip_measurements.txt'))
            if len(xi_check) >= 1500:
                break
            time.sleep(30)
        moments = np.loadtxt(folder / 'comparison_measurements.txt')
        check = []
        for record in xi_check[1000:1500]:
            match = moments[moments[:, 0] == record[0]]
            assert len(match) == 1
            check.append(np.r_[record[2:], match[0, 12:20]])
        check = np.asarray(check)
        # Score in dimensionless reference coordinates for numerical stability.
        factor = np.linalg.cholesky(reference)
        inverse = np.linalg.solve(factor, np.eye(12))
        check = (check-check.mean(axis=0)) @ inverse.T
        direct = np.linalg.inv(inverse @ sample @ inverse.T)
        scores = []
        for order, raw, precision, covariance in results:
            if precision is None:
                scores.append([order, np.nan, np.nan])
                continue
            candidate = factor.T @ precision @ factor
            q = np.einsum('ni,ij,nj->n', check, candidate-direct, check)
            score = q.sum()/499 - np.linalg.slogdet(candidate)[1] + np.linalg.slogdet(direct)[1]
            se = q.std(ddof=1)/np.sqrt(500)*500/499
            scores.append([order, score, se])
        np.savetxt(output / 'validation_scores.txt', scores,
                   header='order score_minus_direct approximate_SE; negative favors fixed-block candidate')
        print('Independent validation:', scores, flush=True)
    print('A is the production G+SSC+cNG submatrix; no simulation fit or perturbation.')


if __name__ == '__main__':
    main()
