"""Measure raw and noise-corrected tomographic aperture moments for the pilot.

This calculation reuses saved aperture values; it never rereads all maps.
The noise correction follows the *assumed* independent Gaussian pixel model.
That model is known for our generated maps, but is only an approximation
for Rafael's archive. These are fixed-center, same-radius aperture moments,
not his published finite-triangle catalogue estimator.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'

from itertools import combinations_with_replacement
import numpy as np
import yaml
from generate_maps import OUTPUT, PROJECT, aperture_weights


def main():
    """Write explicit source-bin labels, moments and the filter noise variance.

    For each filter, independent pixel noise adds in quadrature: its
    variance is sigma_n**2 times the sum of squared component weights.
    Repeated source indices in a cubic product require one subtraction
    for each possible noise pair. Distinct source indices have zero noise
    cross-covariance in this initial model.
    """
    record = yaml.safe_load((OUTPUT / 'calibration.yaml').read_text())
    sigma = np.asarray(record['noise_sigma_per_component'])
    indices = np.load(file=PROJECT / 'maps_cov_512_v2/indices_512.npy')
    first, second, centers = aperture_weights(indices=indices)
    # Sparse multiplication squares each stored weight without making
    # a dense 128-by-369190 matrix. Sum over pixels, retaining centers.
    weight_square = first.multiply(first) + second.multiply(second)
    weight_square = np.asarray(weight_square.sum(axis=1)).ravel()
    weight_square = weight_square.reshape(2, 64)
    variance = sigma[:, None, None]**2 * weight_square[None, :, :]
    np.savetxt(
        fname=OUTPUT / 'aperture_noise_variance.txt',
        X=variance.reshape(8, 64),
        header='Rows source-major, radii 30/60 arcmin; columns 64 centers.',
    )
    labels = []
    triples = list(combinations_with_replacement(range(4), 3))
    pairs = list(combinations_with_replacement(range(4), 2))
    for radius in (30, 60):
        for i, j in pairs:
            labels.append(f'2 {radius} {i+1} {j+1} 0')
        for i, j, k in triples:
            labels.append(f'3 {radius} {i+1} {j+1} {k+1}')
    (OUTPUT / 'tomographic_rows.txt').write_text(
        '# order radius_arcmin source_i source_j source_k (0 unused)\n'
        + '\n'.join(labels) + '\n',
    )
    for name, filename in (
        ('reference', 'fit_measurements.npz'),
        ('generated', 'generated_measurements.npz'),
    ):
        data = np.load(file=OUTPUT / filename)
        aperture = data['aperture']  # [map, source, radius, center]
        raw = []
        corrected = []
        for radius in range(2):
            for i, j in pairs:
                value = aperture[:, i, radius] * aperture[:, j, radius]
                noise_pair = np.zeros(64)
                if i == j:
                    noise_pair = variance[i, radius]
                raw.append(value.mean(axis=-1))
                corrected.append((value-noise_pair).mean(axis=-1))
            for i, j, k in triples:
                ai = aperture[:, i, radius]
                aj = aperture[:, j, radius]
                ak = aperture[:, k, radius]
                value = ai * aj * ak
                # E[A_i A_j A_k | signal] contains three possible
                # noise pairings. Each remaining factor is estimated
                # by its observed aperture, which has the correct mean.
                subtraction = np.zeros_like(value)
                if i == j:
                    subtraction += variance[i, radius] * ak
                if i == k:
                    subtraction += variance[i, radius] * aj
                if j == k:
                    subtraction += variance[j, radius] * ai
                raw.append(value.mean(axis=-1))
                corrected.append((value-subtraction).mean(axis=-1))
        raw = np.asarray(raw).T
        corrected = np.asarray(corrected).T
        np.savetxt(
            fname=OUTPUT / f'{name}_tomographic_raw.txt',
            X=raw,
        )
        np.savetxt(
            fname=OUTPUT / f'{name}_tomographic_corrected.txt',
            X=corrected,
        )
        print(name, raw.shape, 'raw/corrected second and third moments')
    print('Correction uses the stated pixel-noise model, not fitted moments.')


if __name__ == '__main__':
    main()
