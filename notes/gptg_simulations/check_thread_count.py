"""Regenerate ensemble map zero at eight workers and compare saved float32 pixels.

The stored reference was generated at four workers with the same frozen
model and seed. This is a bounded reproducibility check, not a new draw.
Nothing overwrites the saved map. BLAS stays at one worker.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'

import time
import numpy as np
import ducc0
import yaml
from generate_maps import PROJECT, setup, gaussian_spectra, normalized_transform
from generate_maps import aperture_weights


def main():
    """Report same-seed map and covariance-weighted moment differences."""
    folder = PROJECT / 'chains/gptg_ensemble_15000'
    record = yaml.safe_load((folder / 'calibration.yaml').read_text())
    data = np.load(file=folder / 'calibration.npz')
    indices, geometry, ell, factor, nodes, weights = setup(lmax=record['lmax'])
    latent, variance, repair, diagonal = gaussian_spectra(
        cl=data['cl_target'],
        parameters=data['parameters'],
        nodes=nodes,
        weights=weights,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(latent)
    roots = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.))[:, None, :]
    started = time.monotonic()
    rng = np.random.default_rng(seed=341104759)
    white = rng.normal(size=(len(ell), 4))
    white = (white+1j*rng.normal(size=white.shape))/np.sqrt(2.)
    white[:record['lmax']+1] = np.sqrt(2.)*white[:record['lmax']+1].real
    alms = np.einsum('nij,nj->ni', roots[ell], white).T
    result_maps = np.empty((4, 2, len(indices)))
    for source in range(4):
        gaussian = ducc0.sht.synthesis(
            alm=alms[source:source+1],
            spin=0,
            lmax=record['lmax'],
            nthreads=8,
            **geometry,
        )
        kappa = np.sqrt(variance[source])*normalized_transform(
            gaussian=gaussian,
            parameters=data['parameters'][source],
            nodes=nodes,
            weights=weights,
        )
        result = ducc0.sht.pseudo_analysis(
            map=kappa,
            spin=0,
            lmax=record['lmax'],
            maxiter=30,
            epsilon=1.e-6,
            nthreads=8,
            **geometry,
        )
        assert result[1] in (1, 2)
        shear = ducc0.sht.synthesis(
            alm=result[0]*factor,
            spin=2,
            mode='GRAD_ONLY',
            lmax=record['lmax'],
            nthreads=8,
            **geometry,
        )[:, indices]
        shear[1] *= record['second_component_sign_for_ducc']
        shear += data['noise_sigma'][source]*rng.normal(size=shear.shape)
        result_maps[source] = shear
    elapsed = time.monotonic()-started
    rounded = result_maps.astype(np.float32)
    original = np.load(file=folder / 'comparison/GPTG_DES_000000.npy')
    first, second, centers = aperture_weights(indices=indices)
    measured = []
    for maps in (original, rounded):
        maps = maps.astype(np.float64)
        aperture = first @ maps[:, 0].T + second @ maps[:, 1].T
        aperture = aperture.T.reshape(4, 2, 64)
        measured.append(np.concatenate([
            (aperture**2).mean(axis=2).ravel(),
            (aperture**3).mean(axis=2).ravel(),
        ]))
    covariance = np.loadtxt(fname=PROJECT / 'tests/gptg_pilot/generated_covariance.txt')
    difference = measured[1]-measured[0]
    whitened = np.linalg.solve(a=np.linalg.cholesky(covariance), b=difference)
    print('8-worker generation seconds:', elapsed)
    print('Changed saved float32 pixels:', np.count_nonzero(rounded != original))
    print('Maximum absolute saved-pixel change:', np.max(np.abs(rounded-original)))
    print('Covariance-weighted moment change:', np.linalg.norm(whitened))
    assert np.linalg.norm(whitened) < 1.e-6


if __name__ == '__main__':
    main()
