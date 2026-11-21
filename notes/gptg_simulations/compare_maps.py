"""Forward-test the initial GPTG generator with the same masked measurements.

The 32 supplied maps were also used for fitting. This is a diagnostic of
model mismatch, not an independent validation or a cosmological likelihood.
Generated maps are new random draws conditional on that fit. No map is
dropped according to its measured moments or covariance.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

from pathlib import Path
import numpy as np
import ducc0
import yaml
from generate_maps import PROJECT, OUTPUT, setup, pseudo_modes, spectra


def main():
    """Compare pseudo-spectra, reconstructed PDF and sixteen raw moments."""
    record = yaml.safe_load((OUTPUT/'calibration.yaml').read_text())
    reference = np.load(file=OUTPUT/'fit_measurements.npz')
    generated = np.load(file=OUTPUT/'generated_measurements.npz')
    lmax = record['lmax']
    indices, geometry, ell, factor, nodes, weights = setup(lmax=lmax)
    centers = reference['centers'][:, 0].astype(int)
    spectrum = np.zeros((2, 4, 4, lmax+1))
    reconstructed = []
    paths = sorted((OUTPUT/'maps').glob('GPTG_DES_*.npy'))
    inverse = np.zeros_like(factor)
    inverse[ell >= 2] = 1./factor[ell >= 2]
    smooth = np.exp(-ell*(ell+1)*np.deg2rad(10./60.)**2/2.)
    for number, path in enumerate(paths):
        shear = np.load(
            file=path,
            mmap_mode='r',
        )
        modes = []
        proxy = []
        for source in range(4):
            result = pseudo_modes(
                shear=shear[source],
                indices=indices,
                geometry=geometry,
                lmax=lmax,
                workers=4,
                second_sign=record['second_component_sign_for_ducc'],
            )
            modes.append(result)
            convergence = ducc0.sht.synthesis(
                alm=result[:1]*inverse*smooth,
                spin=0,
                lmax=lmax,
                nthreads=4,
                **geometry,
            )
            proxy.append(convergence[0, centers])
        modes = np.asarray(modes)
        reconstructed.append(proxy)
        for i in range(4):
            for j in range(i, 4):
                for parity in range(2):
                    value = spectra(modes[i, parity], modes[j, parity], ell, lmax)
                    spectrum[parity, i, j] += value
                    if i != j:
                        spectrum[parity, j, i] += value
        print('compared', number+1, '/', len(paths), flush=True)
    spectrum /= len(paths)
    np.savez_compressed(
        OUTPUT/'forward_measurements.npz',
        pseudo_cl=spectrum,
        reconstruction=reconstructed,
    )
    target_vectors = np.concatenate([
        np.mean(reference['aperture']**2, axis=-1).reshape(-1, 8),
        np.mean(reference['aperture']**3, axis=-1).reshape(-1, 8),
    ], axis=1)
    new_vectors = np.concatenate([
        np.mean(generated['aperture']**2, axis=-1).reshape(-1, 8),
        np.mean(generated['aperture']**3, axis=-1).reshape(-1, 8),
    ], axis=1)
    covariance_real = np.cov(
        m=target_vectors,
        rowvar=False,
        ddof=1,
    )
    covariance_new = np.cov(
        m=new_vectors,
        rowvar=False,
        ddof=1,
    )
    mean_real = target_vectors.mean(axis=0)
    mean_new = new_vectors.mean(axis=0)
    # This normalization only describes the observed difference. The fit
    # couples model parameters to the reference sample, so these are not
    # independent significance tests or calibrated p-values.
    difference_error = np.sqrt(np.diag(covariance_real)/len(target_vectors)
                               + np.diag(covariance_new)/len(new_vectors))
    ratio = np.diag(covariance_new)/np.diag(covariance_real)
    scales_real = np.sqrt(np.diag(covariance_real))
    scales_new = np.sqrt(np.diag(covariance_new))
    correlation_real = covariance_real/scales_real[:, None]/scales_real[None, :]
    correlation_new = covariance_new/scales_new[:, None]/scales_new[None, :]
    np.savetxt(OUTPUT/'reference_vectors.txt', target_vectors)
    np.savetxt(OUTPUT/'reference_covariance.txt', covariance_real)
    np.savetxt(OUTPUT/'generated_covariance.txt', covariance_new)
    np.savetxt(OUTPUT/'mean_comparison.txt', np.column_stack([
        np.arange(16), mean_real, mean_new, difference_error,
        (mean_new-mean_real)/difference_error, ratio,
    ]), header='row reference_mean GPTG_mean descriptive_difference_SE difference_over_SE variance_ratio')
    band_edges = np.unique(np.geomspace(10, lmax+1, 18).astype(int))
    power_rows = []
    multipoles = np.arange(lmax+1)
    for lower, upper in zip(band_edges[:-1], band_edges[1:]):
        take = (multipoles >= lower) & (multipoles < upper)
        weight = 2*multipoles[take]+1
        for i in range(4):
            for j in range(i, 4):
                for parity in range(2):
                    old = np.average(
                        reference['pseudo_cl'][parity, i, j, take],
                        weights=weight,
                    )
                    new = np.average(spectrum[parity, i, j, take], weights=weight)
                    power_rows.append([lower, upper, i, j, parity, old, new])
    np.savetxt(OUTPUT/'power_comparison.txt', power_rows,
               header='ell_low ell_high bin_i bin_j parity_E0_B1 reference GPTG')
    probabilities = np.array([.01, .02, .05, .1, .25, .5, .75, .9, .95, .98, .99])
    quantile_rows = []
    for source in range(4):
        old = reference['reconstruction'][:, source].ravel()
        new = np.asarray(reconstructed)[:, source].ravel()
        for probability, old_q, new_q in zip(
                probabilities, np.quantile(
                    old,
                    probabilities,
                ), np.quantile(
                    new,
                    probabilities,
                )):
            quantile_rows.append([source, probability, old_q, new_q])
    np.savetxt(OUTPUT/'quantile_comparison.txt', quantile_rows,
               header='source probability reference_kappa_proxy GPTG_kappa_proxy')
    print('variance ratios', ratio, flush=True)
    print('maximum correlation difference',
          np.max(np.abs(correlation_new-correlation_real)), flush=True)
    print('mean difference / descriptive error',
          (mean_new-mean_real)/difference_error, flush=True)


if __name__ == '__main__':
    main()
