"""Compare computation precision using common random skies and a frozen fit.

This tests arithmetic, not just float32 file storage. Calibration and random
numbers stay float64. The candidate uses complex64 harmonic coefficients,
float32 ducc0 arrays and a float32 nonlinear point transformation. The Metal
candidate changes only that point transformation; spherical transforms stay
on CPU. Identical shape-noise draws are also tested, but cannot replace the
noiseless comparison. Output lives in chains/compute_precision.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
from pathlib import Path
import time
import numpy as np
import yaml
from generate_ensemble import generate_one
from generate_maps import PROJECT, setup, gaussian_spectra, aperture_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--metal', action='store_true')
    parser.add_argument('--shape-noise', action='store_true')
    parser.add_argument('--point-batch', type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    source = PROJECT / 'chains/gptg_ensemble_15000'
    output = PROJECT / ('chains/compute_precision_noise' if args.shape_noise
                        else 'chains/compute_precision')
    if args.point_batch == 4:
        output = output.with_name(output.name+'_batch')
    output.mkdir(parents=True, exist_ok=True)
    record = yaml.safe_load((source / 'calibration.yaml').read_text())
    with np.load(source / 'calibration.npz') as archive:
        data = dict(archive)
    indices, geometry, ell, factor, nodes, weights = setup(lmax=record['lmax'])
    first, second, centers = aperture_weights(indices=indices)
    latent, variance, repair, diagonal = gaussian_spectra(
        cl=data['cl_target'],
        parameters=data['parameters'],
        nodes=nodes,
        weights=weights,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(latent)
    roots = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.))[:, None, :]
    modes = [('cpu64', 'float64', 'cpu'), ('cpu32', 'float32', 'cpu')]
    if args.metal:
        modes.append(('metal32', 'float32', 'metal'))
    rows = []
    stages = []
    moments = {name: [] for name, precision, device in modes}
    for number in range(args.count):
        seed = 641104759+number
        # Alternating the order reduces a systematic timing advantage from
        # running one implementation only while the laptop is cooler.
        order = modes if number % 2 == 0 else modes[::-1]
        maps = {}
        for name, precision, device in order:
            fit = dict(data)
            if not args.shape_noise:
                fit['noise_sigma'] = np.zeros_like(data['noise_sigma'])
            path = output / f'{name}_{number:03d}.npy'
            # This directory belongs only to this bounded diagnostic. Remove
            # its previous map so the timing cannot measure a cached read.
            path.unlink(missing_ok=True)
            diagnostics = []
            generation, measurement, values = generate_one(
                path=path,
                seed=seed,
                record=record,
                data=fit,
                indices=indices,
                geometry=geometry,
                ell=ell,
                factor=factor,
                nodes=nodes,
                weights=weights,
                roots=roots,
                variance=variance,
                first=first,
                second=second,
                workers=args.threads,
                precision=precision,
                point_device=device,
                diagnostics=diagnostics,
                point_batch=args.point_batch if device == "metal" else 1,
            )
            maps[name] = np.load(path).astype(np.float64)
            assert np.isfinite(maps[name]).all()
            assert all(row[5] in (1, 2) for row in diagnostics)
            stages.extend([[number, name]+row for row in diagnostics])
            moments[name].append(values)
            print(number, name, 'seconds', generation+measurement, flush=True)
        reference = maps['cpu64']
        for name, precision, device in modes[1:]:
            candidate = maps[name]
            # Error is normalized by signal RMS, never by a pixel that may
            # happen to cross zero. No shape noise dilutes this denominator.
            relative_rms = np.sqrt(np.mean((candidate-reference)**2, axis=(1, 2))
                                   / np.mean(reference**2, axis=(1, 2)))
            aperture_ref = first @ reference[:, 0].T + second @ reference[:, 1].T
            aperture_test = first @ candidate[:, 0].T + second @ candidate[:, 1].T
            # Moments through order six probe sensitivity of non-Gaussian
            # tails. Normalize errors by absolute moments to avoid divergent
            # relative errors when an odd signed moment nearly cancels.
            errors = []
            for power in range(2, 7):
                denominator = np.mean(np.abs(aperture_ref)**power, axis=0)
                errors.append(np.max(np.abs(np.mean(aperture_test**power
                                       - aperture_ref**power, axis=0))/denominator))
            rows.append([number, name, np.max(relative_rms)]+errors)
    with (output / 'errors.txt').open('w') as stream:
        stream.write('# seed_index mode max_signal_relative_rms aperture_absolute_moment_relative_errors_2_3_4_5_6\n')
        for row in rows:
            stream.write(' '.join(map(str, row))+'\n')
    with (output / 'stages.txt').open('w') as stream:
        stream.write('# seed_index mode source synthesis_s point_s solve_s shear_s stop iterations residual quality\n')
        for row in stages:
            stream.write(' '.join(map(str, row))+'\n')
    # This is a numerical screening gate, not a bound on the covariance of
    # an infinite ensemble. A future GPU harmonic transform needs its own test.
    passed = all(max(row[2:]) < 1.e-4 for row in rows)
    (output / 'summary.txt').write_text(
        f'{args.count} skies, shape_noise={args.shape_noise}, four bins, NSIDE512, lmax={record["lmax"]}\n'
        f'threads per CPU transform={args.threads}; BLAS=1\n'
        'Gate: signal relative RMS and normalized aperture moments 2..6 < 1e-4\n'
        f'passed={passed}\n'
    )
    if not passed:
        raise RuntimeError('Computation precision screening failed; see errors.txt')


if __name__ == '__main__':
    main()
