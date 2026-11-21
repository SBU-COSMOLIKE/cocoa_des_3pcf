"""Save 10000 comparison and 5000 validation maps from one frozen GPTG fit.

This long run tests covariance estimators within the fitted distribution.
It does not assert that the initial fit is a validated DES model. The two
seed ranges are disjoint from each other and from the 64-map pilot. A text
journal is flushed after each saved map so an interrupted run can resume.
No second copy of the maps or full-sky temporary files is written to disk.
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'

import ducc0
import numpy as np
import yaml
from generate_maps import (
    PROJECT,
    OUTPUT,
    setup,
    gaussian_spectra,
    normalized_transform,
    aperture_weights,
)


def main():
    """Freeze the fit once, then generate or resume a serial map run with an explicit worker count.

    Each journal row names one completed file and contains its sixteen
    measured raw moments. Existing complete journal rows are reused; an
    unjournaled map is remeasured on resume. Partial .tmp files are replaced
    on the next attempt. A complete saved map is never overwritten.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=15000,
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.limit <= 15000:
        parser.error('Use workers 1..8 and limit 1..15000; BLAS remains one.')
    # ducc0 uses its explicit native worker argument; keep the environment
    # consistent for any other OpenMP consumer without increasing BLAS.
    os.environ['OMP_NUM_THREADS'] = str(args.workers)
    folder = PROJECT / 'chains/gptg_ensemble_15000'
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )
    # Copy the model only on the first launch. A later edit of the pilot
    # calibration cannot silently change the distribution midway through.
    for filename in ('calibration.yaml', 'calibration.npz'):
        destination = folder / filename
        if not destination.exists():
            shutil.copy2(OUTPUT / filename, destination)
    record = yaml.safe_load((folder / 'calibration.yaml').read_text())
    data = np.load(file=folder / 'calibration.npz')
    indices, geometry, ell, factor, nodes, weights = setup(lmax=record['lmax'])
    index_path = folder / 'indices_512.npy'
    if index_path.exists():
        np.testing.assert_array_equal(np.load(file=index_path), indices)
    else:
        np.save(
            file=index_path,
            arr=indices,
        )
    bytes_per_map = 4 * 2 * len(indices) * 4 + 128
    completed_paths = list((folder / 'comparison').glob('GPTG_DES_*.npy'))
    completed_paths += list((folder / 'validation').glob('GPTG_DES_*.npy'))
    remaining = max(0, args.limit-len(completed_paths)) * bytes_per_map
    # Reserve 50 GB for other work. The preflight checks before allocating
    # arrays; the same free-space reserve is checked before each map save.
    reserve = 50_000_000_000
    if shutil.disk_usage(folder).free < remaining+reserve:
        raise OSError('Insufficient space for requested maps plus a 50 GB reserve.')
    fit_hash = hashlib.sha256((folder / 'calibration.npz').read_bytes()).hexdigest()
    metadata = {
        'purpose': 'estimator benchmark conditional on the initial fitted GPTG model',
        'comparison_count': 10000,
        'validation_count': 5000,
        'comparison_seed_start': 341104759,
        'validation_seed_start': 441104759,
        'calibration_sha256': fit_hash,
        'workers': args.workers,
        'blas_workers': 1,
        'map_dtype': 'float32',
        'map_shape': [4, 2, len(indices)],
        'status': 'initial fit; physical DES covariance not validated',
    }
    manifest = folder / 'ensemble.yaml'
    if manifest.exists():
        previous = yaml.safe_load(manifest.read_text())
        if previous['calibration_sha256'] != fit_hash:
            raise ValueError('Frozen ensemble calibration changed; restore it before resuming.')
    manifest.write_text(yaml.safe_dump(
        data=metadata,
        sort_keys=False,
    ))
    first, second, centers = aperture_weights(indices=indices)
    latent, variance, repair, diagonal = gaussian_spectra(
        cl=data['cl_target'],
        parameters=data['parameters'],
        nodes=nodes,
        weights=weights,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(latent)
    roots = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.))[:, None, :]
    print('Frozen fit:', fit_hash, flush=True)
    print('Latent spectral repair:', repair, 'variances:', diagonal, flush=True)
    start_run = time.monotonic()
    processed = 0
    for group, count, seed_start in (
        ('comparison', 10000, 341104759),
        ('validation', 5000, 441104759),
    ):
        maps = folder / group
        maps.mkdir(exist_ok=True)
        journal = folder / f'{group}_measurements.txt'
        done = set()
        if journal.exists():
            for line in journal.read_text().splitlines():
                if line and not line.startswith('#'):
                    values = line.split()
                    if len(values) != 20:
                        raise ValueError(f'Incomplete journal row in {journal}; repair that final row before resuming.')
                    done.add(int(values[0]))
        with journal.open(
            mode='a',
            encoding='utf-8',
        ) as stream:
            if not done:
                stream.write('# number seed generation_seconds save_measure_seconds moment2[8] moment3[8]\n')
                stream.flush()
            for number in range(count):
                if processed >= args.limit:
                    return
                processed += 1
                path = maps / f'GPTG_DES_{number:06d}.npy'
                if number in done:
                    if not path.exists() or path.stat().st_size != bytes_per_map:
                        raise OSError(f'Journaled map is absent or has wrong size: {path}')
                    continue
                if shutil.disk_usage(folder).free < bytes_per_map+reserve:
                    raise OSError('Free disk reached the reserved 50 GB; resume after freeing space.')
                started = time.monotonic()
                seed = seed_start+number
                if path.exists():
                    # A completed file may precede its journal row if the
                    # process stopped between the atomic save and flush.
                    shear_maps = np.load(file=path).astype(np.float64)
                    if shear_maps.shape != (4, 2, len(indices)):
                        raise ValueError(f'Unexpected saved map shape: {path}')
                else:
                    rng = np.random.default_rng(seed=seed)
                    white = rng.normal(size=(len(ell), 4))
                    white = (white+1j*rng.normal(size=white.shape))/np.sqrt(2.)
                    white[:record['lmax']+1] = np.sqrt(2.)*white[:record['lmax']+1].real
                    alms = np.einsum('nij,nj->ni', roots[ell], white).T
                    shear_maps = np.empty((4, 2, len(indices)))
                    for source in range(4):
                        gaussian = ducc0.sht.synthesis(
                            alm=alms[source:source+1],
                            spin=0,
                            lmax=record['lmax'],
                            nthreads=args.workers,
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
                            nthreads=args.workers,
                            **geometry,
                        )
                        if result[1] not in (1, 2):
                            raise RuntimeError(f'Harmonic solver did not converge: {result[1:]}')
                        shear = ducc0.sht.synthesis(
                            alm=result[0]*factor,
                            spin=2,
                            mode='GRAD_ONLY',
                            lmax=record['lmax'],
                            nthreads=args.workers,
                            **geometry,
                        )[:, indices]
                        shear[1] *= record['second_component_sign_for_ducc']
                        shear += data['noise_sigma'][source]*rng.normal(size=shear.shape)
                        shear_maps[source] = shear
                made = time.monotonic()
                if not path.exists():
                    temporary = path.with_suffix('.npy.tmp')
                    # Only the stored pixels are rounded. Decode those same
                    # values to float64 before filtering so the journal can
                    # be reproduced exactly from the saved map.
                    stored_maps = shear_maps.astype(np.float32)
                    with temporary.open(mode='wb') as output:
                        np.save(
                            file=output,
                            arr=stored_maps,
                        )
                    temporary.replace(path)
                    shear_maps = stored_maps.astype(np.float64)
                aperture = first @ shear_maps[:, 0].T + second @ shear_maps[:, 1].T
                aperture = aperture.T.reshape(4, 2, 64)
                values = np.concatenate([
                    (aperture**2).mean(axis=2).ravel(),
                    (aperture**3).mean(axis=2).ravel(),
                ])
                elapsed = time.monotonic()-made
                text = ' '.join(f'{value:.17e}' for value in values)
                stream.write(f'{number} {seed} {made-started:.8f} {elapsed:.8f} {text}\n')
                stream.flush()
                print(group, number+1, '/', count, 'seconds',
                      round(time.monotonic()-started, 3), 'elapsed_minutes',
                      round((time.monotonic()-start_run)/60., 2), flush=True)
    (folder / 'COMPLETE.txt').write_text(
        '10000 comparison and 5000 validation maps saved.\n'
        'Validation measurements have not been used for model/order selection.\n',
    )


if __name__ == '__main__':
    main()
