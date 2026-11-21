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
    transform,
    aperture_weights,
)


def generate_one(
    path,
    seed,
    record,
    data,
    indices,
    geometry,
    ell,
    factor,
    nodes,
    weights,
    roots,
    variance,
    first,
    second,
    workers,
    precision="float64",
    point_device="cpu",
    diagnostics=None,
    point_batch=1,
):
    """Generate one seed, atomically save float32, and measure decoded pixels.

    Serial and MPI drivers call this identical kernel. The seed depends on
    the global map number, never the MPI rank or task order. An existing map
    is read and remeasured after an interrupted save-to-journal transition.
    No physics, transform settings or random draw ordering depends on MPI.
    """
    # Single precision applies to the large map/harmonic arrays. Calibration
    # integrals and the common random stream remain double precision.
    if precision not in ("float64", "float32"):
        raise ValueError("precision must be float64 or float32")
    if point_device not in ("cpu", "metal"):
        raise ValueError("point_device must be cpu or metal")
    if point_batch not in (1, 4) or (point_batch == 4 and point_device != "metal"):
        raise ValueError("point_batch=4 requires the Metal point device")
    if point_device == "metal":
        if precision != "float32":
            raise ValueError("Metal point evaluation requires float32")
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError("Apple Metal GPU is unavailable")
    started = time.monotonic()
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
        if precision == "float32":
            alms = alms.astype(np.complex64)
        shear_maps = np.empty((4, 2, len(indices)))
        if point_batch == 4:
            # Batch the four source-bin sky fields into one GPU submission.
            # CPU transforms produce these fields first. Separate MPI workers
            # can still do CPU transforms while this rank uses the GPU.
            batch_start = time.monotonic()
            gaussian_batch = np.stack([
                ducc0.sht.synthesis(
                    alm=alms[source:source+1],
                    spin=0,
                    lmax=record['lmax'],
                    nthreads=workers,
                    **geometry,
                ) for source in range(4)
            ])
            batch_synthesis_end = time.monotonic()
            means = []
            normalizations = []
            for source in range(4):
                values = transform(
                    gaussian=nodes,
                    parameters=data['parameters'][source],
                )
                mean = weights @ values
                means.append(mean)
                normalizations.append(np.sqrt(weights @ (values-mean)**2))
            with mx.stream(mx.gpu):
                g = mx.array(gaussian_batch)
                parameters = mx.array(data['parameters'].astype(np.float32))
                a1 = parameters[:, 0, None, None]
                a2 = parameters[:, 1, None, None]
                b = parameters[:, 2, None, None]
                x0 = parameters[:, 3, None, None]
                transition = parameters[:, 4, None, None]
                mean = mx.array(np.asarray(means, dtype=np.float32)[:, None, None])
                norm = mx.array(np.asarray(normalizations, dtype=np.float32)[:, None, None])
                amplitude = mx.array(np.sqrt(variance).astype(np.float32)[:, None, None])
                value = (mx.exp(a1*g-a1*a1/2.)+b*g)*mx.exp(
                    (a2-a1)/transition*mx.logaddexp(
                        mx.array(0., dtype=mx.float32), transition*(g-x0),
                    )
                )
                kappa_gpu = amplitude*(value-mean)/norm
                mx.eval(kappa_gpu)
                kappa_batch = np.array(kappa_gpu)
            batch_point_end = time.monotonic()
            del gaussian_batch
        for source in range(4):
            if point_batch == 4:
                kappa = kappa_batch[source]
                # Amortize the batch stages over four bins for the profiler.
                stage_start = time.monotonic()
                synthesis_end = stage_start+(batch_synthesis_end-batch_start)/4.
                point_end = synthesis_end+(batch_point_end-batch_synthesis_end)/4.
            else:
                stage_start = time.monotonic()
                gaussian = ducc0.sht.synthesis(
                    alm=alms[source:source+1],
                    spin=0,
                    lmax=record['lmax'],
                    nthreads=workers,
                    **geometry,
                )
                synthesis_end = time.monotonic()
                if precision == "float64":
                    kappa = np.sqrt(variance[source])*normalized_transform(
                        gaussian=gaussian,
                        parameters=data['parameters'][source],
                        nodes=nodes,
                        weights=weights,
                    )
                else:
                    # The fixed Gaussian expectation is evaluated accurately once
                    # per bin; it must not be estimated from this random sky.
                    values = transform(
                        gaussian=nodes,
                        parameters=data['parameters'][source],
                    )
                    mean = weights @ values
                    normalization = np.sqrt(weights @ (values-mean)**2)
                    amplitude = np.float32(np.sqrt(variance[source]))
                    a1, a2, b, x0, transition = data['parameters'][source].astype(
                        np.float32,
                    )
                    if point_device == "cpu":
                        kappa = amplitude*(transform(
                            gaussian=gaussian,
                            parameters=data['parameters'][source].astype(np.float32),
                        )-np.float32(mean))/np.float32(normalization)
                    else:
                        # Only the point transformation runs on the GPU. ducc0's
                        # spin transforms and iterative solve still use CPU threads.
                        # Evaluate before timing/copying: MLX operations are lazy.
                        with mx.stream(mx.gpu):
                            g = mx.array(gaussian)
                            value = (mx.exp(float(a1)*g-float(a1*a1/2.))
                                     + float(b)*g)*mx.exp(
                                float((a2-a1)/transition)*mx.logaddexp(
                                    mx.array(0., dtype=mx.float32),
                                    float(transition)*(g-float(x0)),
                                )
                            )
                            kappa_gpu = float(amplitude)*(
                                value-float(np.float32(mean))
                            )/float(np.float32(normalization))
                            mx.eval(kappa_gpu)
                            kappa = np.array(kappa_gpu)
                point_end = time.monotonic()
            solve_start = time.monotonic()
            result = ducc0.sht.pseudo_analysis(
                map=kappa,
                spin=0,
                lmax=record['lmax'],
                maxiter=30,
                epsilon=1.e-6,
                nthreads=workers,
                **geometry,
            )
            if result[1] not in (1, 2):
                raise RuntimeError(f'Harmonic solver did not converge: {result[1:]}')
            solve_end = time.monotonic()
            shear_factor = factor.astype(np.float32) if precision == "float32" else factor
            shear = ducc0.sht.synthesis(
                alm=result[0]*shear_factor,
                spin=2,
                mode='GRAD_ONLY',
                lmax=record['lmax'],
                nthreads=workers,
                **geometry,
            )[:, indices]
            shear_end = time.monotonic()
            if diagnostics is not None:
                diagnostics.append([
                    source, synthesis_end-stage_start, point_end-synthesis_end,
                    solve_end-solve_start, shear_end-solve_end, result[1],
                    result[2], result[3], result[4],
                ])
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
    return made-started, elapsed, values


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
    parser.add_argument(
        '--precision',
        choices=('float64', 'float32'),
        default='float64',
        help='Arithmetic precision of large map/harmonic arrays; storage is float32.',
    )
    parser.add_argument(
        '--point-device',
        choices=('cpu', 'metal'),
        default='cpu',
        help='Device for nonlinear pixel transformation; ducc0 stays on CPU.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT / 'chains/gptg_ensemble_15000',
    )
    parser.add_argument(
        '--point-batch',
        type=int,
        choices=(1, 4),
        default=1,
        help='Source fields per Metal point-transform batch.',
    )
    args = parser.parse_args()
    if args.point_batch == 4 and args.point_device != 'metal':
        parser.error('--point-batch 4 requires --point-device metal')
    if args.point_device == 'metal' and args.precision != 'float32':
        parser.error('--point-device metal requires --precision float32')
    if not 1 <= args.workers <= 8 or not 1 <= args.limit <= 15000:
        parser.error('Use workers 1..8 and limit 1..15000; BLAS remains one.')
    # ducc0 uses its explicit native worker argument; keep the environment
    # consistent for any other OpenMP consumer without increasing BLAS.
    os.environ['OMP_NUM_THREADS'] = str(args.workers)
    folder = args.output.resolve()
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
        'compute_precision': args.precision,
        'point_device': args.point_device,
        'point_batch': args.point_batch,
        'map_shape': [4, 2, len(indices)],
        'status': 'initial fit; physical DES covariance not validated',
    }
    manifest = folder / 'ensemble.yaml'
    if manifest.exists():
        previous = yaml.safe_load(manifest.read_text())
        if (previous.get('compute_precision', 'float64') != args.precision
                or previous.get('point_device', 'cpu') != args.point_device):
            raise ValueError('Use a separate --output when changing arithmetic/device.')
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
                seed = seed_start+number
                generation_seconds, elapsed, values = generate_one(
                    path=path,
                    seed=seed,
                    record=record,
                    data=data,
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
                    workers=args.workers,
                    precision=args.precision,
                    point_device=args.point_device,
                    point_batch=args.point_batch,
                )
                text = ' '.join(f'{value:.17e}' for value in values)
                stream.write(f'{number} {seed} {generation_seconds:.8f} {elapsed:.8f} {text}\n')
                stream.flush()
                print(group, number+1, '/', count, 'seconds',
                      round(generation_seconds+elapsed, 3), 'elapsed_minutes',
                      round((time.monotonic()-start_run)/60., 2), flush=True)
    (folder / 'COMPLETE.txt').write_text(
        '10000 comparison and 5000 validation maps saved.\n'
        'Validation measurements have not been used for model/order selection.\n',
    )


if __name__ == '__main__':
    main()
