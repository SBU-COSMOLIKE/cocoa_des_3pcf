"""Resume the configured map ensemble using independent MPI generation workers.

Rank zero owns journals and dispatches disjoint map numbers. Other ranks
call the same generate_one calculation defined below. Maps keep their
global seeds, float32 storage and atomic saves. Results are journaled in
map-number order even when workers finish out of order. An interruption
can leave a saved map ahead of its journal; resume remeasures that file.

Mixed laptop example: five MPI ranks = one coordinator + four workers,
with three transform threads per worker. --gpu-workers 1 selects one Metal
worker; --gpu-pipeline overlaps its CPU shear with next-bin scalar work.
BLAS always one. Worker-count and overlap timing tests use fresh maps.
One process computes directly, with no separate coordinator. The first launch
copies the pilot calibration if the output folder has no frozen copy yet.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
import yaml
from mpi4py import MPI
import ducc0
from generate_maps import (
    PROJECT, OUTPUT, setup, gaussian_spectra, aperture_weights,
    normalized_transform, transform,
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
    gpu_sht=None,
    gpu_pipeline=False,
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
    elif gpu_sht is not None:
        from gpu_map_generation import generate_gpu_map
        shear_maps, gpu_diagnostics = generate_gpu_map(
            sht=gpu_sht,
            seed=seed,
            record=record,
            data=data,
            roots=roots,
            variance=variance,
            ell=ell,
            factor=factor,
            indices=indices,
            geometry=geometry,
            nodes=nodes,
            weights=weights,
            threads=workers,
            pipeline=gpu_pipeline,
        )
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--comparison-count', type=int, default=10000)
    parser.add_argument('--validation-count', type=int, default=5000)
    parser.add_argument('--output', type=Path,
                        default=PROJECT / 'chains/gptg_ensemble_15000')
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
        '--point-batch',
        type=int,
        choices=(1, 4),
        default=1,
        help='Source fields per Metal point-transform batch.',
    )
    parser.add_argument(
        '--gpu-workers',
        type=int,
        choices=(0, 1, 2),
        default=0,
        help='Number of workers using Metal scalar SHT/solve and CPU shear synthesis.',
    )
    parser.add_argument('--gpu-pipeline', action='store_true',
                        help='Overlap GPU scalar work with CPU shear for the previous bin.')
    args = parser.parse_args()
    if args.gpu_pipeline and not args.gpu_workers:
        parser.error('--gpu-pipeline requires a GPU worker')
    if args.gpu_workers and args.precision != 'float32':
        parser.error('--gpu-workers requires --precision float32')
    if args.point_batch == 4 and args.point_device != 'metal':
        parser.error('--point-batch 4 requires --point-device metal')
    if args.point_device == 'metal' and args.precision != 'float32':
        parser.error('--point-device metal requires --precision float32')
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    total = args.comparison_count + args.validation_count
    if args.limit is None:
        args.limit = total
    if not 1 <= args.comparison_count <= 100000000 or args.validation_count < 1:
        raise ValueError('Positive group counts and disjoint seed ranges are required')
    if args.threads < 1 or not 1 <= args.limit <= total:
        raise ValueError('Use positive threads and a limit within group totals.')
    generation_workers = max(1, size-1)
    if args.gpu_workers > generation_workers:
        parser.error('--gpu-workers must not exceed the number of generation ranks')
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    folder = args.output.resolve()
    # Rank zero freezes the calibration before any worker reads it. This is
    # the same first-launch preparation for one process and for many.
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        for filename in ('calibration.yaml', 'calibration.npz'):
            destination = folder / filename
            if not destination.exists():
                shutil.copy2(OUTPUT / filename, destination)
    comm.Barrier()
    record = yaml.safe_load((folder / 'calibration.yaml').read_text())
    fit_hash = hashlib.sha256((folder / 'calibration.npz').read_bytes()).hexdigest()
    reserve = 50_000_000_000
    counts = [('comparison', args.comparison_count, 341104759),
              ('validation', args.validation_count, 441104759)]
    # With one MPI process, rank zero also computes maps. Otherwise only
    # worker ranks allocate the transform tables; the coordinator stays light.
    if size == 1 or rank != 0:
        data = np.load(folder / 'calibration.npz')
        indices, geometry, ell, factor, nodes, weights = setup(lmax=record['lmax'])
        first, second, centers = aperture_weights(indices=indices)
        latent, variance, repair, diagonal = gaussian_spectra(
            cl=data['cl_target'], parameters=data['parameters'], nodes=nodes, weights=weights)
        eigenvalues, eigenvectors = np.linalg.eigh(latent)
        roots = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.))[:, None, :]
        gpu_sht = None
        # Last ranks own independent Metal contexts. They can overlap CPU shear
        # stages, but compete for GPU bandwidth; measure aggregate throughput.
        if args.gpu_workers and (size == 1 or rank >= size-args.gpu_workers):
            from gpu_scalar_sht import MetalScalarSHT
            gpu_sht = MetalScalarSHT(geometry, record['lmax'])
    if rank == 0:
        manifest = folder / 'ensemble.yaml'
        metadata = yaml.safe_load(manifest.read_text()) if manifest.exists() else {}
        if metadata.get('calibration_sha256', fit_hash) != fit_hash:
            raise ValueError('Frozen calibration hash changed; do not mix ensembles.')
        indices = np.load(PROJECT / 'maps_cov_512_v2/indices_512.npy')
        if (folder / 'indices_512.npy').exists():
            np.testing.assert_array_equal(np.load(folder / 'indices_512.npy'), indices)
        else:
            np.save(folder / 'indices_512.npy', indices)
        bytes_per_map = 4*2*len(indices)*4+128
        pending = []
        streams = {}
        completed = 0
        offset = 0
        for group, count, seed_start in counts:
            (folder / group).mkdir(exist_ok=True)
            journal = folder / f'{group}_measurements.txt'
            done = set()
            if journal.exists():
                for line in journal.read_text().splitlines():
                    if not line or line.startswith('#'):
                        continue
                    columns = line.split()
                    if len(columns) != 20:
                        raise ValueError(f'Incomplete journal row: {journal}')
                    number = int(columns[0])
                    if number in done or int(columns[1]) != seed_start+number:
                        raise ValueError(f'Duplicate map or wrong seed: {journal}')
                    path = folder / group / f'GPTG_DES_{number:06d}.npy'
                    if not path.exists() or path.stat().st_size != bytes_per_map:
                        raise ValueError(f'Journaled map absent or wrong size: {path}')
                    done.add(number)
            for number in range(min(count, max(0, args.limit-offset))):
                if number in done:
                    completed += 1
                else:
                    pending.append((group, number, seed_start+number))
            streams[group] = journal.open('a', buffering=1)
            if journal.stat().st_size == 0:
                streams[group].write('# number seed generation_seconds save_measure_seconds moment2[8] moment3[8]\n')
            offset += count
        unsaved = sum(not (folder/g/f'GPTG_DES_{n:06d}.npy').exists() for g,n,s in pending)
        if shutil.disk_usage(folder).free < unsaved*bytes_per_map+reserve:
            raise OSError('Need space for remaining float32 maps plus a 50 GB reserve.')
        previous_arithmetic = {
            'precision': metadata.get('compute_precision', 'float64'),
            'point_device': metadata.get('point_device', 'cpu'),
            'point_batch': metadata.get('point_batch', 1),
            'gpu_workers': metadata.get('gpu_workers', 0),
            'gpu_pipeline': metadata.get('gpu_pipeline', False),
        }
        next_arithmetic = {
            'precision': args.precision,
            'point_device': args.point_device,
            'point_batch': args.point_batch,
            'gpu_workers': args.gpu_workers,
            'gpu_pipeline': args.gpu_pipeline,
        }
        if metadata and previous_arithmetic != next_arithmetic:
            # Keep the boundary explicit when a validated numerical backend
            # changes. Existing maps, including completed but unjournaled
            # saves, are never regenerated under the new arithmetic.
            metadata.setdefault('arithmetic_history', []).append({
                'after_journaled_maps': completed,
                'previous': previous_arithmetic,
                'next': next_arithmetic,
                'existing_unjournaled_maps_kept': [
                    f'{group}/GPTG_DES_{number:06d}.npy'
                    for group, number, seed in pending
                    if (folder / group / f'GPTG_DES_{number:06d}.npy').exists()
                ],
            })
        metadata.update(calibration_sha256=fit_hash, mpi_ranks=size,
                        generation_workers=generation_workers, workers_per_rank=args.threads,
                        gpu_workers=args.gpu_workers,
                        gpu_pipeline=args.gpu_pipeline,
                        blas_workers=1, map_dtype='float32',
                        compute_precision=args.precision, point_device=args.point_device,
                        point_batch=args.point_batch,
                        map_shape=[4, 2, len(indices)],
                        comparison_count=args.comparison_count,
                        validation_count=args.validation_count,
                        comparison_seed_start=341104759, validation_seed_start=441104759)
        # An earlier smaller ensemble may already have passed its audit.
        # Archive those records before extending it so they cannot be mistaken
        # for completion of the new target. Existing maps stay untouched.
        if pending:
            for marker in ('COMPLETE.txt', 'COMPLETENESS.txt'):
                old = folder / marker
                if old.exists():
                    old.rename(folder / f'{marker}.before_{total}')
        manifest.write_text(yaml.safe_dump(metadata, sort_keys=False))
        with (folder / 'worker_history.txt').open('a') as history:
            history.write(f'MPI resume after {completed} journaled maps: {generation_workers} generation ranks x {args.threads} transform threads; BLAS=1; precision={args.precision}; point_device={args.point_device}; point_batch={args.point_batch}.\n')
        arithmetic_stream = (folder / 'arithmetic_workers.txt').open('a', buffering=1)
        if arithmetic_stream.tell() == 0:
            arithmetic_stream.write('# group map_number worker_rank arithmetic; recovered means existing file kept\n')
        iterator = iter(enumerate(pending))
        active = 0
        for worker in range(1, size):
            task = next(iterator, None)
            comm.send(task, dest=worker)
            active += task is not None
        if size == 1:
            active = len(pending)
        buffered = {}
        next_row = 0
        start = time.monotonic()
        while active:
            if size == 1:
                # No messages to ourselves: do one map, then use the same
                # ordered journal writer as the multi-process path.
                position, (group, number, seed) = next(iterator)
                path = folder / group / f'GPTG_DES_{number:06d}.npy'
                if shutil.disk_usage(folder).free < reserve+4*2*len(indices)*4+128:
                    raise OSError('Free space reached the 50 GB reserve.')
                arithmetic = ('recovered' if path.exists() else
                              ('metal_sht_float32_pipeline' if args.gpu_pipeline
                               else 'metal_sht_float32') if gpu_sht is not None else
                              f'{args.point_device}_{args.precision}_batch{args.point_batch}')
                generation, measurement, values = generate_one(
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
                    workers=args.threads,
                    precision=args.precision,
                    point_device=args.point_device,
                    point_batch=args.point_batch,
                    gpu_sht=gpu_sht,
                    # CPU-only ranks ignore the overlap flag. On a Metal rank it
                    # overlaps stages inside this one map; map ownership stays with
                    # the MPI worker until save and measurement have both finished.
                    gpu_pipeline=args.gpu_pipeline,
                )
                worker_rank = 0
            else:
                # Blocking MPI receive can spin a CPU. Poll with a short sleep so
                # the coordinator does not compete with the twelve compute cores.
                while not comm.Iprobe(source=MPI.ANY_SOURCE):
                    time.sleep(.02)
                status = MPI.Status()
                position, generation, measurement, values, arithmetic = comm.recv(source=MPI.ANY_SOURCE, status=status)
                worker_rank = status.Get_source()
            finished_rank = worker_rank
            buffered[position] = (generation, measurement, values, worker_rank, arithmetic)
            while next_row in buffered:
                generation, measurement, values, worker_rank, arithmetic = buffered.pop(next_row)
                group, number, seed = pending[next_row]
                arithmetic_stream.write(f'{group} {number} {worker_rank} {arithmetic}\n')
                text = ' '.join(f'{value:.17e}' for value in values)
                streams[group].write(f'{number} {seed} {generation:.8f} {measurement:.8f} {text}\n')
                completed += 1
                next_row += 1
                print('saved', completed, '/', args.limit, group, number,
                      'elapsed_seconds', round(time.monotonic()-start, 3), flush=True)
            # Dispatch by the rank that actually finished, not by map ID.
            # Ordered journal writes do not impose an all-worker barrier:
            # faster workers receive more maps while earlier IDs are buffered.
            if size == 1:
                active -= 1
            else:
                task = next(iterator, None)
                comm.send(task, dest=finished_rank)
                if task is None:
                    active -= 1
        arithmetic_stream.close()
        for stream in streams.values():
            stream.close()
        if args.limit == total:
            from verify_ensemble import verify_ensemble
            verify_ensemble(folder)
            (folder / 'COMPLETE.txt').write_text(
                f'{args.comparison_count} comparison and {args.validation_count} validation float32 maps saved.\n')
        print('MPI generation target complete:', completed, flush=True)
        return
    while True:
        task = comm.recv(source=0)
        if task is None:
            return
        position, (group, number, seed) = task
        path = folder / group / f'GPTG_DES_{number:06d}.npy'
        if shutil.disk_usage(folder).free < reserve+4*2*len(indices)*4+128:
            raise OSError('Free space reached the 50 GB reserve.')
        arithmetic = ('recovered' if path.exists() else
                      ('metal_sht_float32_pipeline' if args.gpu_pipeline
                       else 'metal_sht_float32') if gpu_sht is not None else
                      f'{args.point_device}_{args.precision}_batch{args.point_batch}')
        generation, measurement, values = generate_one(
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
            workers=args.threads,
            precision=args.precision,
            point_device=args.point_device,
            point_batch=args.point_batch,
            gpu_sht=gpu_sht,
            # CPU-only ranks ignore the overlap flag. On a Metal rank it
            # overlaps stages inside this one map; map ownership stays with
            # the MPI worker until save and measurement have both finished.
            gpu_pipeline=args.gpu_pipeline,
        )
        comm.send((position, generation, measurement, values, arithmetic), dest=0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
