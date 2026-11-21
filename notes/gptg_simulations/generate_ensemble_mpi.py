"""Resume the 15000-map ensemble using independent MPI generation workers.

Rank zero owns journals and dispatches disjoint map numbers. Other ranks
call exactly the serial generator's generate_one kernel. Maps keep their
global seeds, float32 storage and atomic saves. Results are journaled in
map-number order even when workers finish out of order. An interruption
can leave a saved map ahead of its journal; resume remeasures that file.

Laptop example: four MPI ranks = one coordinator + three workers, with
four transform threads per worker (twelve compute cores). BLAS always one.
The output directory must already contain the frozen calibration files.
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
from generate_ensemble import generate_one
from generate_maps import PROJECT, setup, gaussian_spectra, aperture_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int, default=15000)
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
        choices=(0, 1),
        default=0,
        help='Use one worker for Metal scalar SHT/solve and CPU shear synthesis.',
    )
    args = parser.parse_args()
    if args.gpu_workers and args.precision != 'float32':
        parser.error('--gpu-workers 1 requires --precision float32')
    if args.point_batch == 4 and args.point_device != 'metal':
        parser.error('--point-batch 4 requires --point-device metal')
    if args.point_device == 'metal' and args.precision != 'float32':
        parser.error('--point-device metal requires --precision float32')
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2 or args.threads < 1 or not 1 <= args.limit <= 15000:
        raise ValueError('Use at least two ranks, positive threads, and limit 1..15000.')
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    folder = args.output.resolve()
    record = yaml.safe_load((folder / 'calibration.yaml').read_text())
    fit_hash = hashlib.sha256((folder / 'calibration.npz').read_bytes()).hexdigest()
    reserve = 50_000_000_000
    counts = [('comparison', 10000, 341104759), ('validation', 5000, 441104759)]
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
        }
        next_arithmetic = {
            'precision': args.precision,
            'point_device': args.point_device,
            'point_batch': args.point_batch,
            'gpu_workers': args.gpu_workers,
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
                        generation_workers=size-1, workers_per_rank=args.threads,
                        gpu_workers=args.gpu_workers,
                        blas_workers=1, map_dtype='float32',
                        compute_precision=args.precision, point_device=args.point_device,
                        point_batch=args.point_batch,
                        map_shape=[4, 2, len(indices)],
                        comparison_count=10000, validation_count=5000,
                        comparison_seed_start=341104759, validation_seed_start=441104759)
        manifest.write_text(yaml.safe_dump(metadata, sort_keys=False))
        with (folder / 'worker_history.txt').open('a') as history:
            history.write(f'MPI resume after {completed} journaled maps: {size-1} generation ranks x {args.threads} transform threads; BLAS=1; precision={args.precision}; point_device={args.point_device}; point_batch={args.point_batch}.\n')
        arithmetic_stream = (folder / 'arithmetic_workers.txt').open('a', buffering=1)
        if arithmetic_stream.tell() == 0:
            arithmetic_stream.write('# group map_number worker_rank arithmetic; recovered means existing file kept\n')
        iterator = iter(enumerate(pending))
        active = 0
        for worker in range(1, size):
            task = next(iterator, None)
            comm.send(task, dest=worker)
            active += task is not None
        buffered = {}
        next_row = 0
        start = time.monotonic()
        while active:
            # Blocking MPI receive can spin a CPU. Poll with a short sleep so
            # the coordinator does not compete with the twelve compute cores.
            while not comm.Iprobe(source=MPI.ANY_SOURCE):
                time.sleep(.02)
            status = MPI.Status()
            position, generation, measurement, values, arithmetic = comm.recv(source=MPI.ANY_SOURCE, status=status)
            buffered[position] = (generation, measurement, values, status.Get_source(), arithmetic)
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
            task = next(iterator, None)
            comm.send(task, dest=status.Get_source())
            if task is None:
                active -= 1
        arithmetic_stream.close()
        for stream in streams.values():
            stream.close()
        if args.limit == 15000:
            from verify_ensemble import verify_ensemble
            verify_ensemble(folder)
            (folder / 'COMPLETE.txt').write_text('10000 comparison and 5000 validation float32 maps saved.\n')
        print('MPI generation target complete:', completed, flush=True)
        return
    data = np.load(folder / 'calibration.npz')
    indices, geometry, ell, factor, nodes, weights = setup(lmax=record['lmax'])
    first, second, centers = aperture_weights(indices=indices)
    latent, variance, repair, diagonal = gaussian_spectra(
        cl=data['cl_target'], parameters=data['parameters'], nodes=nodes, weights=weights)
    eigenvalues, eigenvectors = np.linalg.eigh(latent)
    roots = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.))[:, None, :]
    gpu_sht = None
    if args.gpu_workers and rank == size-1:
        from gpu_scalar_sht import MetalScalarSHT
        gpu_sht = MetalScalarSHT(geometry, record['lmax'])
    while True:
        task = comm.recv(source=0)
        if task is None:
            return
        position, (group, number, seed) = task
        path = folder / group / f'GPTG_DES_{number:06d}.npy'
        if shutil.disk_usage(folder).free < reserve+4*2*len(indices)*4+128:
            raise OSError('Free space reached the 50 GB reserve.')
        arithmetic = ('recovered' if path.exists() else
                      'metal_sht_float32' if gpu_sht is not None else
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
        )
        comm.send((position, generation, measurement, values, arithmetic), dest=0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
