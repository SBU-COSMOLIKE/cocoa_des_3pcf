"""Measure all 400 xi and 80 finite-triangle Map3 entries with MPI.

Credits: Mike Jarvis and TreeCorr contributors provide the estimators and
aperture conversion. Binning follows Gomes et al. arXiv:2503.03964v2, with
an explicit lower-edge >=8 arcmin cut on the original 20-bin side bank.
NSIDE512 pixels have equal weights; these are not the original NSIDE1024
catalogue measurements. No claim of exact published covariance reproduction.

Rank zero dispatches whole maps with unfinished tomographic measurements and saves small
text vectors. Other ranks use one CPU each; BLAS and native threads are one.
With eight compute ranks plus four map-generation threads, the CPU budget
is twelve. Rank zero handles messages/files only. No map copies are saved.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
from collections import deque
import argparse
from itertools import combinations_with_replacement
from pathlib import Path
import time
import numpy as np
from astropy_healpix import HEALPix
from astropy.io import fits
from mpi4py import MPI
import treecorr

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=40000)
    parser.add_argument('--start', type=int, default=0,
                        help='First global map ID assigned to this job.')
    parser.add_argument('--stop', type=int, default=None,
                        help='Exclusive final global map ID; defaults to --count.')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Separate checkpoint directory for a concurrent map range.')
    parser.add_argument('--merge-shards', action='store_true',
                        help='After all jobs stop, combine saved shard entries into full_joint_480.')
    parser.add_argument('--maps-dir', type=Path,
                        default=PROJECT / 'chains/gptg_ensemble_15000')
    parser.add_argument('--smoke-pixels', type=int, default=0)
    parser.add_argument('--wait-for-ready', action='store_true',
                        help='Only dispatch maps with a completed-transfer .npy.ready marker.')
    args = parser.parse_args()
    if args.merge_shards:
        # Run once without mpirun, after every writer has stopped. Retain
        # saved entries from the original job and fill gaps from the shards.
        folder = args.maps_dir.resolve()
        output = folder / 'full_joint_480'
        merged = 0
        for shard in sorted((folder / 'full_joint_480_shards').glob('*_*')):
            first, stop = (int(part) for part in shard.name.split('_'))
            for path in sorted(shard.glob('map_*.txt')):
                number = int(path.stem.split('_')[1])
                if not first <= number < stop:
                    raise ValueError(f'Map ID outside shard range: {path}')
                incoming = np.loadtxt(path)
                target = output / path.name
                existing = np.loadtxt(target) if target.exists() else np.full(480, np.nan)
                if incoming.shape != (480,) or existing.shape != (480,):
                    raise ValueError(f'Unexpected vector shape: {path}')
                both = np.isfinite(incoming) & np.isfinite(existing)
                if not np.array_equal(incoming[both], existing[both]):
                    raise ValueError(f'Conflicting saved entries for map {number}')
                fill = np.isfinite(incoming) & ~np.isfinite(existing)
                if np.any(fill):
                    existing[fill] = incoming[fill]
                    output.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix('.tmp')
                    np.savetxt(temporary, existing,
                               header='xip[200] xim[200] Map3[80]; NaN means unfinished')
                    temporary.replace(target)
                    merged += 1
        print(f'Updated {merged} map vectors from shard checkpoints.')
        return
    if not 1 <= args.count <= 40000:
        raise ValueError('count must be between 1 and 40000.')
    if args.stop is None:
        args.stop = args.count
    if not 0 <= args.start < args.stop <= args.count:
        raise ValueError('Require 0 <= start < stop <= count.')
    if args.start and args.output_dir is None:
        raise ValueError('A nonzero start needs a separate --output-dir.')
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        raise ValueError('Use mpiexec with one coordinator and at least one worker.')
    folder = args.maps_dir.resolve()
    output = (args.output_dir.resolve() if args.output_dir is not None else
              folder / ('full_joint_smoke' if args.smoke_pixels else 'full_joint_480'))
    if args.output_dir is not None and output == folder / 'full_joint_480':
        raise ValueError('A range must not share the legacy output directory.')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    pairs = list(combinations_with_replacement(range(4), 2))
    triples = list(combinations_with_replacement(range(4), 3))
    radii = np.array([7., 14., 25., 40.])
    # Check row order against the bundled measurement files, not just length.
    data = PROJECT / 'data/upstream_y3/data/dv'
    xrows = fits.getdata(data / 'sim_2pt-NLA-cosmoCosmogrid-04Nov24.fits', extname='xip')
    yrows = fits.getdata(data / 'sim_map3_emu-NLA-cosmoCosmogrid-24FEBRUARY2025_WITH_COV.fits', extname='map3')
    for index, pair in enumerate(pairs):
        np.testing.assert_array_equal(xrows['BIN1'][20*index:20*index+20], pair[0]+1)
        np.testing.assert_array_equal(xrows['BIN2'][20*index:20*index+20], pair[1]+1)
    for index, triple in enumerate(triples):
        for j in range(3):
            np.testing.assert_array_equal(yrows[f'BIN{j+1}'][4*index:4*index+4], triple[j]+1)
        np.testing.assert_allclose(yrows['THETA1'][4*index:4*index+4], radii)
    side_edges = np.geomspace(.5, 80., 21)
    first_side = np.flatnonzero(side_edges >= 8.)[0]
    if rank == 0:
        (output / 'measurement.txt').write_text(
            f'480 rows: 200 xi-plus, 200 xi-minus, 80 Map3.\n'
            f'TreeCorr {treecorr.__version__}; NSIDE512 RING; equal pixel weights.\n'
            f'Side bank .5..80, 20 log bins, keep lower edge >=8: index {first_side}, '
            f'minimum {side_edges[first_side]:.17g} arcmin.\n'
            'Multipole max_n=100; 63 SAS bins, edges 0.016..3.126 radians.\n'
            'bin_slop=0; angle_slop=0; symmetrized tomographic legs.\n'
            f'{size-1} MPI compute ranks; one coordinator; BLAS=1; native threads=1.\n'
            f'smoke_pixels={args.smoke_pixels}; smoke outputs are not science vectors.\n')
        # Each task owns disjoint vector entries. Rank zero is the only writer.
        # Partial text vectors retain NaNs for unfinished tasks and can resume.
        pending = []
        vectors = {}
        # Sum worker task times separately for each map and measurement block.
        # These cover current-run work, not queue waits or previous allocations.
        block_seconds = {}
        for number in range(args.start, args.stop):
            path = output / f'map_{number:06d}.txt'
            values = np.loadtxt(path) if path.exists() else np.full(480, np.nan)
            if values.shape != (480,):
                raise ValueError(f'Unexpected vector shape: {path}')
            jobs = []
            for job in range(30):
                indices = np.r_[np.arange(20*job, 20*job+20), np.arange(200+20*job, 220+20*job)] if job < 10 else np.arange(400+4*(job-10), 404+4*(job-10))
                if not np.all(np.isfinite(values[indices])):
                    jobs.append(job)
            if jobs:
                pending.append((number, jobs))
                vectors[number] = values
        # A completion marker from a smaller prefix is not proof that the
        # enlarged target finished. Keep its history while new tasks run.
        if pending and (output / 'COMPLETE.txt').exists():
            (output / 'COMPLETE.txt').rename(output / f'COMPLETE.before_{args.count}.txt')
        # Workers receive the next available whole map, never a fixed map slice.
        # Readiness markers are written only after rsync succeeds. A visible
        # .npy alone may be an interrupted partial upload.
        waiting = pending
        available = deque()
        idle = list(range(1, size))
        active = 0
        next_scan = 0.
        while waiting or available or active:
            if waiting and time.monotonic() >= next_scan:
                ready = set()
                if args.wait_for_ready:
                    for group in ('comparison', 'validation'):
                        for marker in (folder / group).glob('GPTG_DES_*.npy.ready'):
                            local = int(marker.name.split('_')[-1].split('.')[0])
                            if group == 'comparison':
                                number = local if local < 10000 else local+5000
                            else:
                                number = local+10000 if local < 5000 else local+30000
                            ready.add(number)
                remaining = []
                for task in waiting:
                    if not args.wait_for_ready or task[0] in ready:
                        available.append(task)
                    else:
                        remaining.append(task)
                waiting = remaining
                next_scan = time.monotonic()+10.
            while idle and available:
                comm.send(available.popleft(), dest=idle.pop())
                active += 1
            if not comm.Iprobe(source=MPI.ANY_SOURCE):
                time.sleep(.02)
                continue
            status = MPI.Status()
            number, job, result, seconds = comm.recv(source=MPI.ANY_SOURCE, status=status)
            timing_key = (number, job < 10)
            block_seconds[timing_key] = block_seconds.get(timing_key, 0.0) + seconds
            values = vectors[number]
            # Announce a block only when this result completes it. Checking
            # the previous state also avoids repeating milestones on resume.
            block = values[:400] if job < 10 else values[400:]
            block_was_complete = np.all(np.isfinite(block))
            if job < 10:
                values[20*job:20*job+20] = result[:20]
                values[200+20*job:220+20*job] = result[20:]
            else:
                values[400+4*(job-10):404+4*(job-10)] = result
            path = output / f'map_{number:06d}.txt'
            temporary = path.with_suffix('.tmp')
            np.savetxt(temporary, values, header='xip[200] xim[200] Map3[80]; NaN means unfinished')
            temporary.replace(path)
            if not block_was_complete and np.all(np.isfinite(block)):
                print('map', number, '2pt done' if job < 10 else '3pt done',
                      'compute seconds this run', round(block_seconds.pop(timing_key), 3),
                      'completed entries',
                      f'{np.count_nonzero(np.isfinite(values))}/{values.size}',
                      flush=True)
            if np.all(np.isfinite(values)):
                del vectors[number]
                # Intermediate task messages checkpoint progress but do not
                # release the worker. It owns this map until all rows finish.
                idle.append(status.Get_source())
                active -= 1
        for worker in range(1, size):
            comm.send(None, dest=worker)
        (output / 'COMPLETE.txt').write_text(
            f'{args.stop - args.start} full joint vectors complete '
            f'for map IDs [{args.start}, {args.stop}).\n')
        return
    indices = np.load(folder / 'indices_512.npy')
    if args.smoke_pixels:
        indices = indices[:args.smoke_pixels]
    ra, dec = HEALPix(nside=512, order='ring').healpix_to_lonlat(indices)
    cached_number = None
    catalogs = None
    while True:
        task = comm.recv(source=0)
        if task is None:
            return
        number, jobs = task
        # Keep all thirty tomographic calculations for a map on one worker.
        # A resumed map receives only unfinished jobs. Send each result back
        # immediately so the coordinator can save it before the next job ends.
        for job in jobs:
            started = time.monotonic()
            if cached_number != number:
                # Preserve the original 15000 output IDs, including validation.
                # Append new comparison maps, then new validation maps. Moving
                # the old validation boundary would reuse results for wrong skies.
                if number < 10000:
                    group, local_number = 'comparison', number
                elif number < 15000:
                    group, local_number = 'validation', number-10000
                elif number < 35000:
                    group, local_number = 'comparison', number-5000
                else:
                    group, local_number = 'validation', number-30000
                path = folder / group / f'GPTG_DES_{local_number:06d}.npy'
                while not path.exists():
                    time.sleep(10)
                maps = np.load(path, mmap_mode='r')
                catalogs = []
                for source in range(4):
                    catalogs.append(treecorr.Catalog(
                        ra=ra.degree,
                        dec=dec.degree,
                        ra_units='deg',
                        dec_units='deg',
                        g1=np.array(maps[source, 0, :len(indices)], dtype=np.float64),
                        g2=np.array(maps[source, 1, :len(indices)], dtype=np.float64),
                        flip_g2=True,
                    ))
                cached_number = number
            if job < 10:
                first, second = pairs[job]
                corr = treecorr.GGCorrelation(
                    min_sep=2.5,
                    max_sep=250.,
                    nbins=20,
                    sep_units='arcmin',
                    metric='Arc',
                    bin_slop=0.,
                    angle_slop=0.,
                )
                corr.process(cat1=catalogs[first],
                             cat2=None if first == second else catalogs[second],
                             num_threads=1)
                result = np.r_[corr.xip, corr.xim]
            else:
                first, second, third = triples[job-10]
                corr = treecorr.GGGCorrelation(
                    bin_type='LogMultipole',
                    min_sep=float(side_edges[first_side]),
                    max_sep=80.,
                    nbins=20-first_side,
                    max_n=100,
                    sep_units='arcmin',
                    metric='Arc',
                    bin_slop=0.,
                    angle_slop=0.,
                )
                if first == third:
                    corr.process(cat1=catalogs[first], num_threads=1)
                else:
                    corr.process(cat1=catalogs[first], cat2=catalogs[second],
                                 cat3=catalogs[third], ordered=False, num_threads=1)
                sas = corr.toSAS(min_phi=.016, max_phi=3.126, nphi_bins=63, phi_units='radians')
                result = sas.calculateMap3(R=radii)[0]
            if result.shape != ((40,) if job < 10 else (4,)) or not np.isfinite(result).all():
                raise ValueError(f'Nonfinite or incorrectly sized measurement: map {number}, task {job}')
            comm.send((number, job, result, time.monotonic()-started), dest=0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
