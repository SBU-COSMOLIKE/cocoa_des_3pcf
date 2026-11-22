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


MEASUREMENT_REVISION = 'shared-pixel-auto-polarization-v1'
VECTOR_HEADER = (f'measurement_revision={MEASUREMENT_REVISION}\n'
                 'xip[200] xim[200] Map3[80]; NaN means unfinished')


def check_measurement_revision(output, smoke_pixels=None):
    """Refuse older measurements before writing metadata or merging rows."""
    metadata = output / 'measurement.txt'
    expected = f'measurement_revision={MEASUREMENT_REVISION}'
    recorded_smoke = None
    if metadata.exists():
        lines = metadata.read_text().splitlines()
        if lines[:1] != [expected]:
            raise ValueError(f'Legacy measurement definition in {output}; use a fresh folder.')
        smoke = [line for line in lines if line.startswith('smoke_pixels=')]
        if len(smoke) != 1:
            raise ValueError(f'Missing smoke_pixels definition in {metadata}.')
        recorded_smoke = int(smoke[0].split(';', 1)[0].split('=', 1)[1])
        # A smoke run uses a different pixel subset. Even with the repaired
        # estimator, its rows must never be resumed as a full-footprint run.
        # Worker counts and requested map ranges may change on resubmission.
        if smoke_pixels is not None and recorded_smoke != smoke_pixels:
            raise ValueError(f'smoke_pixels mismatch in {output}; use a fresh folder.')
    for path in output.glob('map_*.txt'):
        with path.open() as stream:
            first_line = stream.readline().strip()
        if not metadata.exists() or first_line != '# ' + expected:
            raise ValueError(f'Legacy measurement vector {path}; use a fresh folder.')
    return recorded_smoke


def polarized_map3(catalogs, triple, radii, configuration, cache):
    """Measure one symmetric source triple through cached auto correlations.

    All four fields must share positions, pixel weights and the same mask.
    The three aperture radii must be equal at each requested radius. These
    are properties of this map runner, not assumptions for general catalogues.
    Only the four aperture values are cached; the dictionary is cleared for
    each new map. All source triples together require twenty auto evaluations.
    """
    first, second, third = triple
    unit = np.eye(4, dtype=int)
    # P(a) is the auto aperture moment of shear field a. The common triangle
    # counts and mean geometry do not depend on shear, so P is cubic even
    # after normalization. Equal radii make its trilinear form symmetric.
    # P(a+b) = P(a) + 3 T(a,a,b) + 3 T(a,b,b) + P(b).
    # P(a-b) = P(a) - 3 T(a,a,b) + 3 T(a,b,b) - P(b).
    # Subtracting leaves 6 T(a,a,b) + 2 P(b); adding leaves
    # 2 P(a) + 6 T(a,b,b). This explains both factors of six below.
    if first == third:
        terms = [(1., unit[first])]
    elif first == second:
        terms = [(1/6, unit[first] + unit[third]),
                 (-1/6, unit[first] - unit[third]),
                 (-1/3, unit[third])]
    elif second == third:
        terms = [(1/6, unit[first] + unit[second]),
                 (1/6, unit[first] - unit[second]),
                 (-1/3, unit[first])]
    else:
        # In P(a+b+c), subtracting all three pair sums removes the mixed
        # terms with only two source labels. Each pure cube was subtracted
        # twice, so add the three pure cubes back. The remaining six terms
        # are the permutations of T(a,b,c), which are equal here.
        terms = [(1/6, unit[first] + unit[second] + unit[third]),
                 (-1/6, unit[first] + unit[second]),
                 (-1/6, unit[first] + unit[third]),
                 (-1/6, unit[second] + unit[third]),
                 (1/6, unit[first]), (1/6, unit[second]),
                 (1/6, unit[third])]
    result = np.zeros(len(radii))
    for factor, coefficients in terms:
        key = tuple(coefficients)
        if key not in cache:
            if np.count_nonzero(coefficients) == 1:
                catalog = catalogs[int(np.argmax(coefficients))]
            else:
                # Catalogue shears already include the input flip_g2 sign.
                # Combining them must not apply that sign a second time.
                catalog = treecorr.Catalog(
                    ra=catalogs[0].ra,
                    dec=catalogs[0].dec,
                    ra_units='rad',
                    dec_units='rad',
                    g1=sum(c * cat.g1 for c, cat in zip(coefficients, catalogs)),
                    g2=sum(c * cat.g2 for c, cat in zip(coefficients, catalogs)),
                    w=catalogs[0].w,
                )
            corr = treecorr.GGGCorrelation(**configuration)
            # Always use the auto path: it excludes coincident pixels. The
            # mixed-field multipole paths retain unwanted self products when
            # different source fields occupy the same pixel positions.
            corr.process(
                cat1=catalog,
                num_threads=1,
            )
            sas = corr.toSAS(
                min_phi=.016,
                max_phi=3.126,
                nphi_bins=63,
                phi_units='radians',
            )
            cache[key] = sas.calculateMap3(R=radii)[0].copy()
        result += factor * cache[key]
    return result


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
        shards = sorted((folder / 'full_joint_480_shards').glob('*_*'))
        # Validate every input before changing any output. A repaired result
        # must never silently absorb rows from the old measurement definition.
        merge_smoke = check_measurement_revision(output)
        for shard in shards:
            shard_smoke = check_measurement_revision(shard, merge_smoke)
            if merge_smoke is None:
                merge_smoke = shard_smoke
        merged = 0
        for shard in shards:
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
                    if not (output / 'measurement.txt').exists():
                        (output / 'measurement.txt').write_text(
                            (shard / 'measurement.txt').read_text())
                    temporary = target.with_suffix('.tmp')
                    np.savetxt(temporary, existing,
                               header=VECTOR_HEADER)
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
        check_measurement_revision(output, args.smoke_pixels)
        (output / 'measurement.txt').write_text(
            f'measurement_revision={MEASUREMENT_REVISION}\n'
            f'480 rows: 200 xi-plus, 200 xi-minus, 80 Map3.\n'
            f'TreeCorr {treecorr.__version__}; NSIDE512 RING; equal pixel weights.\n'
            f'Side bank .5..80, 20 log bins, keep lower edge >=8: index {first_side}, '
            f'minimum {side_edges[first_side]:.17g} arcmin.\n'
            'Multipole max_n=100; 63 SAS bins, edges 0.016..3.126 radians.\n'
            'bin_slop=0; angle_slop=0; equal-radius auto polarization.\n'
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
            np.savetxt(temporary, values, header=VECTOR_HEADER)
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
    map3_cache = {}
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
                map3_cache.clear()
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
                configuration = dict(
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
                result = polarized_map3(
                    catalogs=catalogs,
                    triple=triples[job-10],
                    radii=radii,
                    configuration=configuration,
                    cache=map3_cache,
                )
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
