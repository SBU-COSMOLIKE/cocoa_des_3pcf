"""Measure all 400 xi and 80 finite-triangle Map3 entries with MPI.

Credits: Mike Jarvis and TreeCorr contributors provide the estimators and
aperture conversion. Binning follows Gomes et al. arXiv:2503.03964v2, with
an explicit lower-edge >=8 arcmin cut on the original 20-bin side bank.
NSIDE512 pixels have equal weights; these are not the original NSIDE1024
catalogue measurements. No claim of exact published covariance reproduction.

Rank zero dispatches independent tomographic measurements and saves small
text vectors. Other ranks use one CPU each; BLAS and native threads are one.
With eight compute ranks plus four map-generation threads, the CPU budget
is twelve. Rank zero handles messages/files only. No map copies are saved.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from itertools import combinations_with_replacement
from pathlib import Path
import time
import numpy as np
from astropy_healpix import HEALPix
from astropy.io import fits
from mpi4py import MPI
import treecorr

PROJECT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=1500)
    parser.add_argument('--smoke-pixels', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.count <= 15000:
        raise ValueError('count must be between 1 and 15000.')
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        raise ValueError('Use mpiexec with one coordinator and at least one worker.')
    folder = PROJECT / 'chains/gptg_ensemble_15000'
    output = folder / ('full_joint_smoke' if args.smoke_pixels else 'full_joint_480')
    if rank == 0:
        output.mkdir(exist_ok=True)
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
        for number in range(args.count):
            path = output / f'map_{number:06d}.txt'
            values = np.loadtxt(path) if path.exists() else np.full(480, np.nan)
            if values.shape != (480,):
                raise ValueError(f'Unexpected vector shape: {path}')
            for job in range(30):
                indices = np.r_[np.arange(20*job, 20*job+20), np.arange(200+20*job, 220+20*job)] if job < 10 else np.arange(400+4*(job-10), 404+4*(job-10))
                if not np.all(np.isfinite(values[indices])):
                    pending.append((number, job))
            if not np.all(np.isfinite(values)):
                vectors[number] = values
        iterator = iter(pending)
        active = 0
        for worker in range(1, size):
            task = next(iterator, None)
            comm.send(task, dest=worker)
            active += task is not None
        while active:
            while not comm.Iprobe(source=MPI.ANY_SOURCE):
                time.sleep(.02)
            status = MPI.Status()
            number, job, result, seconds = comm.recv(source=MPI.ANY_SOURCE, status=status)
            values = vectors[number]
            if job < 10:
                values[20*job:20*job+20] = result[:20]
                values[200+20*job:220+20*job] = result[20:]
            else:
                values[400+4*(job-10):404+4*(job-10)] = result
            path = output / f'map_{number:06d}.txt'
            temporary = path.with_suffix('.tmp')
            np.savetxt(temporary, values, header='xip[200] xim[200] Map3[80]; NaN means unfinished')
            temporary.replace(path)
            print('map', number, 'task', job, 'seconds', round(seconds, 3),
                  'complete', np.count_nonzero(np.isfinite(values)), flush=True)
            if np.all(np.isfinite(values)):
                del vectors[number]
            task = next(iterator, None)
            comm.send(task, dest=status.Get_source())
            if task is None:
                active -= 1
        (output / 'COMPLETE.txt').write_text(f'{args.count} full joint vectors complete.\n')
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
        number, job = task
        started = time.monotonic()
        if cached_number != number:
            # Global IDs 10000..14999 refer to the independent ensemble.
            group = 'comparison' if number < 10000 else 'validation'
            local_number = number if number < 10000 else number-10000
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
