"""Measure four DES xi-plus bins on saved GPTG maps without new skies.

This bounded joint test uses source bin four, angular bins 12..15 of the
20 log bins from 2.5 to 250 arcmin, plus the eight saved cubic moments.
It is not the published 480-row statistic. Pixel weights are uniform.
Run with TreeCorr and astropy-healpix installed in Cocoa's environment.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
from pathlib import Path
import argparse
import time
import numpy as np
from astropy_healpix import HEALPix
import treecorr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=1500)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[2] / 'chains/gptg_ensemble_15000'
    output = folder / 'joint_xip_measurements.txt'
    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            if line and not line.startswith('#'):
                completed.add(int(line.split()[0]))
    indices = np.load(folder / 'indices_512.npy')
    ra, dec = HEALPix(nside=512, order='ring').healpix_to_lonlat(indices)
    edges = np.geomspace(2.5, 250., 21)[12:17]
    with output.open('a', buffering=1) as stream:
        if not completed:
            stream.write('# map_index seconds xip_bin12 xip_bin13 xip_bin14 xip_bin15\n')
        for index in range(args.count):
            if index in completed:
                continue
            path = folder / 'comparison' / f'GPTG_DES_{index:06d}.npy'
            while not path.exists():
                time.sleep(30)
            start = time.monotonic()
            shear = np.load(path, mmap_mode='r')
            # Our generator stores east/north components; TreeCorr uses west/
            # north. Reflecting that basis reverses the second shear component.
            catalog = treecorr.Catalog(
                ra=ra.degree,
                dec=dec.degree,
                ra_units='deg',
                dec_units='deg',
                g1=np.asarray(shear[3, 0], dtype=np.float64),
                g2=np.asarray(shear[3, 1], dtype=np.float64),
                flip_g2=True,
            )
            correlation = treecorr.GGCorrelation(
                min_sep=edges[0],
                max_sep=edges[-1],
                nbins=4,
                sep_units='arcmin',
                bin_slop=0.,
                angle_slop=0.,
                metric='Arc',
            )
            correlation.process(cat1=catalog, num_threads=4)
            np.testing.assert_allclose(correlation.left_edges, edges[:-1])
            elapsed = time.monotonic()-start
            stream.write(f'{index} {elapsed:.6f} ' + ' '.join(f'{v:.17e}' for v in correlation.xip)+'\n')
            print(index, 'seconds', elapsed, flush=True)


if __name__ == '__main__':
    main()
