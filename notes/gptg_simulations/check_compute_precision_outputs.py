"""Check saved precision-test maps, including masked E/B cross spectra.

Run compute_precision_tests.py --metal and benchmark_compute_precision.py
first. These are numerical agreement tests, not a new physical calibration.
The mask is identical in every comparison; no mask deconvolution is needed
when measuring the difference between two numerical implementations.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import argparse
import numpy as np
import yaml
from generate_maps import PROJECT, setup, pseudo_modes


def spectra(maps, indices, geometry, ell, lmax):
    """Return E/B band powers for four bins and all ten source pairs.

    Packed harmonic arrays store m>=0. Real sky fields supply the missing
    negative m through complex conjugation, so multiply m>0 power by two.
    Each returned block is (E or B, source i, source j, multipole band).
    """
    coefficients = []
    for source in range(4):
        coefficients.append(pseudo_modes(
            shear=maps[source],
            indices=indices,
            geometry=geometry,
            lmax=lmax,
            workers=1,
        ))
    multiplicity = np.full(len(ell), 2.)
    multiplicity[:lmax+1] = 1.
    edges = np.unique(np.geomspace(2, lmax+1, 21).astype(int))
    result = np.empty((2, 4, 4, len(edges)-1))
    for component in range(2):
        for first in range(4):
            for second in range(first, 4):
                product = np.real(coefficients[first][component]
                                  * coefficients[second][component].conj())
                total = np.bincount(ell, weights=multiplicity*product,
                                    minlength=lmax+1)
                for band, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                    value = total[low:high].sum() / np.sum(2*np.arange(low, high)+1)
                    result[component, first, second, band] = value
                    result[component, second, first, band] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--precision-output', default='chains/compute_precision')
    parser.add_argument('--mpi-output', default='chains/compute_precision_mpi')
    args = parser.parse_args()
    source = PROJECT / 'chains/gptg_ensemble_15000'
    record = yaml.safe_load((source / 'calibration.yaml').read_text())
    indices, geometry, ell, factor, nodes, weights = setup(lmax=record['lmax'])
    folder = PROJECT / args.precision_output
    rows = []
    for number in range(args.count):
        reference = np.load(folder / f'cpu64_{number:03d}.npy').astype(float)
        reference_power = spectra(reference, indices, geometry, ell, record['lmax'])
        for mode in ('cpu32', 'metal32'):
            candidate = np.load(folder / f'{mode}_{number:03d}.npy').astype(float)
            power = spectra(candidate, indices, geometry, ell, record['lmax'])
            errors = []
            for component in range(2):
                for first in range(4):
                    for second in range(first, 4):
                        scale = np.sqrt(reference_power[component, first, first]
                                        * reference_power[component, second, second])
                        errors.append(np.max(np.abs(power[component, first, second]
                                                    - reference_power[component, first, second])/scale))
            rows.append([number, mode, max(errors)])
            print('spectra', number, mode, max(errors), flush=True)
    mpi = PROJECT / args.mpi_output
    journal = np.loadtxt(mpi / 'cpu64/comparison_measurements.txt')
    for number in journal[:, 0].astype(int):
        np.testing.assert_array_equal(
            np.load(mpi / f'cpu64/comparison/GPTG_DES_{number:06d}.npy'),
            np.load(source / f'comparison/GPTG_DES_{number:06d}.npy'),
        )
    with (folder / 'output_checks.txt').open('w') as report:
        report.write('# seed mode max_EE_BB_cross_band_error_normalized_by_auto_power\n')
        for row in rows:
            report.write(' '.join(map(str, row))+'\n')
        report.write(f'# {len(journal)} default maps bitwise equal to production\n')
        for mode in ('cpu32', 'metal32'):
            candidate = np.loadtxt(mpi / mode / 'comparison_measurements.txt')
            np.testing.assert_array_equal(candidate[:, :2], journal[:, :2])
            error = np.max(np.abs(candidate[:, 4:]-journal[:, 4:])
                           / np.std(journal[:, 4:], axis=0, ddof=1))
            report.write(f'# {mode} max noisy-aperture error / sample scatter: {error:.9e}\n')
            if not np.isfinite(error) or error >= 1.e-4:
                raise RuntimeError('Noisy aperture comparison failed')
    if any(not np.isfinite(row[2]) or row[2] >= 1.e-4 for row in rows):
        raise RuntimeError('Masked spectrum comparison failed')


if __name__ == '__main__':
    main()
