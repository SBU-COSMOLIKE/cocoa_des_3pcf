"""Check every expected map and journal row without rescanning all pixels.

The audit checks exact IDs, seeds, finite recorded moments, NumPy headers,
float32 dtype, four-bin/two-component shape and complete file lengths.
It does not claim a fresh all-pixel finiteness scan or a physical validation.
"""
import argparse
from pathlib import Path
import numpy as np
import yaml


def verify_ensemble(folder):
    indices = np.load(folder / 'indices_512.npy')
    shape = (4, 2, len(indices))
    metadata = yaml.safe_load((folder / 'ensemble.yaml').read_text())
    comparison_count = int(metadata['comparison_count'])
    validation_count = int(metadata['validation_count'])
    checked = 0
    for group, count, seed_start in (
        ('comparison', comparison_count, int(metadata['comparison_seed_start'])),
        ('validation', validation_count, int(metadata['validation_seed_start'])),
    ):
        journal = np.loadtxt(folder / f'{group}_measurements.txt', ndmin=2)
        if journal.shape != (count, 20):
            raise ValueError(f'{group}: incomplete journal or unexpected columns')
        np.testing.assert_array_equal(journal[:, 0], np.arange(count))
        np.testing.assert_array_equal(journal[:, 1], seed_start+np.arange(count))
        if not np.isfinite(journal).all():
            raise ValueError(f'{group}: nonfinite journal entry')
        names = {f'GPTG_DES_{number:06d}.npy' for number in range(count)}
        actual = {path.name for path in (folder / group).glob('GPTG_DES_*.npy')}
        if actual != names:
            raise ValueError(f'{group}: missing or unexpected map IDs')
        for number in range(count):
            path = folder / group / f'GPTG_DES_{number:06d}.npy'
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            if values.shape != shape or values.dtype != np.dtype('float32'):
                raise ValueError(f'Wrong dtype or shape: {path}')
            if path.stat().st_size != values.offset+values.nbytes:
                raise ValueError(f'Incomplete or trailing map bytes: {path}')
            del values
            checked += 1
    (folder / 'COMPLETENESS.txt').write_text(
        f'{checked} maps verified: {comparison_count} comparison + {validation_count} validation.\n'
        'Exact IDs/seeds, finite journal moments, shape/dtype and file sizes pass.\n'
        'Headers checked without a full-pixel rescan. TreeCorr not run.\n'
    )
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    arguments = parser.parse_args()
    print('Verified maps:', verify_ensemble(arguments.folder))
