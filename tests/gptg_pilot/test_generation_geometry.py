"""Check refined RING geometry and empty validation groups on small arrays.

No map generation, harmonic solve or TreeCorr measurement runs here. The
parent/child relation is exact in HEALPix NESTED indexing; map files remain
in RING order. A temporary ensemble also exercises the public header audit.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import ducc0
import numpy as np
import yaml

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project / 'scripts'))
from generate_maps import setup
from verify_ensemble import verify_ensemble

# Imported production support must not overwrite the explicit one-thread
# allocation set above. This also covers the nested aperture-module import.
assert os.environ["OMP_NUM_THREADS"] == "1"

parents = np.array([0, 1, 20, 101], dtype=np.int64)
children, geometry, ell, factor, nodes, weights = setup(
    lmax=3,
    nside=16,
    indices=parents,
    index_nside=4,
)
coarse = ducc0.healpix.Healpix_Base(4, 'RING')
fine = ducc0.healpix.Healpix_Base(16, 'RING')
recovered = coarse.nest2ring(fine.ring2nest(children)//16)
np.testing.assert_array_equal(np.sort(recovered), np.repeat(parents, 16))
assert len(np.unique(children)) == 16*len(parents)
assert np.all(np.diff(children) > 0)
assert np.sum(geometry['nphi']) == 12*16**2
unchanged = setup(lmax=3, nside=4, indices=parents, index_nside=4)[0]
np.testing.assert_array_equal(unchanged, parents)
for invalid in (0, 2, 6):
    try:
        setup(lmax=3, nside=invalid, indices=parents, index_nside=4)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid resolution accepted')

# A finer pixel has smaller solid angle. Preserve angular noise power by
# multiplying the component standard deviation by the NSIDE ratio.
sigma = np.array([0.035, 0.04])
np.testing.assert_allclose((4*sigma)**2/(12*2048**2), sigma**2/(12*512**2))

with TemporaryDirectory() as temporary:
    folder = Path(temporary)
    metadata = dict(indices_file='indices_16.npy', comparison_count=1,
                    validation_count=0, comparison_seed_start=341104759,
                    validation_seed_start=441104759)
    (folder / 'ensemble.yaml').write_text(yaml.safe_dump(metadata))
    np.save(folder / 'indices_16.npy', children)
    for group in ('comparison', 'validation'):
        (folder / group).mkdir()
        (folder / f'{group}_measurements.txt').write_text('# journal\n')
    np.save(folder / 'comparison/GPTG_DES_000000.npy',
            np.ones((4, 2, len(children)), dtype=np.float32))
    row = np.zeros((1, 20))
    row[0, 1] = metadata['comparison_seed_start']
    np.savetxt(folder / 'comparison_measurements.txt', row)
    assert verify_ensemble(folder) == 1
    # A nominally unused group must not conceal old journal entries.
    np.savetxt(folder / 'validation_measurements.txt', row)
    try:
        verify_ensemble(folder)
    except ValueError as error:
        assert 'unexpected rows' in str(error)
    else:
        raise AssertionError('Nonempty unused validation journal accepted')
print('PASS: RING refinement, unchanged geometry, noise area and journal audit')
