"""Check the 480-dimensional ordering and a known finite-polynomial limit.

Zero input vectors are an algebraic test, not simulated observations. With
A held fixed, their masked factors have residual diag(0_400,I_80). Thus the
order-r constrained covariance must be diag(A,D_reference/(r+1)).
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import numpy as np
from astropy.io import fits
from fixed_block_precision import estimates
p = Path(__file__).resolve().parents[2]
production = np.load(p / 'chains/gptg_ensemble_15000/des_y3_full_covariance.npz')
rows = np.flatnonzero(production['rows'][:, 0] < 2)
indices = (rows[:, None]*20+np.arange(20)).ravel()
assert len(indices) == 400
reference = np.zeros((480, 480))
reference[:400, :400] = production['total'][np.ix_(indices, indices)]
archived = fits.getdata(p / 'data/upstream_y3/data/covariance/xip_xim_map3_covariance_8ARCMINCUT_25Feb25.fits', extname='COVMAT')
reference[400:, 400:] = archived[400:, 400:]
result, used = estimates(np.zeros((12, 480)), reference, 400)
for order, raw, precision, covariance in result:
    assert covariance is not None
    np.testing.assert_array_equal(covariance[:400, :400], reference[:400, :400])
    np.testing.assert_allclose(covariance[400:, 400:], reference[400:, 400:]/(order+1), rtol=1.e-10, atol=1.e-30)
    np.testing.assert_array_equal(covariance[:400, 400:], np.zeros((400, 80)))
    print('480 dimensions, order', order, ': exact A, zero cross block, D/(r+1) passed.')
