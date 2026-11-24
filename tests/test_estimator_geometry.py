"""Keep radial binning, representative sides and triangle cuts distinct.

Only FFT/aperture geometry is prepared here; no physical model is run.
The mask comparison contracts a supplied complex correlation array, so
the expected answer is an explicit retained-triangle sum independent of
the transform. This catches flattened-axis and cut-placement mistakes.
"""

import numpy as np
import pytest

from cosmolike_des_3pcf_interface.production import TransformPlan
from cobaya.likelihoods.des_3pcf.survey import survey_rows


def engine(compiled, side_keep=None, rule="geometric", offset=0.):
    """Prepare three radial bins and one aperture at inexpensive angular order."""
    rows = survey_rows(nbin=1, aperture_radii=[.003], xi_edges=None)
    return TransformPlan.for_aperture(
        bins=np.ascontiguousarray(rows.bins, dtype=np.int32),
        filters=np.ascontiguousarray(rows.angles),
        theta_lower=.001 * np.exp(.3 * np.arange(3)),
        phi=(np.arange(4) + .5) * np.pi / 4,
        dlnt=.3,
        dphi=np.pi / 4,
        ell_min=20.,
        ell_max=3000.,
        minimum_size=64,
        lmax=0,
        mmax=0,
        nmu=4,
        side_keep=side_keep,
        representative_rule=rule,
        representative_offset=offset,
        transform_options={"npsi": 20},
    )


def test_side_cut_keeps_original_transform_and_contracts_only_retained_bins(compiled):
    """A noncontiguous side selection retains both side axes in their FFT bank."""
    complete = engine(compiled=compiled)
    keep = np.array([True, False, True])
    selected = engine(compiled=compiled, side_keep=keep)
    np.testing.assert_array_equal(x=complete.ell, y=selected.ell)
    random = np.random.default_rng(seed=27)
    gamma_real = random.normal(size=(4, 3, 3, 4))
    gamma_imag = random.normal(size=gamma_real.shape)
    real = complete.weights_real.reshape(1, 4, 3, 3, 4)
    imag = complete.weights_imag.reshape(1, 4, 3, 3, 4)
    expected = 0.
    for first in (0, 2):
        for second in (0, 2):
            expected += np.sum(real[0, :, first, second] * gamma_real[:, first, second])
            expected -= np.sum(imag[0, :, first, second] * gamma_imag[:, first, second])
    actual = compiled.production.aperture_apply(
        wr=selected.weights_real, wi=selected.weights_imag,
        gr=gamma_real.reshape(1, 4, -1), gi=gamma_imag.reshape(1, 4, -1),
    )[0, 0]
    np.testing.assert_allclose(actual=actual, desired=expected, rtol=2.e-14)
    keep[:] = False
    np.testing.assert_array_equal(x=selected.side_keep, y=[True, False, True])
    restored = engine(compiled=compiled)
    np.testing.assert_array_equal(x=complete.weights_real, y=restored.weights_real)


@pytest.mark.parametrize("offset", [0., -.015226])
def test_area_representative_uses_radial_area_measure(compiled, offset):
    """The analytic representative equals integral(t^2 dt)/integral(t dt)."""
    selected = engine(compiled=compiled, rule="area_weighted", offset=offset)
    lower = .001 * np.exp(.3 * np.arange(3))
    upper = lower * np.exp(.3)
    expected = (2. / 3.) * (upper**3 - lower**3) / (upper**2 - lower**2)
    expected += lower * offset
    np.testing.assert_allclose(actual=selected.representative_sides,
                              desired=expected, rtol=3.e-15)
    geometric = engine(compiled=compiled)
    assert np.max(np.abs(selected.weights_real - geometric.weights_real)) > 0.


@pytest.mark.parametrize("mask", [[1, 0, 1], [True, False], [False, False, False]])
def test_invalid_side_masks_fail_before_fft_setup(compiled, mask):
    """Integer, mismatched and empty selections cannot define triangle cuts."""
    with pytest.raises(ValueError, match="side_keep"):
        engine(compiled=compiled, side_keep=mask)


def test_archived_lower_edge_selection_and_invalid_representative(compiled):
    """The 8-arcmin cut selects the original bank; it does not restart its grid."""
    lower = np.geomspace(.5, 80., 21)[:-1]
    np.testing.assert_array_equal(x=np.flatnonzero(lower >= 8.), y=np.arange(11, 20))
    with pytest.raises(ValueError, match="outside its bin"):
        engine(compiled=compiled, rule="area_weighted", offset=-.5)
