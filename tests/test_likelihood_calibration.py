"""Check the multiplicative shear calibration of the Cobaya likelihood.

Terms used in this file:
  m_i = multiplicative shear calibration of source bin i; the measured
      shear is (1 + m_i) times the true shear.
  row factor = the product of (1 + m_i) over the shear legs of a row: two
      legs for a xi row, three for a Map3 row.

The likelihood applies the row factors after the projection, so changing
only m_1..m_4 must rescale every row by the ratio of its new and old
factors, to rounding. Returning to the first point must give the first
data vector bit for bit: no state of the displaced point may survive.
Every evaluation passes cached=False: Cobaya otherwise serves a revisited
point from its state cache without calling logp, and last_datavector
would still hold the previous point's vector.
The test runs EXAMPLE_EVALUATE2 (CAMB, the bispectrum block and the joint
xi + Map3 likelihood) at its fiducial point.
"""

from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
COCOA = PROJECT.parents[1]


def row_factors(bins, point):
    """Return the product of (1 + m_i) over the shear legs of each row.

    Arguments:
      bins = [nrow, 3] zero-based source bins; xi rows hold -1 in the
          third column.
      point = parameter dictionary with DES_M1..DES_M4.
    Returns:
      one factor per row.
    """
    factors = np.ones(bins.shape[0])
    for row in range(bins.shape[0]):
        for source in bins[row]:
            if source >= 0:
                factors[row] *= 1.0 + point[f"DES_M{source + 1}"]
    return factors


def test_calibration_rescales_each_row_and_restores():
    """Only m_i changes: rows scale by their factor ratio; A/B/A restores."""
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file

    # ----------------------------------------------------------------------
    # EXAMPLE_EVALUATE2 at its fiducial point, CAMB taken from Cocoa.
    # ----------------------------------------------------------------------
    info = yaml_load_file(str(PROJECT / "EXAMPLE_EVALUATE2.yaml"))
    first = dict(info["sampler"]["evaluate"]["override"])
    info.pop("sampler")
    info.pop("output")
    info["theory"]["camb"]["path"] = str(COCOA / "external_modules/code/CAMB")

    # ----------------------------------------------------------------------
    # The displaced point changes only the four calibrations.
    # ----------------------------------------------------------------------
    second = dict(first)
    second.update(DES_M1=0.04, DES_M2=-0.02, DES_M3=0.01, DES_M4=-0.03)

    with get_model(info_or_yaml_or_file=info) as model:
        likelihood = model.likelihood["des_3pcf.joint_shear"]
        bins = np.array(likelihood.rows.bins, copy=True)

        model.logposterior(params_values=first, cached=False)
        baseline = np.array(likelihood.last_datavector, copy=True)
        model.logposterior(params_values=second, cached=False)
        displaced = np.array(likelihood.last_datavector, copy=True)
        model.logposterior(params_values=first, cached=False)
        restored = np.array(likelihood.last_datavector, copy=True)

    # ----------------------------------------------------------------------
    # Both probes are present and every row is computed.
    # ----------------------------------------------------------------------
    assert np.any(bins[:, 2] < 0) and np.any(bins[:, 2] >= 0)
    assert np.all(np.isfinite(baseline)) and np.any(baseline != 0.0)

    ratio = row_factors(bins=bins, point=second) / row_factors(bins=bins, point=first)
    np.testing.assert_allclose(displaced, baseline * ratio, rtol=1.0e-14, atol=0.0)
    np.testing.assert_array_equal(restored.view(np.uint64), baseline.view(np.uint64))
