"""Bounded tests of numbered inputs and the native IA response they exercise."""

from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits
from getdist import IniFile
from cobaya.yaml import yaml_load_file

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("number", [1, 2])
def test_numbered_covariance_keeps_all_rows_and_the_cross_block(number):
    """Read the actual numbered YAML and dataset, without initializing theory."""
    info = yaml_load_file(str(PROJECT / f"EXAMPLE_EVALUATE{number}.yaml"))
    dataset = info["likelihood"]["des_3pcf.joint_shear"]["data_file"]
    inputs = IniFile(str(PROJECT / "data" / dataset))
    covariance = fits.getdata(inputs.relativeFileName("cov_file"))
    assert covariance.shape == (480, 480)
    assert np.any(covariance[:400, 400:] != 0.)
    assert inputs.string("mask_file") == ""
    assert inputs.string("compression_file") == ""
    two = inputs.relativeFileName("two_point_file")
    three = inputs.relativeFileName("aperture_file")
    assert len(fits.getdata(two, extname="xip")) == 200
    assert len(fits.getdata(two, extname="xim")) == 200
    assert len(fits.getdata(three, extname="map3")) == 80
    # These Gaussian widths are read by the numbered likelihood; their
    # truncation bounds are not extra Gaussian Fisher information.
    assert info["params"]["dz3"]["prior"]["scale"] == .011


def test_native_ia_signs_power_choices_and_restoration(compiled):
    """Check C projections on tiny supplied tables, without a forecast driver.

    The existing analytic bispectrum fixture avoids a BiHalofit computation.
    Its power table has non-D-squared redshift evolution, so changing the
    quadratic-IA power prescription produces a detectable physical change.
    """
    from test_supplied_baseline import baseline_input, supplied_inputs
    from theory.bihalofit.variance import reference_physics_model

    model = reference_physics_model(
        factory=compiled.production.PhysicsModelWithBispectrum,
        **supplied_inputs(),
        baseline=baseline_input(),
    )
    ell = np.geomspace(30., 3000., 12)
    sides = np.ascontiguousarray(np.repeat(ell[:, None], 3, axis=1))
    triplets = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
    results = []
    for mode in (2, 0, 2):
        signals = np.empty((3, 2, 12))
        for case, (a1, a2, bta) in enumerate(((0., 0., 0.), (-.2, .4, .91),
                                             (.2, -.4, .91))):
            for index, bins in enumerate(triplets):
                signals[case, index] = model.projected(
                    sides=sides,
                    bins=bins,
                    parameters={"a1": a1, "a2": a2, "bta": bta,
                                "ct": 0., "power_mode": mode},
                )
        results.append(signals)
    nonlinear, linear, restored = results
    assert np.all(np.isfinite(nonlinear))
    assert not np.array_equal(nonlinear[1], nonlinear[2])
    assert not np.array_equal(nonlinear[1:], linear[1:])
    np.testing.assert_array_equal(nonlinear[0], linear[0])
    np.testing.assert_array_equal(nonlinear, restored)
