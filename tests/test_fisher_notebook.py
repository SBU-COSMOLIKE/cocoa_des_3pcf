"""Bounded tests of numbered inputs and the native IA response they exercise."""

from pathlib import Path
import numpy as np
import pytest
from getdist import IniFile
from cobaya.yaml import yaml_load_file

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("number", [2, 3])
def test_numbered_covariance_keeps_all_rows_and_the_cross_block(number):
    """Read the numbered YAML and its ASCII dataset, without initializing theory.

    Cases 2 and 3 (the synthetic NLA and TATT vectors) keep every row and
    the full joint covariance, including its xi-Map3 cross block.
    """
    from cobaya.likelihoods.des_3pcf.survey import (
        ARCMIN_TO_RADIAN, read_covariance, survey_rows)

    # Step 1: the dataset named by the example yaml.
    info = yaml_load_file(str(PROJECT / f"EXAMPLE_EVALUATE{number}.yaml"))
    dataset = info["likelihood"]["des_3pcf.joint_shear"]["data_file"]
    inputs = IniFile(str(PROJECT / "data" / dataset))
    assert inputs.string("mask_file") == ""
    assert inputs.string("compression_file") == ""

    # Step 2: the rows its geometry keys define: 200 xi+, 200 xi-, 80 Map3.
    radii = [float(value) for value in inputs.string("map3_radii_arcmin").split()]
    rows = survey_rows(
        nbin=inputs.int("source_ntomo"),
        aperture_radii=np.asarray(radii) * ARCMIN_TO_RADIAN,
        xi_edges=np.geomspace(inputs.float("theta_min_arcmin"),
                              inputs.float("theta_max_arcmin"),
                              inputs.int("n_theta") + 1) * ARCMIN_TO_RADIAN,
    )
    assert [int(np.sum(rows.kind == kind)) for kind in ("xip", "xim", "map3")] == [200, 200, 80]

    # Step 3: the ASCII covariance covers all 480 rows, cross block included.
    covariance = read_covariance(path=inputs.relativeFileName("cov_file"), size=480)
    assert np.any(covariance[:400, 400:] != 0.)
    # These Gaussian widths are read by the numbered likelihood; their
    # truncation bounds are not extra Gaussian Fisher information.
    assert info["params"]["DES_DZ_S3"]["prior"]["scale"] == .011


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
