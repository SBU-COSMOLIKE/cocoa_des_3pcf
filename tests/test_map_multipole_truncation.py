"""Check the opening-angle multipole cutoff of the Map3 aperture plan.

Terms used in this file:
  multipole cutoff = the shear three-point function is expanded as a
      Fourier series in the opening angle between two triangle sides;
      the cutoff is the largest series order n kept.
  map_mmax = the likelihood key holding that cutoff for the Map3
      aperture transform (likelihood/joint_shear.yaml).

The published DES Y3 measurement and its theory kept multipoles up to
n = 100; the likelihood default is map_mmax = 20. This test runs
EXAMPLE_EVALUATE1 (CAMB, the bispectrum block and the joint xi + Map3
likelihood) at its fiducial point once per cutoff and bounds the
difference two ways: the largest relative change of a Map3 row, and the
covariance-weighted distance d^T C^-1 d of the full-vector change d
with the joint covariance C. The distance is the quadratic form of the
difference between the two computations, so it measures their
disagreement directly; the loaded data vector never enters it.
Every evaluation passes cached=False: Cobaya otherwise serves a
revisited point from its state cache without calling logp, and
last_datavector would still hold the previous point's vector.
"""

from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
COCOA = PROJECT.parents[1]

# Bounds on the map_mmax = 20 vs 100 difference at the fiducial. The
# measured values are about 2e-10 (relative Map3 change) and 6e-18
# (covariance-weighted distance); the bounds leave two and six orders of
# magnitude of room while staying far below any analysis-relevant size
# (the project's accuracy comfort target is a distance below 0.2).
RELATIVE_BOUND = 1.0e-8
DISTANCE_BOUND = 1.0e-12


def evaluate_with_cutoff(mmax):
    """Evaluate EXAMPLE_EVALUATE1 at its fiducial with one map_mmax.

    Arguments:
      mmax = the opening-angle multipole cutoff handed to the likelihood
          as map_mmax; the aperture plan is rebuilt for each value, so
          every cutoff needs its own model.
    Returns:
      (vector, map_indices, covariance) = the full data vector, the
      positions of its Map3 rows, and the joint covariance the
      likelihood installed (identical for every cutoff: the covariance
      is read from the released file, not computed).
    """
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file

    import cosmolike_des_3pcf_interface as ci

    info = yaml_load_file(str(PROJECT / "EXAMPLE_EVALUATE1.yaml"))
    point = dict(info["sampler"]["evaluate"]["override"])
    info.pop("sampler")
    info.pop("output")
    info["theory"]["camb"]["path"] = str(COCOA / "external_modules/code/CAMB")
    info["likelihood"]["des_3pcf.joint_shear"]["map_mmax"] = mmax

    with get_model(info_or_yaml_or_file=info) as model:
        likelihood = model.likelihood["des_3pcf.joint_shear"]
        model.logposterior(params_values=point, cached=False)
        vector = np.array(likelihood.last_datavector, copy=True)
        map_indices = np.array(likelihood.map_indices, copy=True)
        covariance = np.array(ci.get_joint_covariance(), copy=True)
    return vector, map_indices, covariance


def test_default_cutoff_matches_the_measurement_cutoff():
    """map_mmax = 20 must reproduce the n = 100 vector within the bounds."""
    default, map_indices, covariance = evaluate_with_cutoff(mmax=20)
    reference, _, _ = evaluate_with_cutoff(mmax=100)

    # Both probes are present and every row is computed.
    assert map_indices.size > 0 and map_indices.size < default.size
    assert np.all(np.isfinite(default)) and np.all(np.isfinite(reference))
    assert np.any(reference[map_indices] != 0.0)

    # The largest relative change of a Map3 row; the xi rows do not
    # depend on the aperture cutoff, so only the Map3 rows are compared
    # row by row.
    change = default - reference
    relative = np.max(np.abs(change[map_indices] / reference[map_indices]))

    # d^T C^-1 d of the full-vector change: one scalar that weights
    # every row by the measurement errors and their correlations.
    distance = change @ np.linalg.solve(covariance, change)

    print(f"map_mmax 20 vs 100: max |dMap3/Map3| = {relative:.3e}, "
          f"d^T C^-1 d = {distance:.3e}")
    assert relative < RELATIVE_BOUND
    assert distance < DISTANCE_BOUND
