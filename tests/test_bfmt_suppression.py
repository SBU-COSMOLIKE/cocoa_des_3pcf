"""Check the power-only bfmt feedback option of the joint likelihood.

Terms used in this file:
  bfmt = Cocoa's baryonic-feedback Cobaya theory block; it returns
      R_P = P_feedback/P_DMO, the ratio of the matter power spectrum with
      baryonic feedback to the gravity-only (DMO, dark-matter-only) one,
      at fixed wavenumbers in 1/Mpc.
  external_baryon_suppression = the likelihood yaml key that requests R_P
      from bfmt and applies it to the nonlinear power of the xi rows.

The likelihood tables use k in h/Mpc, so R_P must be read at h * k. The
first test checks that read against numpy's interpolation on synthetic
responses, without Cobaya. The refusal tests build the real likelihood
class. The last test runs CAMB and bfmt and compares the likelihood's read
with bfmt evaluated directly at h * k.
"""

import copy
from pathlib import Path
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
COCOA = PROJECT.parents[1]


class FixedResult:
    """Stand in for Cobaya's provider: get_result returns one fixed result."""

    def __init__(self, rows):
        """Keep the bfmt-format dictionary returned by every get_result call."""
        self.rows = rows

    def get_result(self, name):
        """Return the stored dictionary whatever product name is asked."""
        return self.rows


def zero_loglike(_self):
    """Return log L = 0: a Cobaya likelihood that only requests products."""
    return 0.0


def synthetic_likelihood(h, rows):
    """Build a likelihood object holding only what the R_P read uses.

    The object is created without Cobaya's initialization (object.__new__
    reserves the instance and runs no __init__), then receives the table
    grids of joint_shear.yaml, the value of h and a provider stand-in whose
    get_result returns the supplied bfmt dictionary.

    Arguments:
      h = dimensionless Hubble parameter of the point.
      rows = {z: R_P array over the request grid}, the bfmt result format.
    Returns:
      a _cosmolike_prototype_base instance ready for baryon_suppression_table.
    """
    from cobaya.likelihoods.des_3pcf._cosmolike_prototype_base import (
        _cosmolike_prototype_base,
    )

    likelihood = object.__new__(_cosmolike_prototype_base)
    likelihood.z_interp_2D = np.linspace(start=0.0, stop=3.1, num=5)
    likelihood.log10k_interp_2D = np.linspace(start=-5.0, stop=3.0, num=256)
    likelihood.h = h
    likelihood.provider = FixedResult(rows=rows)
    return likelihood


def synthetic_rows(z, log10k):
    """Return a smooth suppression with a feature near k = 1/Mpc per redshift.

    Arguments:
      z = redshift nodes; log10k = log10 of the request wavenumbers [1/Mpc].
    Returns:
      {z: R_P array}, positive, equal to one on large scales.
    """
    rows = {}
    for redshift in z:
        depth = 0.2 / (1.0 + redshift)
        rows[redshift] = 1.0 - depth / (1.0 + np.exp(-4.0 * log10k))
    return rows


@pytest.mark.parametrize("h", [0.55, 0.722, 0.91, 1.0])
def test_shifted_read_matches_numpy_interpolation(h):
    """The index-arithmetic read equals np.interp of ln R_P at log10(h k)."""
    z = np.linspace(start=0.0, stop=3.1, num=5)
    log10k = np.linspace(start=-5.0, stop=3.0, num=256)
    rows = synthetic_rows(z=z, log10k=log10k)
    likelihood = synthetic_likelihood(h=h, rows=rows)
    actual = likelihood.baryon_suppression_table()

    # np.interp holds the first value below the grid, which is the
    # documented behaviour of the read on the largest scales.
    target = log10k + np.log10(h)
    for index in range(z.size):
        expected = np.exp(np.interp(target, log10k, np.log(rows[z[index]])))
        np.testing.assert_allclose(actual[index], expected, rtol=1.0e-13)


def test_read_refuses_bad_bfmt_results():
    """Missing redshifts, wrong lengths, non-positive ratios and h > 1 stop."""
    z = np.linspace(start=0.0, stop=3.1, num=5)
    log10k = np.linspace(start=-5.0, stop=3.0, num=256)

    rows = synthetic_rows(z=z, log10k=log10k)
    rows.pop(z[2])
    with pytest.raises(ValueError, match="lacks z"):
        synthetic_likelihood(h=0.7, rows=rows).baryon_suppression_table()

    rows = synthetic_rows(z=z, log10k=log10k)
    rows[z[1]] = rows[z[1]][:-1]
    with pytest.raises(ValueError, match="must have 256 values"):
        synthetic_likelihood(h=0.7, rows=rows).baryon_suppression_table()

    rows = synthetic_rows(z=z, log10k=log10k)
    rows[z[3]] = rows[z[3]].copy()
    rows[z[3]][10] = 0.0
    with pytest.raises(ValueError, match="positive and finite"):
        synthetic_likelihood(h=0.7, rows=rows).baryon_suppression_table()

    rows = synthetic_rows(z=z, log10k=log10k)
    with pytest.raises(ValueError, match="h <= 1"):
        synthetic_likelihood(h=1.05, rows=rows).baryon_suppression_table()


def test_map3_rows_and_combined_feedback_are_refused():
    """A power-only model cannot score Map3 rows or stack on the BACCO pair."""
    from cobaya.likelihoods.des_3pcf.joint_shear import joint_shear

    with pytest.raises(ValueError, match="80 Map3 rows"):
        joint_shear(info={
            "data_file": "des_y3_joint2_sim.dataset",
            "external_baryon_suppression": True,
        })
    with pytest.raises(ValueError, match="both True"):
        joint_shear(info={
            "data_file": "des_y3_xi2_sim.dataset",
            "external_baryon_suppression": True,
            "external_baryon_response": True,
        })

    # The xi-only dataset keeps the 400 xi rows and cuts all 80 Map3 rows.
    likelihood = joint_shear(info={
        "data_file": "des_y3_xi2_sim.dataset",
        "external_baryon_suppression": True,
    })
    assert likelihood.map_indices.size == 80


def test_cobaya_read_agrees_with_direct_bfmt_request():
    """The likelihood's R_P equals bfmt asked directly at h * k, to bfmt accuracy.

    Two Cobaya models share CAMB and the FLAMINGO bfmt settings: the
    example likelihood with external_baryon_suppression, and a minimal
    model whose only request is R_P at exactly h * k. bfmt places its own
    coarse internal grid over each requested range and splines from it, so
    the two answers differ at its interpolation level (about 1e-3), not at
    the level of the likelihood's read (tested above to 1e-13).
    """
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file

    # ----------------------------------------------------------------------
    # Example model on the xi-only dataset with the power-only key on.
    # ----------------------------------------------------------------------
    info = yaml_load_file(str(PROJECT / "EXAMPLE_EVALUATE2.yaml"))
    point = dict(info["sampler"]["evaluate"]["override"])
    info.pop("sampler")
    info.pop("output")
    info["theory"]["camb"]["path"] = str(COCOA / "external_modules/code/CAMB")
    likelihood_info = info["likelihood"]["des_3pcf.joint_shear"]
    likelihood_info["data_file"] = "des_y3_xi2_sim.dataset"
    likelihood_info["external_baryon_suppression"] = True
    info["theory"]["bfmt"] = {
        "baryon_model": 3,
        "nz": 20,
        "nk": 100,
        "above_zmax": "unity",
    }
    feedback = {
        "fgas_sigma_flamingo": 0.0,
        "mstar_sigma_flamingo": 0.0,
        "jet_frac_flamingo": 0.0,
    }
    info["params"].update(feedback)
    with get_model(info_or_yaml_or_file=info) as model:
        model.logposterior(params_values=point, cached=False)
        likelihood = model.likelihood["des_3pcf.joint_shear"]
        actual = likelihood.baryon_suppression_table()
        datavector = np.array(likelihood.last_datavector, copy=True)
        h = likelihood.h
        z = likelihood.z_interp_2D.copy()
        k = 10.0**likelihood.log10k_interp_2D
        map_indices = likelihood.map_indices.copy()

    # Map3 rows are refused (NaN), every xi row is computed.
    assert np.all(np.isnan(datavector[map_indices]))
    assert np.all(np.isfinite(np.delete(datavector, map_indices)))

    # ----------------------------------------------------------------------
    # The same CAMB + bfmt asked for R_P at exactly h * k. The likelihood
    # and the bispectrum block are dropped, with the DES_ nuisance
    # parameters only they read; the external likelihood asks CAMB for
    # sigma8(z = 0) so the model keeps one CAMB requirement of its own.
    # ----------------------------------------------------------------------
    direct = copy.deepcopy(info)
    direct["likelihood"] = {"power": {
        "external": zero_loglike,
        "requires": {"sigma8_z": {"z": [0.0]}},
    }}
    direct["theory"].pop("bihalofit")
    cosmology = {}
    for name, value in point.items():
        if name in ("omegam", "omegab", "H0", "ns", "As_1e9"):
            cosmology[name] = value
    for name in list(direct["params"]):
        if name.startswith("DES_"):
            direct["params"].pop(name)
    with get_model(info_or_yaml_or_file=direct) as model:
        model.add_requirements(requirements={"baryon_suppression": {
            "z": z,
            "k": h * k,
        }})
        model.logposterior(params_values=cosmology)
        rows = model.provider.get_result("baryon_suppression")
        expected = np.stack([rows[value] for value in z])

    # Compare where the xi rows draw their small-scale signal.
    selected = (k > 0.05) & (k < 8.0)
    difference = np.max(np.abs(actual[:, selected] / expected[:, selected] - 1.0))
    assert difference < 1.0e-3
    assert np.min(actual[:, selected]) < 0.95
