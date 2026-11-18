"""Analytic checks of fixed dark energy with all remaining nuisances varied."""

import copy

import numpy as np
import pytest

from cosmolike_des_3pcf_fisher_wrappers import fixed_dark_energy_summary


def toy_saved_fisher():
    """Define a Gaussian quadratic with coupled cosmology and IA variables.

    In the (Omega_m,w0) subspace, the likelihood precision is [[1,1],[1,2]].
    In the (sigma8,a1) subspace it is [[2,1],[1,1]], with one independent
    Gaussian prior adding precision1 to a1. The wa direction is independent.
    Joint data multiply the likelihood precision by2; the prior stays1.
    Parameters are interleaved so fixing dark energy cannot mean truncating
    the end of an array. All coordinates are displacements about a fiducial.
    """
    names = np.array(object=[
        "omega_m",
        "w0",
        "a1",
        "sigma8",
        "wa",
    ])
    likelihood = np.diag(v=[
        1.0,
        2.0,
        1.0,
        2.0,
        1.0,
    ])
    likelihood[0, 1] = 1.0
    likelihood[1, 0] = 1.0
    likelihood[2, 3] = 1.0
    likelihood[3, 2] = 1.0
    prior = np.zeros(shape=(5, 5))
    prior[2, 2] = 1.0
    result = {
        "parameter_names": names,
        "fisher_prior": prior,
        "fisher_two_point_likelihood": likelihood,
        "fisher_joint_likelihood": 2.0 * likelihood,
        "fisher_two_point": likelihood + prior,
        "fisher_joint": 2.0 * likelihood + prior,
    }
    metadata = {
        "configuration": {
            "parameters": {
                "omega_m": 0.3,
                "sigma8": 0.8,
                "w0": -1.0,
                "wa": 0.0,
            },
            "fisher": {
                "priors": {"a1": 1.0},
            },
        },
    }
    return result, metadata


def test_fixing_dark_energy_keeps_ia_marginalization_and_prior_once():
    """Conditional cosmology errors differ from marginalizing dark energy."""
    result, metadata = toy_saved_fisher()
    saved_arrays = copy.deepcopy(result)
    saved_metadata = copy.deepcopy(metadata)
    summary = fixed_dark_energy_summary(result=result, metadata=metadata)

    # Fixing w0 changes Var(Omega_m) from2 to1 (xi) and1 to1/2 (joint).
    # Marginalizing a1 gives Var(sigma8)=2/3 and3/8. Fixing a1 instead
    # would give1/2 and1/4. Adding its prior twice gives1/3, not3/8, joint.
    # At Omega_m=.3 and sigma8=.8, grad(S8)=(4/3,1), with zero cross term
    # in this toy. Hence conditional Var(S8)=22/9 and91/72. Marginalizing
    # dark energy would instead give38/9 and155/72.
    np.testing.assert_allclose(
        actual=summary["two_point"]["sigma_s8"] ** 2,
        desired=22.0 / 9.0,
        rtol=3.0e-14,
    )
    np.testing.assert_allclose(
        actual=summary["joint"]["sigma_s8"] ** 2,
        desired=91.0 / 72.0,
        rtol=3.0e-14,
    )
    np.testing.assert_allclose(
        actual=summary["two_point"]["omega_m_s8_fom"],
        desired=np.sqrt(3.0 / 2.0),
        rtol=3.0e-14,
    )
    np.testing.assert_allclose(
        actual=summary["joint_over_two_point_fom"],
        desired=4.0 * np.sqrt(2.0) / 3.0,
        rtol=3.0e-14,
    )
    np.testing.assert_allclose(
        actual=summary["two_point"]["sigma_ia"]["a1"] ** 2,
        desired=2.0 / 3.0,
        rtol=3.0e-14,
    )
    np.testing.assert_allclose(
        actual=summary["joint"]["sigma_ia"]["a1"] ** 2,
        desired=0.5,
        rtol=3.0e-14,
    )
    assert summary["marginalized_parameters"] == [
        "omega_m",
        "a1",
        "sigma8",
    ]
    for key in result:
        np.testing.assert_array_equal(
            x=result[key],
            y=saved_arrays[key],
        )
    assert metadata == saved_metadata
    assert "parameter_covariance_joint" not in summary
    assert "configuration" not in summary


def test_unresolved_removed_direction_does_not_require_a_full_inverse():
    """A null wa can be fixed; a null retained parameter cannot be hidden."""
    result, metadata = toy_saved_fisher()
    for key in (
        "fisher_two_point_likelihood",
        "fisher_joint_likelihood",
        "fisher_two_point",
        "fisher_joint",
    ):
        result[key][4, 4] = 0.0
    summary = fixed_dark_energy_summary(result=result, metadata=metadata)
    np.testing.assert_allclose(
        actual=summary["two_point"]["sigma_s8"] ** 2,
        desired=22.0 / 9.0,
        rtol=3.0e-14,
    )
    # Removing the sigma8 response is not an allowed rank repair: that
    # parameter remains part of the requested marginalized experiment.
    for key in (
        "fisher_two_point_likelihood",
        "fisher_joint_likelihood",
        "fisher_two_point",
        "fisher_joint",
    ):
        result[key][3, :] = 0.0
        result[key][:, 3] = 0.0
    with pytest.raises(ValueError, match="Unconstrained Fisher direction"):
        fixed_dark_energy_summary(result=result, metadata=metadata)


def test_fixing_at_a_different_fiducial_requires_new_derivatives():
    """Fixing a parameter does not move the Fisher matrix to a new cosmology."""
    result, metadata = toy_saved_fisher()
    metadata["configuration"]["parameters"]["w0"] = -0.9
    with pytest.raises(ValueError, match="saved fiducial must have w0"):
        fixed_dark_energy_summary(result=result, metadata=metadata)


def test_inconsistent_posterior_and_saved_prior_are_refused():
    """A mislabeled prior-added matrix must not become a new forecast."""
    result, metadata = toy_saved_fisher()
    result["fisher_joint"] += result["fisher_prior"]
    with pytest.raises(ValueError, match="posterior does not equal likelihood"):
        fixed_dark_energy_summary(result=result, metadata=metadata)
