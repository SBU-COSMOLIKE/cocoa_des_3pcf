"""Test the notebook's fixed-dark-energy calculation, with IA still varied."""

from copy import deepcopy
import numpy as np
from test_fisher_derived_quantities import amplitude_reports


def toy_forecast():
    """Interleave dark energy and nuisance coordinates to test index selection.

    The coordinates are (Omega_m, w, DES_A1_1, sigma8, wa) at Omega_m = 0.3,
    sigma8 = 0.8, so the gradient of S8 = sigma8 (Omega_m/0.3)^(1/2) the
    notebook reads from the forecast record is (4/3, 0, 0, 1, 0).
    """
    names = ["omegam", "w", "DES_A1_1", "sigma8", "wa"]
    likelihood = np.diag([1., 2., 1., 2., 1.])
    likelihood[0, 1] = likelihood[1, 0] = 1.
    likelihood[2, 3] = likelihood[3, 2] = 1.
    prior = np.diag([0., 0., 1., 0., 0.])
    return {
        "names": names,
        "fiducial": {"omegam": .3, "sigma8": .8, "w": -1., "wa": 0.},
        "s8_gradient": np.array([4/3, 0., 0., 1., 0.]),
        "errors": {},
        "information": {"fisher_two_point": likelihood+prior,
                        "fisher_joint": 2*likelihood+prior},
    }


def test_fixed_dark_energy_keeps_ia_marginalization_and_prior_once():
    """Known Gaussian variances distinguish fixing w from fixing all nuisances."""
    forecast = toy_forecast()
    original = deepcopy(forecast)
    reports = amplitude_reports(forecast)
    first, joint = reports[True, "two_point"], reports[True, "joint"]
    # Fixing w gives Var(Omega_m)=1 and 1/2. Marginalizing DES_A1_1 with its
    # single prior gives Var(sigma8)=2/3 and 3/8. grad(S8)=(4/3,1).
    np.testing.assert_allclose(first["sigma_s8"]**2, 22/9, rtol=3e-14)
    np.testing.assert_allclose(joint["sigma_s8"]**2, 91/72, rtol=3e-14)
    np.testing.assert_allclose(first["omega_m_s8_fom"], np.sqrt(3/2), rtol=3e-14)
    np.testing.assert_allclose(joint["omega_m_s8_fom"]/first["omega_m_s8_fom"],
                               4*np.sqrt(2)/3, rtol=3e-14)
    assert first["names"] == ["omegam", "DES_A1_1", "sigma8"]
    np.testing.assert_allclose(first["parameter_covariance"][1, 1], 2/3, rtol=3e-14)
    np.testing.assert_allclose(joint["parameter_covariance"][1, 1], .5, rtol=3e-14)
    # Keeping dark energy marginalized is a different experiment.
    np.testing.assert_allclose(reports[False, "two_point"]["sigma_s8"]**2, 38/9)
    for key in forecast["information"]:
        np.testing.assert_array_equal(forecast["information"][key], original["information"][key])
    assert forecast["fiducial"] == original["fiducial"]


def test_null_removed_direction_does_not_block_a_valid_fixed_forecast(capsys):
    """A null wa can be fixed; a null retained sigma8 must stay unresolved."""
    forecast = toy_forecast()
    for matrix in forecast["information"].values():
        matrix[4, 4] = 0.
    reports = amplitude_reports(forecast)
    assert (False, "joint") not in reports
    np.testing.assert_allclose(reports[True, "two_point"]["sigma_s8"]**2, 22/9)
    assert "Unconstrained Fisher direction" in capsys.readouterr().out
    for matrix in forecast["information"].values():
        matrix[3, :] = 0.
        matrix[:, 3] = 0.
    assert amplitude_reports(forecast) == {}
    assert "Unconstrained Fisher direction" in capsys.readouterr().out
