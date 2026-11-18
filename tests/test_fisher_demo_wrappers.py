"""Check derived forecast quantities without running the physical mean."""

import numpy as np

import cosmolike_des_3pcf_fisher_wrappers as notebook


def test_s8_error_keeps_cross_covariance_and_fom_ratio():
    """A negative Omega_m/sigma8 correlation reduces the S8 variance."""
    covariance = np.array([
        [0.0009, -0.0024],
        [-0.0024, 0.01],
    ])
    result = {
        "parameter_names": np.array(["omega_m", "sigma8"]),
        "parameter_covariance_two_point": covariance,
        "parameter_covariance_joint": 0.25 * covariance,
    }
    metadata = {
        "configuration": {
            "parameters": {
                "omega_m": 0.3,
                "sigma8": 0.8,
            },
        },
    }
    summary = notebook.derived_amplitude_summary(
        result=result, metadata=metadata,
    )
    # Variance is .0016+.01-.0064=.0052. Omitting the covariance
    # would produce .0116 and fail this independent numerical identity.
    np.testing.assert_allclose(
        summary["two_point"]["sigma_s8"], np.sqrt(0.0052), rtol=2e-14,
    )
    np.testing.assert_allclose(
        summary["joint_over_two_point_fom"], 4.0, rtol=2e-14,
    )


def test_derivative_summary_resolves_small_amplitude_aperture_rows():
    """A tiny raw row can carry order-one information in its own error units."""
    metadata = {
        "comparison_rank_status": {"joint": "resolved"},
        "configuration": {"fisher": {"priors": {}}},
    }
    for aperture_unit in (1.0e-8, 1.0):
        result = {
            "parameter_names": np.array(["a1"]),
            "fisher_joint_likelihood": np.array([[2.0]]),
            "fisher_two_point_likelihood": np.array([[1.5]]),
            "fisher_conditional_three_point": np.array([[0.5]]),
            "fiducial_covariance": np.diag([1.0, aperture_unit**2]),
            "derivatives_h": np.array([[1.0], [aperture_unit]]),
            "derivatives_h2": np.array([[1.0], [2.0 * aperture_unit]]),
            "derivative_relative_change": np.array([aperture_unit]),
        }
        summary = notebook.fisher_summary(result=result, metadata=metadata)
        # In covariance units the columns are (1,1) and (1,2), so the
        # difference has norm1 and the reference has normsqrt(5).
        np.testing.assert_allclose(
            actual=summary["derivative_covariance_relative_change"],
            desired=[1.0 / np.sqrt(5.0)], rtol=2.0e-14,
        )
