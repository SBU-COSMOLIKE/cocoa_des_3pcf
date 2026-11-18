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
