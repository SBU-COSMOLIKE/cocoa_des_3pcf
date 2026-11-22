"""Check finite model distances with independent Gaussian innovations.

No CAMB or native mean calculation is required. The latent displacements
provide known distances before the observational covariance is constructed.
"""

import numpy as np
import pytest

from cosmolike_notebook_utils import fisher as cnu
from test_fisher_cobaya_notebooks import notebook_cells


def finite_sign_pair_separation(first, second, covariance):
    """Run the visible finite-contrast cell on two known predictions.

    Row identity is supplied once by the same notebook model. These tests
    vary row ordering and units coherently across means and covariance.
    """
    source = next(cell for cell in notebook_cells("notes/intrinsic_alignments/EXAMPLE_FISHER_IA.ipynb")
                  if 'names=["sign_contrast"]' in cell)
    means = {}
    for model in ("nla", "tatt"):
        means[model + "_positive"] = first["mean"]
        means[model + "_negative"] = second["mean"]
    namespace = dict(np=np, cnu=cnu, sign_means=means,
                     row_kind=first["row_kind"], covariance=covariance)
    exec(source, namespace)
    result = namespace["distance"]
    two = result["fisher_two_point_likelihood"][0, 0]
    third = result["fisher_conditional_three_point"][0, 0]
    joint = result["fisher_joint_likelihood"][0, 0]
    return {"chi2_joint": joint, "chi2_two_point": two,
            "chi2_conditional_three_point": third,
            "conditional_residual": result["conditional_derivatives"][:, 0],
            "identity_relative_error": abs(joint-two-third)/max(abs(joint), 1.)}


def gaussian_sign_cases(latent_change, aperture_unit=1.0, interleaved=False):
    """Build four correlated observations from independent unit Gaussians.

    The first two observations are x1=2*u1 and x2=3*u2. The last two
    are y1=u1+2*u3 and y2=2*u2+u3+u4. Thus the joint distance of a
    latent displacement is its Euclidean squared length. The two-point
    distance contains only its first two entries. These facts give the
    expected answer independently of a Schur-complement calculation.
    """
    mixing = np.diag(v=[
        2.0,
        3.0,
        2.0,
        1.0,
    ])
    mixing[2, 0] = 1.0
    mixing[3, 1] = 2.0
    mixing[3, 2] = 1.0
    mixing[2:] *= aperture_unit
    covariance = mixing @ mixing.T
    difference = mixing @ np.asarray(a=latent_change, dtype=np.float64)
    labels = np.array(object=[
        "xi_plus",
        "xi_minus",
        "map3_small",
        "map3_large",
    ])
    kinds = np.array(object=[
        "xip",
        "xim",
        "map3",
        "map3",
    ])
    bins = np.zeros(shape=(4, 3), dtype=np.int32)
    angles = np.zeros(shape=(4, 3), dtype=np.float64)
    angles[0, 2] = 0.001
    angles[1, 2] = 0.002
    angles[2] = 0.003
    angles[3] = 0.004
    order = np.arange(4)
    if interleaved:
        order = np.array(object=[
            2,
            0,
            3,
            1,
        ])
    first = {
        "mean": difference[order].copy(),
        "row_labels": labels[order].copy(),
        "row_kind": kinds[order].copy(),
        "row_bins": bins[order].copy(),
        "row_angles_radians": angles[order].copy(),
    }
    second = {}
    for key, value in first.items():
        second[key] = value.copy()
    second["mean"][:] = 0.0
    return first, second, covariance[np.ix_(order, order)]


@pytest.mark.parametrize("aperture_unit", [1.0, 1.0e-8])
@pytest.mark.parametrize("interleaved", [False, True])
def test_latent_gaussian_distance_and_conditional_residual(
    aperture_unit, interleaved,
):
    """Four known independent innovations give 15=5+10 in any row units."""
    first, second, covariance = gaussian_sign_cases(
        latent_change=[
            1.0,
            -2.0,
            3.0,
            -1.0,
        ],
        aperture_unit=aperture_unit,
        interleaved=interleaved,
    )
    result = finite_sign_pair_separation(
        first=first,
        second=second,
        covariance=covariance,
    )
    np.testing.assert_allclose(
        actual=result["chi2_joint"],
        desired=15.0,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        actual=result["chi2_two_point"],
        desired=5.0,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        actual=result["chi2_conditional_three_point"],
        desired=10.0,
        rtol=2.0e-14,
    )
    # Removing the u1/u2 parts leaves (2*u3,u3+u4)=(6,2).
    np.testing.assert_allclose(
        actual=result["conditional_residual"] / aperture_unit,
        desired=[6.0, 2.0],
        rtol=2.0e-14,
    )
    assert result["identity_relative_error"] < 2.0e-14


def test_map3_change_can_be_entirely_predicted_by_two_point_rows():
    """Nonzero Map3 differences need not add conditional information."""
    first, second, covariance = gaussian_sign_cases(
        latent_change=[
            1.0,
            -2.0,
            0.0,
            0.0,
        ],
    )
    result = finite_sign_pair_separation(
        first=first,
        second=second,
        covariance=covariance,
    )
    assert np.all(first["mean"][2:] != 0.0)
    np.testing.assert_allclose(
        actual=result["chi2_joint"],
        desired=5.0,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        actual=result["conditional_residual"],
        desired=0.0,
        atol=2.0e-14,
    )
    assert result["chi2_conditional_three_point"] < 1.0e-26


def test_zero_contrast_and_sign_reversal_need_no_parameter_inverse():
    """Identical means give zero; exchanging means reverses only residuals."""
    first, second, covariance = gaussian_sign_cases(
        latent_change=[
            1.0,
            -2.0,
            3.0,
            -1.0,
        ],
    )
    forward = finite_sign_pair_separation(
        first=first,
        second=second,
        covariance=covariance,
    )
    reverse = finite_sign_pair_separation(
        first=second,
        second=first,
        covariance=covariance,
    )
    zero = finite_sign_pair_separation(
        first=first,
        second=first,
        covariance=covariance,
    )
    for key in (
        "chi2_joint",
        "chi2_two_point",
        "chi2_conditional_three_point",
    ):
        assert forward[key] == reverse[key]
        assert zero[key] == 0.0
    np.testing.assert_array_equal(
        x=forward["conditional_residual"],
        y=-reverse["conditional_residual"],
    )


def test_map3_sign_change_requires_the_matching_cross_covariance_change():
    """Changing both row signs is a convention; changing only C23 is not."""
    first, second, covariance = gaussian_sign_cases(
        latent_change=[
            1.0,
            -2.0,
            3.0,
            -1.0,
        ],
    )
    signs = np.ones(shape=4)
    signs[2:] = -1.0
    changed_covariance = signs[:, None] * covariance * signs[None, :]
    inconsistent = finite_sign_pair_separation(
        first=first,
        second=second,
        covariance=changed_covariance,
    )
    # With only C23 reversed, the implied independent innovations become
    # (1,-2,-4,10), whose squared length is 121 rather than 15.
    np.testing.assert_allclose(
        actual=inconsistent["chi2_joint"],
        desired=121.0,
        rtol=2.0e-14,
    )
    first["mean"] *= signs
    second["mean"] *= signs
    consistent = finite_sign_pair_separation(
        first=first,
        second=second,
        covariance=changed_covariance,
    )
    np.testing.assert_allclose(
        actual=consistent["chi2_joint"],
        desired=15.0,
        rtol=2.0e-14,
    )
