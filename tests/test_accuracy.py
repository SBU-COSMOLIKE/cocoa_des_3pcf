"""Check count refinement separately from the physical convergence tests.

These checks establish what a control changes. They do not claim that a
larger count has converged a survey calculation. Input table resolution,
integration limits, estimator bins and angular truncations remain separate
physical/numerical choices and must never be changed by an implicit boost.
"""



import numpy as np
import pytest

from cosmolike_des_3pcf_interface import covariance_inputs
from cobaya.likelihoods.des_3pcf.survey import survey_rows


import cosmolike_des_3pcf_interface as ci


def test_nested_los_refinement_preserves_nodes_and_baseline():
    """Refining twice from one baseline must not compound or move old nodes."""
    baseline = np.array([0.03, 0.12, 0.57, 1.8])
    controls = dict(accuracy_boost=1.1, chi_boost=1.5)
    first = ci.accuracy_los_nodes(accuracy=controls, baseline=baseline)
    second = ci.accuracy_los_nodes(accuracy=controls, baseline=baseline)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[::2], baseline)
    np.testing.assert_array_equal(ci.accuracy_los_nodes(accuracy=dict(), baseline=baseline), baseline)
    assert first.size == 7


@pytest.mark.parametrize("name", [
    "chi_boost", "kernel_boost", "source_boost", "power_table_boost", "fit_z_boost",
    "triangle_boost", "multipole_boost", "angular_boost", "fft_boost",
    "profile_boost", "profile_upsample_boost", "aperture_ell_boost",
    "aperture_phi_boost", "xi_ell_boost", "xi_theta_boost",
    "covariance_ell_boost", "covariance_window_boost", "covariance_phi_boost",
    "covariance_radius_boost", "covariance_k_boost", "covariance_mass_boost",
])
def test_individual_controls_do_not_change_unrelated_counts(name):
    """Every stage-only knob must leave all other stage multipliers at one."""
    controls = ci.init_3pcf_accuracy(value={name: 2.0})
    for other, value in controls.items():
        if other == name:
            assert value == 2.0
        else:
            assert value == 1.0
    assert ci.accuracy_count(accuracy=controls, baseline=9, stage=name, endpoints=True) == 17
    assert ci.accuracy_count(accuracy=controls, baseline=9, stage=name, integration=True) == 18


def test_global_integration_and_sampling_have_distinct_effects():
    """Separate interpolation density from fixed-rule node counts."""
    sampling = dict(accuracy_boost=2.0)
    integration = dict(integration_accuracy_boost=3.0)
    assert ci.accuracy_count(accuracy=sampling, baseline=9, stage="triangle_boost", endpoints=True) == 17
    assert ci.accuracy_count(accuracy=sampling, baseline=9, stage="angular_boost", integration=True) == 9
    assert ci.accuracy_count(accuracy=integration, baseline=9, stage="triangle_boost", endpoints=True) == 9
    assert ci.accuracy_count(accuracy=integration, baseline=9, stage="angular_boost", integration=True) == 27
    baseline = {"lmax": 3, "ell_min": 20.0, "ell_max": 3000.0}
    resolved = ci.accuracy_projection_settings(accuracy=integration, baseline=baseline)
    assert resolved["lmax"] == 3
    assert resolved["ell_min"] == 20.0
    assert resolved["ell_max"] == 3000.0
    assert resolved["nmu_log"] == 90
    assert baseline == {"lmax": 3, "ell_min": 20.0, "ell_max": 3000.0}


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan, True])
def test_bad_boost_refused_before_preparation(value):
    """Count controls have a named failure before any C allocation can occur."""
    with pytest.raises(ValueError, match="accuracy_boost"):
        ci.init_3pcf_accuracy(value={"accuracy_boost": value})
    with pytest.raises(ValueError, match="Unknown accuracy"):
        ci.init_3pcf_accuracy(value={"chi_bost": 2.0})


def test_covariance_controls_refine_only_declared_integrals():
    """Mass-only refinement must not alter ell nodes, source labels or bin edges."""
    rows = survey_rows(nbin=2, aperture_radii=[0.003], xi_edges=[0.001, 0.002])
    grids = {}
    for name in ("ell", "window_ell", "phi", "radius", "k", "mass"):
        grids[name] = {"lower": 0.1, "upper": 2.0, "count": 3}
    grids["profile_radius"] = np.array([0.0, 0.01, 0.03, 0.1])
    arguments = {
        "kind": rows.kind.tolist(),
        "bins": rows.bins,
        "angles": rows.angles,
        "noise": np.eye(2),
        "grids": grids,
        "physical": {"large_area": 1, "noise_mode": 1},
        "theta_count": 3,
        "pair_weight": "uniform_area",
    }
    baseline = covariance_inputs(**arguments)
    refined = covariance_inputs(**arguments, accuracy={"covariance_mass_boost": 2.0})
    assert baseline["mass"].size == 3
    assert refined["mass"].size == 6
    for name in ("ell", "dell", "theta_edges", "row_fields", "field_source"):
        np.testing.assert_array_equal(refined[name], baseline[name])
    assert grids["mass"]["count"] == 3
    # Returned preparation arrays must not borrow a writable caller buffer.
    arguments["noise"][:] = 0.0
    grids["profile_radius"][:] = 4.0
    np.testing.assert_array_equal(baseline["noise"], np.eye(2))
    np.testing.assert_array_equal(baseline["profile_radius"], [0.0, 0.01, 0.03, 0.1])


def test_measured_pair_weights_cannot_be_silently_resampled():
    """A numerical boost cannot invent catalogue pair weights at new angles."""
    rows = survey_rows(nbin=2, aperture_radii=[], xi_edges=[0.001, 0.002])
    grids = {}
    for name in ("ell", "window_ell", "phi", "radius", "k", "mass"):
        grids[name] = {"lower": 0.1, "upper": 2.0, "count": 3}
    grids["profile_radius"] = np.array([0.0, 0.01, 0.03, 0.1])
    with pytest.raises(ValueError, match="new measured pair weights"):
        covariance_inputs(
            kind=rows.kind.tolist(),
            bins=rows.bins,
            angles=rows.angles,
            noise=np.eye(2),
            grids=grids,
            physical={},
            theta_count=3,
            pair_weight=np.ones((rows.values.size, 3)),
            accuracy={"xi_theta_boost": 2.0},
        )
