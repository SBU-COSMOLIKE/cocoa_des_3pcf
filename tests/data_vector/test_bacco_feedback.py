"""Check the declared neutrino closure, continuation and dynamic response cache.

Analytic component sums test physics independently of the neural network.
The small installed-network check verifies actual API ordering and native
table ownership; its coarse grids are not a survey accuracy certificate.
"""

from test_pipeline import compact_bispectrum_grid

from itertools import permutations, product

import numpy as np
import pytest

from des3pcf.accuracy import Accuracy
from des3pcf.bacco_feedback import (
    BaccoFeedback, BaccoSettings, continued_triangles, smooth_turn_on,
    total_bispectrum_response, total_power_response,
)
from test_pipeline import toy_tables


def response_inputs():
    """Use a small, positive toy neutrino spectrum with a known cold fraction."""
    tables, parameters = toy_tables()
    parameters.update(M_c_baccoemu=14., eta_baccoemu=-.3, beta_baccoemu=-.22,
                      M1_z0_cen_baccoemu=10.5, theta_inn_baccoemu=-.86)
    tables["p_linear_nu"] = tables["p_nonlinear"] * .25
    metadata = {"omega_nu_h2": .00065, "sigma8_cb": .802,
                "parameters": dict(parameters)}
    return tables, parameters, metadata


def test_power_is_the_sum_of_coherent_components():
    """Changing cold auto-power changes the cross-power by its square root."""
    response = np.array([.3, .7, 1.2])
    cold = np.array([.5, .9, .995])
    nu = 1. - cold
    expected = cold**2 * response + 2. * cold * nu * np.sqrt(response) + nu**2
    actual = total_power_response(response, cold, "coherent_linear_neutrinos")
    np.testing.assert_allclose(actual, expected, rtol=1.e-15)
    # Each field covariance is positive semidefinite, including suppression.
    for r, a in zip(response, cold):
        matrix = np.array([[a*a*r, a*(1.-a)*np.sqrt(r)],
                           [a*(1.-a)*np.sqrt(r), (1.-a)**2]])
        assert np.linalg.eigvalsh(matrix).min() >= -1.e-16


@pytest.mark.parametrize("weight", [0., .997, 1.])
def test_unity_and_massless_limits(weight):
    """No feedback leaves unity; zero neutrino amplitude recovers cold ratios."""
    ones = np.ones((2, 3))
    mode = "coherent_linear_neutrinos"
    np.testing.assert_allclose(total_power_response(ones, weight, mode), ones)
    np.testing.assert_allclose(
        total_bispectrum_response(np.ones(2), ones, ones*weight, mode), 1.,
        rtol=3.e-16,
    )
    if weight == 1.:
        np.testing.assert_array_equal(total_power_response(.7, weight, mode), .7)
        np.testing.assert_array_equal(
            total_bispectrum_response(np.array([.3, 1.4]), ones*.7, ones, mode),
            [.3, 1.4],
        )


def test_bispectrum_all_eight_assignments_and_permutations():
    """Sum labeled component placements, including the independently supplied B_cb."""
    ratios = np.array([[.6, .8, 1.1]])
    weights = np.array([[.4, .7, .99]])
    cold_b = np.array([.4])
    expected = 0.
    for assignment in product((0, 1), repeat=3):
        if all(assignment):
            expected += np.prod(weights) * cold_b[0]
        else:
            factors = [weights[0, j]*np.sqrt(ratios[0, j]) if bit else
                       1.-weights[0, j] for j, bit in enumerate(assignment)]
            expected += np.prod(factors)
    for order in permutations(range(3)):
        actual = total_bispectrum_response(cold_b, ratios[:, order], weights[:, order],
                                          "coherent_linear_neutrinos")
        np.testing.assert_allclose(actual, expected, rtol=4.e-16)
    np.testing.assert_array_equal(
        total_bispectrum_response(cold_b, ratios, weights, "cold_on_total"), cold_b,
    )


@pytest.mark.parametrize("mode", ["shape_preserving", "side_clamp"])
def test_continuation_keeps_squeezed_feedback_and_triangle_rows(mode):
    """One soft side retains feedback; an all-soft triangle approaches unity."""
    settings = BaccoSettings(high_k_mode=mode)
    triangles = np.array([[1.e-8, 2., 2.], [.001, .001, .001],
                          [20., 30., 40.], [1., 2., 3.]])
    mapped, taper = continued_triangles(triangles, settings)
    assert mapped.shape == triangles.shape
    assert np.all(mapped[:, 0] + mapped[:, 1] > mapped[:, 2])
    np.testing.assert_array_equal(taper[[0, 2, 3]], 1.)
    assert 0. < taper[1] < .01
    if mode == "shape_preserving":
        np.testing.assert_allclose(mapped[2], [5., 7.5, 10.])
    else:
        np.testing.assert_array_equal(mapped[2], [10., 10., 10.])


@pytest.mark.parametrize("sides", [[[1., 1., 3.]], [[0., 1., 1.]],
                                 [[1., np.nan, 1.]]])
def test_bad_triangles_are_not_repaired(sides):
    """Extrapolation must never turn an invalid physical request into a valid one."""
    with pytest.raises(ValueError):
        continued_triangles(sides, BaccoSettings())


def test_smooth_taper_endpoints():
    """The low-k and high-z joins retain positivity through log responses."""
    np.testing.assert_array_equal(smooth_turn_on([-10., 0., .5, 1., 10.]),
                                  [0., 0., .5, 1., 1.])


def test_actual_bacco_native_and_cache_sequences(compiled):
    """Change physics, source spectra and sampling, then restore the old native model.

    The neural networks and C constructor are real. No CAMB solve is needed:
    toy tables give a controlled positive neutrino amplitude. Restoring the
    parameter point must recover the same table without mutating an older
    model still held by a notebook variable.
    """
    from theory_inputs import prepared_physics_model as physics_model

    tables, point, metadata = response_inputs()
    producer = BaccoFeedback(BaccoSettings(nz=4, nk=9, nu=5, nv=4, boost=1))
    first = producer.prepare(point, tables, metadata)
    assert producer.prepare(point, tables, metadata) is first
    original = first.native["log_bispectrum"].copy()
    native = physics_model(compiled=compiled, tables=tables, feedback=first)
    sides = np.array([[.1, .2, .25], [1., 2., 2.5]])
    z = np.array([.4, 1.2])
    old_response = native.bispectrum_response(sides=sides, redshifts=z, amplitude=1.)
    moved = dict(point, M_c_baccoemu=14.2)
    second = producer.prepare(moved, tables, metadata)
    assert np.max(np.abs(second.native["log_bispectrum"] - original)) > 1.e-4
    third = producer.prepare(point, tables, metadata)
    np.testing.assert_array_equal(third.native["log_bispectrum"], original)
    np.testing.assert_array_equal(
        native.bispectrum_response(sides=sides, redshifts=z, amplitude=1.), old_response,
    )
    tables["p_linear_nu"] *= 2.
    changed = producer.prepare(point, tables, metadata)
    assert np.max(np.abs(changed.native["log_power"]-third.native["log_power"])) > 0.
    refined = producer.prepare(point, tables, metadata, Accuracy(feedback_k_boost=2.))
    assert refined.native["logk"].size == 17
    with pytest.raises(AttributeError):
        producer.settings = BaccoSettings()
    record = refined.record()
    record["coordinates"]["M_c"] = 99.
    assert refined.record()["coordinates"]["M_c"] == 14.
    with pytest.raises(ValueError, match="changing sigma8"):
        producer.prepare(dict(point, sigma8=.81), tables, metadata)
    incomplete = dict(metadata)
    incomplete["parameters"] = dict(metadata["parameters"])
    del incomplete["parameters"]["sigma8"]
    with pytest.raises(ValueError, match="complete CAMB provenance"):
        producer.prepare(parameters=point, tables=tables, metadata=incomplete)
    absent = dict(tables)
    absent.pop("p_linear_nu")
    with pytest.raises(ValueError, match="p_linear_nu"):
        producer.prepare(point, absent, metadata)


def test_joint_prediction_rejects_amplitude_and_rebuilds_bcm(compiled):
    """A notebook's BCM A/B/A sequence changes both xi and Map3 and then restores."""
    from des3pcf.pipeline import JointPrediction, PhysicalProvider
    from des3pcf.survey import forecast_rows

    tables, point, metadata = response_inputs()
    provider = PhysicalProvider(
        bispectrum_table=compact_bispectrum_grid(zmax=tables["z"][-1]),
        compiled=compiled, reference_tables=tables, reference_metadata=metadata,
        reference_parameters=point,
        feedback=BaccoFeedback(BaccoSettings(nz=4, nk=9, nu=5, nv=4, boost=1)),
    )
    rows = forecast_rows(nbin=2, aperture_radii=[.003], xi_edges=[.001, .002])
    prediction = JointPrediction(
        compiled=compiled, rows=rows, provider=provider,
        ia_defaults={"a1": .4, "feedback_amplitude": 1.},
        aperture_estimator="full_filter", aperture_settings={
            "projection_mode": "direct",
            "ell": {"lower": 20., "upper": 3000., "count": 6, "logarithmic": True},
            "phi": {"lower": 0., "upper": 2*np.pi, "count": 6},
        }, xi_settings={
            "ell": {"lower": 20., "upper": 3000., "count": 24, "logarithmic": True},
            "theta_count": 4, "pair_weight": "uniform_area",
        },
    )
    a = prediction(point)
    b = prediction(dict(point, eta_baccoemu=-.2))
    for selected in (rows.kind == "map3", rows.kind != "map3"):
        assert np.max(np.abs(b[selected] - a[selected])) > 0.
    np.testing.assert_array_equal(prediction(point), a)
    with pytest.raises(ValueError, match="fixed feedback_amplitude"):
        prediction(dict(point, feedback_amplitude=.5))
