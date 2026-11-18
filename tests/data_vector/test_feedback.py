"""Check paired matter feedback without assuming a P-to-B scaling relation.

Constant independent responses have exact projected predictions. Multilinear
log responses test the uniform-coordinate reader between input nodes, while
the IA split tests the declared fixed-quadratic-IA approximation. Smooth
test tables are numerical references, not hydro simulations.
"""

import numpy as np
import pytest

from des3pcf.feedback import FeedbackPair, feedback_power_tables
from test_physics_model import physical_inputs, compact_grid_settings


def paired_input(power=0.8, bispectrum=0.6, boost=2):
    """Specify independent constant ratios over a broad explicit clamped domain."""
    z = np.linspace(0., 2., 4)
    logk = np.linspace(-12., 12., 4)
    logu = np.linspace(-12., 0., 4)
    v = np.linspace(0., 1., 4)
    return FeedbackPair(
        z=z, logk=logk, logu=logu, v=v,
        log_power=np.full(shape=(4, 4), fill_value=np.log(power)),
        log_bispectrum=np.full(shape=(4, 4, 4, 4), fill_value=np.log(bispectrum)),
        metadata={
            "name": "independent constant response test",
            "source_url": "analytic constant ratios; no hydro calibration",
            "matter_field": "total",
            "cosmology_dependence": "fixed_ratio",
            "ia_response": "nla_only",
            "amplitude_model": "log_response",
            "domain": "clamp",
        },
        boost=boost,
    )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_independent_power_and_bispectrum_scalings(compiled, adapter):
    """Pure NLA scales by R_P and R_B independently and restores at A=0."""
    pair = paired_input()
    inputs = physical_inputs()
    model = getattr(compiled, adapter).PhysicsModel(**inputs, feedback=dict(pair.native))
    reference = compiled.PhysicsModel(**inputs)
    ell = np.array([100., 210., 350.])
    bins2 = np.array([0, 1], dtype=np.int32)
    bins3 = np.array([0, 1, 0], dtype=np.int32)
    sides = np.array([[100., 210., 250.], [200., 270., 340.]])
    parameters = {"a1": 0.6}
    power_method = model.nla_power
    bispectrum_method = model.projected
    if adapter == "notebook":
        power_method = model.nla_power_notebook
        bispectrum_method = model.projected_notebook
    original_p = reference.nla_power(ell=ell, bins=bins2, parameters=parameters)
    original_b = reference.projected(sides=sides, bins=bins3, parameters=parameters)
    for amplitude in (0., 1., -0.3, 0.):
        selected = dict(parameters, feedback_amplitude=amplitude)
        actual_p = power_method(ell=ell, bins=bins2, parameters=selected)
        actual_b = bispectrum_method(sides=sides, bins=bins3, parameters=selected)
        np.testing.assert_allclose(actual_p, original_p * 0.8**amplitude, rtol=3.e-14)
        np.testing.assert_allclose(actual_b, original_b * 0.6**amplitude, rtol=3.e-14)
        if amplitude == 0.:
            np.testing.assert_array_equal(actual_p, original_p)
            np.testing.assert_array_equal(actual_b, original_b)
    modified = feedback_power_tables(model=model, tables=inputs, amplitude=1.)
    np.testing.assert_allclose(modified["p_nonlinear"], inputs["p_nonlinear"] * .8)
    np.testing.assert_array_equal(modified["p_linear0"], inputs["p_linear0"])


def test_feedback_leaves_quadratic_ia_exactly_fixed(compiled):
    """Changing A must not change even the residual's interpolation scale."""
    pair = paired_input()
    model = compiled.PhysicsModel(**physical_inputs(), feedback=dict(pair.native))
    bins = np.array([0, 1, 0], dtype=np.int32)
    sides = np.array([[120., 180., 250.], [200., 270., 340.]])
    parameters = {"a1": .4, "a2": .7, "bta": .9, "ct": .2}
    settings = compact_grid_settings(reference_mode=0)
    settings.update(signed_baseline=1, residual_scaling=1)
    baseline = model.projection_grid(bins=bins, parameters=parameters, settings=settings)
    residual = baseline.projected(sides=sides, component=2)
    for amplitude in (.5, 1., 0.):
        selected = dict(parameters, feedback_amplitude=amplitude)
        grid = model.projection_grid(bins=bins, parameters=selected, settings=settings)
        np.testing.assert_array_equal(grid.projected(sides=sides, component=2), residual)
        np.testing.assert_allclose(
            grid.projected(sides=sides, component=1),
            baseline.projected(sides=sides, component=1) * .6**amplitude,
            rtol=5.e-14,
        )


@pytest.mark.parametrize("boost", [1, 3])
def test_multilinear_log_response_and_input_ownership(compiled, boost):
    """Analytic linear coordinate dependence survives cubic preparation exactly."""
    pair = paired_input(boost=boost)
    native = dict(pair.native)
    native["z"] = np.linspace(0., 2., 5)
    native["logk"] = np.linspace(-12., 12., 6)
    native["logu"] = np.linspace(-40., 0., 7)
    native["log_power"] = np.full(shape=30, fill_value=np.log(.8))
    z, logk, logu, v = np.meshgrid(
        native["z"], native["logk"], native["logu"], native["v"], indexing="ij",
    )
    native["log_bispectrum"] = (
        -.03*z + .01*logk + .02*logu + .04*v + .004*z*logk + .003*logu*v
    ).ravel()
    model = compiled.PhysicsModel(**physical_inputs(), feedback=native)
    native["log_bispectrum"][:] = 500.
    sides = np.array([[.3, .4, .5], [.8, 1., 1.3], [1.e-12, 1., 1.+5.e-13]])
    redshift = np.array([.4, 1.4, .7])
    expected_log = -.03*redshift + .01*np.log(sides[:, 1])
    expected_log += .02*np.log(sides[:, 0]/sides[:, 1])
    expected_log += .04*(sides[:, 2] - sides[:, 1])/sides[:, 0]
    expected_log += .004*redshift*np.log(sides[:, 1])
    expected_log += (.003*np.log(sides[:, 0]/sides[:, 1])
                     * (sides[:, 2] - sides[:, 1])/sides[:, 0])
    for amplitude in (1., 0.3):
        expected = np.exp(amplitude * expected_log)
        actual = model.bispectrum_response(
            sides=sides, redshifts=redshift, amplitude=amplitude,
        )
        swapped = model.bispectrum_response_notebook(
            sides=sides[:, ::-1], redshifts=redshift, amplitude=amplitude,
        )
        np.testing.assert_allclose(actual, expected, rtol=1.e-14)
        np.testing.assert_allclose(swapped, expected, rtol=1.e-14)


def test_joint_feedback_cache_and_metadata(compiled):
    """One retained DMO model supports A/B/A without a new Boltzmann solve."""
    from des3pcf.pipeline import PhysicalProvider, JointPrediction
    from des3pcf.survey import forecast_rows

    tables = physical_inputs()
    parameters = {"omega_m": .3, "omega_b": .05, "h": .7, "ns": .965,
                  "sigma8": .8, "w0": -1., "wa": 0.}
    provider = PhysicalProvider(
        compiled=compiled, reference_tables=tables, reference_metadata={},
        reference_parameters=parameters, feedback=paired_input(),
    )
    rows = forecast_rows(nbin=2, aperture_radii=[.003], xi_edges=[.001, .002])
    prediction = JointPrediction(
        compiled=compiled, rows=rows, provider=provider, ia_defaults={"a1": .4},
        aperture_estimator="full_filter",
        aperture_settings={
            "projection_mode": "direct",
            "ell": {"lower": 20., "upper": 3000., "count": 6,
                    "logarithmic": True},
            "phi": {"lower": 0., "upper": 2*np.pi, "count": 6},
        },
        xi_settings={
            "ell": {"lower": 20., "upper": 3000., "count": 24,
                    "logarithmic": True},
            "theta_count": 4,
            "pair_weight": "uniform_area",
        },
    )
    original = prediction(parameters=parameters)
    owner = provider.model
    changed = prediction(parameters=dict(parameters, feedback_amplitude=1.))
    expected = np.where(rows.kind == "map3", .6, .8) * original
    np.testing.assert_allclose(changed, expected, rtol=5.e-14)
    assert provider.model is owner
    np.testing.assert_array_equal(prediction(parameters=parameters), original)
    with pytest.raises(AttributeError, match="new.*provider"):
        provider.feedback = paired_input(power=.7)


def test_feedback_two_point_tatt_keeps_loop_terms_fixed(compiled):
    """Actual CFASTPT loops stay fixed while the nonlinear baseline is suppressed."""
    import sys
    from pathlib import Path
    from des3pcf.core_two_point import CoreTwoPoint

    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project.parent / "des_y3/interface"))
    import cosmolike_des_y3_interface as core

    tables = physical_inputs()
    tables["p_linear"] = tables["growth"][:, None]**2 * tables["p_linear0"]
    model = compiled.PhysicsModel(**physical_inputs(), feedback=dict(paired_input().native))
    adapter = CoreTwoPoint(core=core, nbin=2, threads=1, ell_internal=32,
                           accuracy_boost=.5)
    parameters = {"h": .7, "omega_b": .05, "mnu": 0.}
    ell = np.array([60., 170., 500.])
    bins = np.array([[0, 1]], dtype=np.int32)
    references = None
    for amplitude in (0., 1., 0.):
        modified = feedback_power_tables(model=model, tables=tables, amplitude=amplitude)
        metadata = {"paired_feedback_amplitude": amplitude}
        ia = {"a1": .4, "a2": .7, "bta": .9, "feedback_amplitude": amplitude}
        even, odd = adapter.spectra(parameters=parameters, tables=modified,
                                    metadata=metadata, ia=ia, ell=ell, bins=bins)
        baseline, zero = adapter.spectra(
            parameters=parameters, tables=modified, metadata=metadata,
            ia=dict(ia, a2=0., bta=0.), ell=ell, bins=bins,
        )
        residual = even - baseline
        if references is None:
            references = (baseline, residual, odd)
        np.testing.assert_allclose(baseline, references[0] * .8**amplitude, rtol=2.e-9)
        np.testing.assert_allclose(residual, references[1], rtol=2.e-9)
        np.testing.assert_array_equal(odd, references[2])
