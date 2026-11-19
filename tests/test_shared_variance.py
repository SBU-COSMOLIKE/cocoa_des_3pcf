"""Check shared smoothing through the real project adapter.

These tests use a controlled spectrum, not survey data. They establish
state ownership, physical input selection and numerical control wiring.
Full joint means and Fisher derivatives require separate survey checks.
"""

import numpy as np
import pytest

from des3pcf.accuracy import Accuracy
from des3pcf.pipeline import PhysicalProvider
from des3pcf.variance import SharedVariance, ReferenceVariance
from test_pipeline import toy_tables


def request(engine, tables, parameters, **settings):
    """Read three fit redshifts with an explicitly massless input model."""
    return engine.prepare(
        tables=tables, parameters=parameters, metadata={},
        z=np.array([0.0, 0.5, 1.0]), **settings,
    )


@pytest.mark.parametrize("fit_nz,table_boost", [(17, 4), (33, 8)])
@pytest.mark.parametrize("source_readers", [False, True])
def test_python_reference_can_supply_the_native_fit(
        compiled, fit_nz, table_boost, source_readers):
    """External reference arrays reproduce the legacy native matter equation."""
    from des3pcf.mean import physics_model

    tables, unused = toy_tables()
    # Call the native legacy constructor directly: the Python factory also
    # supplies reference fit arrays, so comparing two factory calls would
    # fail to test the independent C integration being replaced.
    reference = compiled.production.PhysicsModel(
        **tables, fit_nz=fit_nz, table_boost=table_boost,
        source_readers=source_readers,
    )
    engine = ReferenceVariance()
    fit, multiplier, record = engine.prepare(
        tables=tables, fit_nz=fit_nz, table_boost=table_boost,
        source_readers=source_readers,
    )
    normalized = dict(tables)
    normalized["p_linear0"] = tables["p_linear0"] * multiplier
    candidate = physics_model(compiled=compiled, tables=normalized,
                              fit_nz=fit_nz, table_boost=table_boost,
                              variance=fit, source_readers=source_readers)
    sides = np.array([[0.03, 0.04, 0.05], [0.1, 0.2, 0.25], [0.5, 0.7, 1.0]])
    redshift = np.array([0.0, 0.4, 1.8])
    for component in range(4):
        expected = reference.matter(sides=sides, redshifts=redshift,
                                    component=component)
        actual = candidate.matter(sides=sides, redshifts=redshift,
                                   component=component)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=0.0)
    assert record["power_normalization"] == multiplier


def test_reference_variance_retains_inputs_outputs_and_exact_keys():
    """A/B/A restores fit arrays and a caller cannot alter retained results."""
    tables, unused = toy_tables()
    engine = ReferenceVariance()
    original, multiplier, record = engine.prepare(tables=tables)
    saved = original["rnl"].copy()
    with pytest.raises(ValueError):
        original["rnl"][0] = 0.0
    record["sampling"]["variance_nodes"] = 0
    same, same_multiplier, clean = engine.prepare(tables=tables)
    assert same is original
    assert same_multiplier == multiplier
    assert clean["sampling"]["variance_nodes"] == 1000
    changed = dict(tables)
    changed["sigma8"] += 0.02
    displaced, unused_multiplier, unused_record = engine.prepare(tables=changed)
    assert not np.array_equal(displaced["rnl"], saved)
    restored, restored_multiplier, unused_record = engine.prepare(tables=tables)
    np.testing.assert_array_equal(restored["rnl"], saved)
    assert restored_multiplier == multiplier
    np.testing.assert_array_equal(original["rnl"], saved)


def test_snapshot_ownership_and_interleaved_owners(compiled):
    """A second owner or a changed spectrum cannot alter an earlier result."""
    tables, parameters = toy_tables()
    first = SharedVariance(compiled=compiled)
    original, record = request(engine=first, tables=tables, parameters=parameters)
    saved = original["rnl"].copy()
    with pytest.raises(ValueError):
        original["rnl"][0] = 0.0
    record["sampling"]["mass_nodes"] = 2
    second = SharedVariance(compiled=compiled)
    changed = dict(tables)
    changed["p_linear0"] = 1.1 * tables["p_linear0"]
    displaced, unused = request(engine=second, tables=changed, parameters=parameters)
    assert not np.array_equal(displaced["rnl"], saved)
    restored, clean_record = request(engine=first, tables=tables, parameters=parameters)
    assert restored is original
    assert clean_record["sampling"]["mass_nodes"] != 2
    np.testing.assert_array_equal(restored["rnl"], saved)
    request(engine=first, tables=changed, parameters=parameters)
    again, unused = request(engine=first, tables=tables, parameters=parameters)
    np.testing.assert_array_equal(again["rnl"], saved)


@pytest.mark.parametrize("boost,fields", [
    ("variance_mass_boost", ["mass_nodes"]),
    ("variance_a_boost", ["scale_factor_nodes", "gaussian_scale_factor_nodes"]),
    ("variance_radius_boost", ["gaussian_radius_nodes"]),
    ("variance_inverse_boost", ["gaussian_inverse_nodes"]),
])
def test_individual_sampling_roundtrip(compiled, boost, fields):
    """Change only the requested grids and restore the original snapshot."""
    tables, parameters = toy_tables()
    engine = SharedVariance(compiled=compiled)
    original, baseline = request(engine=engine, tables=tables, parameters=parameters)
    accuracy = Accuracy(**{boost: 2.0})
    refined, record = request(
        engine=engine, tables=tables, parameters=parameters, accuracy=accuracy,
    )
    for name, count in baseline["sampling"].items():
        expected = count
        if name in fields:
            expected = 2 * (count - 1) + 1
        assert record["sampling"][name] == expected
    repeated, unused = request(
        engine=engine, tables=tables, parameters=parameters, accuracy=accuracy,
    )
    assert repeated is refined
    restored, unused = request(engine=engine, tables=tables, parameters=parameters)
    for name in ("rnl", "neff", "sigma8"):
        np.testing.assert_array_equal(restored[name], original[name])


def test_power_amplitude_and_growth_are_explicit(compiled):
    """Trust the power amplitude; full evolving power is a separate choice."""
    tables, parameters = toy_tables()
    engine = SharedVariance(compiled=compiled)
    original, unused = request(engine=engine, tables=tables, parameters=parameters)
    changed = dict(tables)
    changed["sigma8"] = 4.0
    same, unused = request(engine=engine, tables=changed, parameters=parameters)
    assert same is original
    changed["p_linear0"] = 1.21 * tables["p_linear0"]
    amplified, unused = request(engine=engine, tables=changed, parameters=parameters)
    np.testing.assert_allclose(amplified["sigma8"], 1.1 * original["sigma8"], rtol=1e-12)
    tables["p_linear"] = tables["growth"][:, None]**2 * tables["p_linear0"][None, :]
    evolving, unused = request(
        engine=engine, tables=tables, parameters=parameters, growth_mode="evolving",
    )
    np.testing.assert_array_equal(evolving["rnl"], original["rnl"])
    tables["p_linear"] *= 1.0 + 0.1 * tables["z"][:, None]
    different, record = request(
        engine=engine, tables=tables, parameters=parameters, growth_mode="evolving",
    )
    assert record["growth_mode"] == "evolving"
    assert different["rnl"][-1] != evolving["rnl"][-1]


def test_invalid_request_preserves_sampling_and_snapshot(compiled):
    """Refuse missing neutrino density before installing new accuracy counts."""
    tables, parameters = toy_tables()
    engine = SharedVariance(compiled=compiled)
    original, record = request(engine=engine, tables=tables, parameters=parameters)
    parameters["mnu"] = 0.06
    with pytest.raises(ValueError, match="omega_nu_h2"):
        request(engine=engine, tables=tables, parameters=parameters,
                accuracy=Accuracy(variance_a_boost=2.0))
    assert compiled.core.variance_sampling() == record["sampling"]
    parameters["mnu"] = 0.0
    restored, unused = request(engine=engine, tables=tables, parameters=parameters)
    assert restored is original


def test_provider_fit_copies_survive_accuracy_and_source_changes(compiled):
    """An old model stays valid while a notebook changes preparation inputs."""
    tables, parameters = toy_tables()
    provider = PhysicalProvider(
        compiled=compiled, reference_tables=tables, reference_metadata={},
        reference_parameters=parameters, cosmology_mode="fixed_tables",
        variance_source="shared",
    )
    sides = np.array([[0.1, 0.2, 0.25], [0.5, 0.7, 1.0]])
    redshift = np.array([0.0, 0.8])
    original, unused, record = provider.get(parameters=parameters)
    values = original.matter(sides=sides, redshifts=redshift)
    assert record["variance"]["growth_mode"] == "factorized"
    changed = dict(parameters)
    changed["dz1"] = 0.02
    displaced, unused, unused_record = provider.get(parameters=changed)
    np.testing.assert_array_equal(displaced.matter(sides=sides, redshifts=redshift), values)
    provider.accuracy = Accuracy(variance_a_boost=2.0)
    provider.get(parameters=parameters)
    np.testing.assert_array_equal(original.matter(sides=sides, redshifts=redshift), values)
    provider.accuracy = Accuracy()
    restored, unused, unused_record = provider.get(parameters=parameters)
    np.testing.assert_array_equal(restored.matter(sides=sides, redshifts=redshift), values)


def test_integration_boost_does_not_refine_analytic_smoothing(compiled):
    """The FFTLog smoothing counts are independent of fixed quadrature counts."""
    engine = SharedVariance(compiled=compiled)
    original = Accuracy().variance_settings(baseline=engine.baseline)
    refined = Accuracy(integration_accuracy_boost=2.0).variance_settings(baseline=engine.baseline)
    assert original == refined


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
def test_covariance_preparation_uses_matching_shared_tables(compiled, adapter_name):
    """The covariance consumes the recorded mass, density and redshift axes."""
    from covariance.test_physical_covariance import covariance_arguments

    tables, parameters = toy_tables()
    provider = PhysicalProvider(
        compiled=compiled, reference_tables=tables, reference_metadata={},
        reference_parameters=parameters, cosmology_mode="fixed_tables",
        variance_source="shared",
    )
    inputs = covariance_arguments()
    model, prepared, record = provider.covariance_model(parameters=parameters, inputs=inputs)
    assert "halo_sigma2" not in inputs
    assert record["halo_variance"]["reference_rho"] == 2.77536627e11 * tables["omega_m"]
    np.testing.assert_array_equal(prepared["halo_mass"], inputs["mass"])
    np.testing.assert_array_equal(prepared["halo_z"], model.radial()["z"])
    adapter = getattr(compiled, adapter_name)
    values = adapter.halo_tables(model=model, inputs=prepared, z=prepared["halo_z"],
                                 k_profile=np.array([0.0, 0.1]))
    np.testing.assert_array_equal(values[0], prepared["halo_sigma2"])
    saved = prepared["halo_sigma2"].copy()
    provider.accuracy = Accuracy(variance_mass_boost=2.0)
    provider.covariance_model(parameters=parameters, inputs=inputs)
    np.testing.assert_array_equal(prepared["halo_sigma2"], saved)
