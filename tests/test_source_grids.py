"""Separate source reconstruction, source sampling and power-table sampling.

The curves are controlled positive densities, not a DES calibration. Tests
check the actual C adapters and ordinary-core bridge where available. A
smooth source reconstruction is an explicit model choice; its successful
preparation does not certify the later IA-kernel interpolation or a forecast.
"""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

from des3pcf.inputs import load_tables, reconstructed_sources, save_tables
from des3pcf.inputs import shifted_sources, validate_tables
from des3pcf.inputs import ARRAY_NAMES, SCALAR_NAMES
from des3pcf.mean import physics_model
from des3pcf.pipeline import PhysicalProvider
from des3pcf.accuracy import Accuracy
from test_pipeline import toy_tables


PADDING = "zero_at_observer_and_background_edge"


def source_tables(count=65):
    """Keep the same power grid while preparing a separate source curve."""
    tables, parameters = toy_tables()
    native_z = np.linspace(start=0.05, stop=1.95, num=39)
    native = np.empty(shape=(2, native_z.size))
    native[0] = native_z**2 * np.exp(-(native_z / 0.6)**1.5)
    native[1] = native_z**2 * np.exp(-(native_z / 0.95)**1.5)
    source_z = np.linspace(start=0.0, stop=2.0, num=count)
    values, metadata = reconstructed_sources(
        native_z=native_z, native_sources=native, source_z=source_z,
        shifts=np.zeros(2), reconstruction="pchip_midpoint", padding=PADDING,
    )
    tables["source_z"] = source_z
    tables["sources"] = values
    tables["source_native_z"] = native_z
    tables["source_native"] = native
    return tables, parameters, metadata


def assert_bits(first, second):
    """Require identical float64 values, including signed zeros."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    np.testing.assert_array_equal(x=left.view(np.uint64), y=right.view(np.uint64))


def test_pchip_removes_the_linear_knot_cusp_without_clipping():
    """At a midpoint knot, symmetric changes scale quadratically for C1 PCHIP.

    The same linear midpoint curve instead leaves an absolute-shift term.
    Zero plateaus at both support edges remove a separate normalization
    cusp: PCHIP followed by zero extension is only C0 at an endpoint with
    nonzero slope. This test isolates the interior-knot mechanism; it does
    not assert that every padded source curve is globally differentiable.
    """
    native_z = np.array([0.0, 0.2, 0.5, 1.0, 1.5, 1.8, 2.0])
    native = np.array([[0.0, 0.0, 0.2, 1.0, 0.3, 0.0, 0.0]])
    nodes = np.linspace(start=0.0, stop=2.0, num=101)
    ratios = {}
    for mode, padding in (("linear_midpoint", "zero_outside_native"),
                          ("pchip_midpoint", PADDING)):
        center, _ = reconstructed_sources(
            native_z, native, nodes, [0.0], reconstruction=mode, padding=padding,
        )
        residuals = []
        for step in (0.001, 0.0005):
            minus, _ = reconstructed_sources(
                native_z, native, nodes, [-step], reconstruction=mode, padding=padding,
            )
            plus, _ = reconstructed_sources(
                native_z, native, nodes, [step], reconstruction=mode, padding=padding,
            )
            assert np.all(minus >= 0.0) and np.all(plus >= 0.0)
            residuals.append(plus[0, 50] + minus[0, 50] - 2.0 * center[0, 50])
        ratios[mode] = residuals[1] / residuals[0]
    assert 0.48 < ratios["linear_midpoint"] < 0.52
    assert 0.23 < ratios["pchip_midpoint"] < 0.27


def test_source_shift_sign_area_support_and_owned_originals():
    """Positive shifts increase the mean; every returned density has unit area."""
    native_z = np.linspace(start=0.0, stop=2.0, num=201)
    density = np.exp(-0.5 * ((native_z - 0.8) / 0.08)**2)[None, :]
    density[:, (0, -1)] = 0.0
    original = density.copy()
    nodes = np.linspace(start=0.0, stop=2.0, num=801)
    shifted, record = reconstructed_sources(
        native_z, density, nodes, [0.1], reconstruction="pchip_midpoint", padding=PADDING,
    )
    np.testing.assert_allclose(np.trapz(shifted, x=nodes, axis=1), [1.0], atol=2.0e-15)
    np.testing.assert_allclose(np.trapz(shifted[0] * nodes, x=nodes), 0.9, atol=2.0e-6)
    assert shifted[0, 0] == 0.0
    assert record["source_shifts"] == [0.1]
    assert record["source_padding"] == PADDING
    np.testing.assert_array_equal(x=density, y=original)
    with pytest.raises(ValueError, match="removes sources"):
        reconstructed_sources(native_z, density, nodes, [3.0],
                              reconstruction="pchip_midpoint", padding=PADDING)


@pytest.mark.parametrize("problem", ("padding", "negative", "upper", "irregular"))
def test_reconstruction_refuses_undeclared_or_invalid_source_changes(problem):
    """Do not hide density clipping, lost positive support or a moved read grid."""
    native_z = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    density = np.array([[0.0, 0.2, 1.0, 0.3, 0.0]])
    nodes = np.linspace(start=0.0, stop=2.0, num=33)
    padding = PADDING
    if problem == "padding":
        padding = "zero_outside_native"
    elif problem == "negative":
        density[0, 1] = -0.001
    elif problem == "upper":
        density[0, -1] = 0.1
    else:
        nodes[3] += 0.001
    with pytest.raises(ValueError):
        reconstructed_sources(native_z, density, nodes, [0.0],
                              reconstruction="pchip_midpoint", padding=padding)


def test_independent_source_archive_preserves_native_arrays(tmp_path):
    """The archive retains source details without enlarging a power-table row."""
    tables, _, metadata = source_tables(count=65)
    path = tmp_path / "independent_sources.npz"
    save_tables(path=path, tables=tables, metadata=metadata)
    loaded, record = load_tables(path=path)
    assert loaded["z"].size == 41
    assert loaded["sources"].shape == (2, 65)
    assert loaded["p_nonlinear"].shape == (41, tables["k"].size)
    for name in ("source_z", "source_native_z", "source_native"):
        assert_bits(loaded[name], tables[name])
    assert record["source_reconstruction"] == "pchip_midpoint"
    wrong = dict(tables)
    wrong["source_z"] = np.linspace(start=0.0, stop=2.1, num=65)
    with pytest.raises(ValueError, match="background"):
        validate_tables(tables=wrong)


@pytest.mark.parametrize("adapter", ("production", "notebook"))
def test_explicit_legacy_source_axis_is_bitwise_identical(compiled, adapter):
    """The default constructor and an explicit copy of z retain the old arithmetic."""
    tables, _ = toy_tables()
    original = physics_model(compiled=compiled, tables=tables, kernel_nz=9,
                             table_boost=1, fit_nz=5, adapter=adapter)
    explicit = dict(tables)
    explicit["source_z"] = tables["z"].copy()
    other = physics_model(compiled=compiled, tables=explicit, kernel_nz=9,
                          table_boost=1, fit_nz=5, adapter=adapter)
    for name, values in original.radial().items():
        assert_bits(values, other.radial()[name])


@pytest.mark.parametrize("adapter", ("production", "notebook"))
def test_source_models_own_axes_and_keep_power_unchanged(compiled, adapter):
    """Grow, shrink and restore source tables while retaining older C owners."""
    first, _, _ = source_tables(count=33)
    base = physics_model(compiled=compiled, tables=first, kernel_nz=9,
                         table_boost=1, fit_nz=5, adapter=adapter)
    reference = base.radial()
    wave = np.array([0.01, 0.3, 2.0])
    redshift = np.array([0.0, 0.5, 1.5])
    power = base.power(k=wave, z=redshift, mode=2)
    for count in (65, 17, 33):
        tables, _, _ = source_tables(count=count)
        model = physics_model(compiled=compiled, tables=tables, kernel_nz=9,
                              table_boost=1, fit_nz=5, adapter=adapter)
        assert_bits(power, model.power(k=wave, z=redshift, mode=2))
        if count == 33:
            for name, values in reference.items():
                assert_bits(values, model.radial()[name])
        else:
            assert not np.array_equal(model.radial()["shape"], reference["shape"])
    first["source_z"][:] *= 0.9
    first["sources"][:] *= 2.0
    for name, values in reference.items():
        assert_bits(values, base.radial()[name])
    changed_output = base.radial()
    changed_output["shape"][:] = -9.0
    assert_bits(reference["shape"], base.radial()["shape"])


def test_provider_reconstructs_originals_for_shift_and_resolution_sequences(compiled):
    """A/B/A source changes reuse power but refill the source-weighted model."""
    tables, parameters, metadata = source_tables(count=33)
    provider = PhysicalProvider(
        compiled=compiled, reference_tables=tables, reference_metadata=metadata,
        reference_parameters=parameters, kernel_nz=9, fit_nz=5, table_boost=1,
    )
    first, original, _ = provider.get(parameters=parameters)
    power_tables = provider.cosmology_tables
    reference = first.radial()
    for count, shift in ((65, 0.001), (17, -0.002), (33, 0.0)):
        provider.source_nz = count
        point = dict(parameters)
        point["dz1"] = shift
        model, actual, record = provider.get(parameters=point)
        assert provider.cosmology_tables is power_tables
        expected, _ = reconstructed_sources(
            native_z=tables["source_native_z"], native_sources=tables["source_native"],
            source_z=actual["source_z"], shifts=[shift, 0.0],
            reconstruction="pchip_midpoint", padding=PADDING,
        )
        assert_bits(actual["sources"], expected)
        assert record["source_nz"] == count
        assert_bits(actual["p_nonlinear"], original["p_nonlinear"])
        if count == 33:
            for name, values in reference.items():
                assert_bits(values, model.radial()[name])
    provider.source_nz = True
    with pytest.raises(ValueError, match="source_nz"):
        provider.get(parameters=parameters)


def test_native_storage_does_not_change_legacy_shift_results(compiled):
    """Storing the original samples alone never opts an old model into PCHIP."""
    tables, parameters = toy_tables()
    tables["source_native_z"] = tables["z"].copy()
    tables["source_native"] = tables["sources"].copy()
    provider = PhysicalProvider(
        compiled=compiled, reference_tables=tables, reference_metadata={},
        reference_parameters=parameters, kernel_nz=9, fit_nz=5, table_boost=1,
    )
    point = dict(parameters)
    point["dz1"] = 0.001
    _, actual, _ = provider.get(parameters=point)
    expected = shifted_sources(
        z=tables["z"], sources=tables["sources"], shifts=[0.001, 0],
    )
    assert "source_z" not in actual
    assert_bits(actual["sources"], expected)


def test_source_boost_refines_once_and_restores_without_recomputing_power(compiled):
    """Resolve source counts from one immutable baseline, including global boosts."""
    tables, parameters, metadata = source_tables(count=17)
    provider = PhysicalProvider(
        compiled=compiled, reference_tables=tables, reference_metadata=metadata,
        reference_parameters=parameters, kernel_nz=9, fit_nz=5, table_boost=1,
    )
    first, _, _ = provider.get(parameters=parameters)
    reference = first.radial()
    power_tables = provider.cosmology_tables
    for controls, expected_count in ((Accuracy(source_boost=2.0), 33),
                                     (Accuracy(accuracy_boost=1.5), 25),
                                     (Accuracy(), 17)):
        provider.accuracy = controls
        model, actual, record = provider.get(parameters=parameters)
        repeated, _, _ = provider.get(parameters=parameters)
        assert repeated is model
        assert actual["source_z"].size == expected_count
        assert record["source_nz"] == expected_count
        assert provider.cosmology_tables is power_tables
        assert actual["z"].size == tables["z"].size
        assert_bits(actual["p_nonlinear"], tables["p_nonlinear"])
        if expected_count == 17:
            for name, values in reference.items():
                assert_bits(values, model.radial()[name])
    with pytest.raises(AttributeError, match="baseline"):
        provider.source_interpolation = "linear_midpoint"


def test_independent_source_lensing_measure_has_an_analytic_answer(compiled):
    """For chi proportional to z and n(z)=z/2 on [0,2], g is quadratic.

    Its normalized efficiency is (1-z/2)^2. Source and background grids
    have different lengths; accidentally indexing background arrays with
    source indices changes this answer. All quadrature breakpoints align
    here, so the trapezoid of each linear source integrand is exact.
    """
    tables, _ = toy_tables()
    tables["chi"] = 3000.0 * tables["z"]
    tables["dz_dchi"] = np.full(shape=tables["z"].size, fill_value=1.0 / 3000.0)
    tables["source_z"] = np.linspace(start=0.0, stop=2.0, num=17)
    tables["sources"] = np.repeat(tables["source_z"][None, :] / 2.0, 2, axis=0)
    tables["los_z"] = np.array([0.25, 0.5, 1.0, 1.5, 1.75])
    model = physics_model(compiled=compiled, tables=tables, kernel_nz=8,
                          table_boost=1, fit_nz=5)
    prefactor = 1.5 * (100.0 / 299792.0)**2 * tables["omega_m"]
    expected = prefactor * (1.0 - tables["los_z"] / 2.0)**2
    np.testing.assert_allclose(model.radial()["lens"][0], expected, rtol=3.0e-13)


def test_notebook_constructor_copies_strided_readonly_source_arrays(compiled):
    """The notebook boundary owns both independent axes and density values."""
    tables, _, _ = source_tables(count=33)
    reference = physics_model(compiled=compiled, tables=tables, kernel_nz=9,
                              table_boost=1, fit_nz=5)
    arguments = {}
    for name in ARRAY_NAMES + SCALAR_NAMES:
        arguments[name] = tables[name]
    axes = np.repeat(tables["source_z"], 2)[::2]
    samples = np.repeat(tables["sources"], 2, axis=1)[:, ::2]
    axes.setflags(write=False)
    samples.setflags(write=False)
    arguments["source_z"] = axes
    arguments["sources"] = samples
    arguments["kernel_nz"] = 9
    arguments["table_boost"] = 1
    arguments["fit_nz"] = 5
    copied = compiled.notebook.PhysicsModel(**arguments)
    for name, values in reference.radial().items():
        assert_bits(values, copied.radial()[name])


def test_two_point_bridge_accepts_source_resolution_independent_of_power():
    """The actual ordinary core receives both independent grids without rebinning."""
    from des3pcf.core_two_point import CoreTwoPoint

    directory = Path(__file__).resolve().parents[2] / "des_y3/interface"
    if not (directory / "cosmolike_des_y3_interface.so").is_file():
        pytest.skip("Existing DES Y3 extension is required for the joint-source check.")
    sys.path.insert(0, str(directory))
    core = importlib.import_module("cosmolike_des_y3_interface")
    adapter = CoreTwoPoint(core=core, nbin=2, threads=1)
    outputs = []
    for count in (33, 65, 33):
        tables, parameters, metadata = source_tables(count=count)
        tables["p_linear"] = tables["p_nonlinear"].copy()
        outputs.append(adapter.spectra(
            parameters=parameters, tables=tables, metadata=metadata,
            ia={"a1": 0.3, "a2": 0.0}, ell=np.array([100.0, 300.0]),
            bins=np.array([[0, 0], [0, 1]], dtype=np.int32),
        ))
    for index in range(2):
        assert_bits(outputs[0][index], outputs[2][index])
    assert not np.array_equal(outputs[0][0], outputs[1][0])


def test_measured_edge_padding_keeps_an_endpoint_spike_inside_its_support():
    """A last midpoint spike must not create a tail to the power-table boundary.

    The two policies intentionally produce different source reconstructions.
    This check establishes support and ownership, not histogram mass fidelity.
    """
    native_z = np.array([0.0, 0.1, 0.4, 0.8, 1.0])
    native = np.array([[0.0, 0.2, 0.8, 0.1, 1.0]])
    low = np.array([-0.05, 0.05, 0.25, 0.6, 0.9])
    high = np.array([0.05, 0.25, 0.6, 0.9, 1.1])
    nodes = np.linspace(start=0.0, stop=1.5, num=301)
    padded, record = reconstructed_sources(
        native_z, native, nodes, [0.0], reconstruction="pchip_midpoint",
        padding="zero_at_measured_edges", native_low=low, native_high=high,
    )
    extended, _ = reconstructed_sources(
        native_z, native, nodes, [0.0], reconstruction="pchip_midpoint", padding=PADDING,
    )
    assert record["source_unshifted_support"] == [0.0, 1.1]
    np.testing.assert_array_equal(x=padded[:, nodes >= 1.1], y=0.0)
    assert np.any(extended[:, nodes > 1.1] > 0.0)
    np.testing.assert_allclose(np.trapz(padded, x=nodes, axis=1), [1.0], atol=2.0e-15)
    shifted, _ = reconstructed_sources(
        native_z, native, nodes, [0.02], reconstruction="pchip_midpoint",
        padding="zero_at_measured_edges", native_low=low, native_high=high,
    )
    np.testing.assert_array_equal(x=shifted[:, nodes > 1.12], y=0.0)
    with pytest.raises(ValueError, match="edges"):
        reconstructed_sources(native_z, native, nodes, [0.0],
                              reconstruction="pchip_midpoint",
                              padding="zero_at_measured_edges")


def test_native_edges_survive_fits_and_physical_archive(tmp_path):
    """Keep actual supplied cell edges, including the zero observer cell."""
    from astropy.io import fits
    from des3pcf.survey import read_source_distributions

    tables, _, metadata = source_tables(count=65)
    native_z = np.concatenate((np.zeros(1), tables["source_native_z"]))
    tables["source_native_z"] = native_z
    tables["source_native"] = np.column_stack((np.zeros(2), tables["source_native"]))
    low = native_z - 0.025
    high = native_z + 0.025
    columns = [
        fits.Column(name="Z_MID", format="D", array=native_z),
        fits.Column(name="Z_LOW", format="D", array=low),
        fits.Column(name="Z_HIGH", format="D", array=high),
    ]
    for index, values in enumerate(tables["source_native"]):
        columns.append(fits.Column(name=f"BIN{index + 1}", format="D", array=values))
    path = tmp_path / "sources.fits"
    source_hdu = fits.BinTableHDU.from_columns(columns, name="nz_source")
    fits.HDUList([fits.PrimaryHDU(), source_hdu]).writeto(path)
    z, native, supplied_low, supplied_high = read_source_distributions(
        path=path, nbin=2, with_edges=True,
    )
    assert_bits(z, native_z)
    assert_bits(native, tables["source_native"])
    assert_bits(supplied_low, low)
    assert_bits(supplied_high, high)
    tables["source_native_low"] = supplied_low
    tables["source_native_high"] = supplied_high
    archive = tmp_path / "edges.npz"
    save_tables(path=archive, tables=tables, metadata=metadata)
    restored, _ = load_tables(path=archive)
    assert_bits(restored["source_native_low"], low)
    assert_bits(restored["source_native_high"], high)
