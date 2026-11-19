"""Check the optional Python bispectrum through the physical-model provider.

These small positive power spectra and source curves test input selection,
model ownership and numerical-control changes. They do not establish DES
grid convergence. The observable comparison and its covariance-weighted
accuracy test use separately recorded grids and fixed scientific inputs.
"""

import numpy as np
import pytest

from des3pcf.accuracy import Accuracy
from des3pcf.bihalofit import BaselineBispectrum
from des3pcf.inputs import load_tables, save_tables
from des3pcf.mean import physics_model
from des3pcf.pipeline import PhysicalProvider
from test_pipeline import toy_tables


def supplied_tables():
    """Return the controlled model with its explicit matter-density fraction.

    Returns:
      tables and cosmological parameters. The background is flat, contains
      pressureless matter and a cosmological constant, and has no radiation.
      This test definition is independent of the CAMB provider route.
    """
    tables, parameters = toy_tables()
    matter = tables["omega_m"] * (1.0 + tables["z"])**3
    tables["omega_m_z"] = matter / (matter + tables["omega_de"])
    return tables, parameters


def grid():
    """Return small uniform axes covering the triangles queried in these tests.

    Returns:
      the four stored coordinates and a dense-reader interval ratio of one.
      The coarse counts deliberately test refinement without a large table.
    """
    return {
        "z": np.linspace(start=0.0, stop=2.0, num=4),
        "logk": np.linspace(start=-10.0, stop=10.0, num=6),
        "logu": np.linspace(start=-30.0, stop=0.0, num=5),
        "v": np.linspace(start=0.0, stop=1.0, num=4),
        "boost": 1,
    }


def provider_arguments(compiled):
    """Return an explicit fixed-table configuration with modest fit counts.

    Arguments:
      compiled = the extension fixture whose production factory is exercised.
    Returns:
      keyword arguments for PhysicalProvider, including the supplied selector.
    """
    tables, parameters = supplied_tables()
    return {
        "compiled": compiled,
        "reference_tables": tables,
        "reference_metadata": {},
        "reference_parameters": parameters,
        "cosmology_mode": "fixed_tables",
        "kernel_nz": 33,
        "fit_nz": 8,
        "table_boost": 2,
        "bispectrum_source": "supplied",
        "bispectrum_table": grid(),
    }


def test_supplied_background_survives_physical_archive(tmp_path):
    """Saving and loading cannot drop a prescribed background-density array."""
    tables, _ = supplied_tables()
    path = tmp_path / "physical.npz"
    save_tables(path=path, tables=tables, metadata={"test": "explicit background"})
    loaded, _ = load_tables(path=path)
    np.testing.assert_array_equal(loaded["omega_m_z"], tables["omega_m_z"])


def test_omitted_selector_preserves_internal_values(compiled):
    """An unused density array and an explicit internal selector move no bits."""
    arguments = provider_arguments(compiled=compiled)
    arguments.pop("bispectrum_table")
    arguments.pop("bispectrum_source")
    implicit = PhysicalProvider(**arguments)
    arguments["bispectrum_source"] = "internal"
    explicit = PhysicalProvider(**arguments)
    parameters = arguments["reference_parameters"]
    first, _, _ = implicit.get(parameters=parameters)
    second, _, _ = explicit.get(parameters=parameters)
    sides = np.array([[0.1, 0.12, 0.15], [1.0, 1.2, 1.5]])
    redshifts = np.array([0.2, 1.2])
    np.testing.assert_array_equal(
        first.matter(sides=sides, redshifts=redshifts),
        second.matter(sides=sides, redshifts=redshifts),
    )
    assert implicit._active_baseline is None
    assert explicit._active_baseline is None


def test_missing_density_fraction_is_refused_before_model(compiled):
    """The supplied selector cannot silently reconstruct a background in C."""
    arguments = provider_arguments(compiled=compiled)
    del arguments["reference_tables"]["omega_m_z"]
    with pytest.raises(ValueError, match="requires provider omega_m_z"):
        PhysicalProvider(**arguments)


def test_source_and_accuracy_changes_preserve_baseline_ownership(compiled):
    """Sources reuse B; numerical refinement rebuilds B and retains old owners.

    A source shift changes lensing kernels without changing the underlying
    matter bispectrum. Returning to the original source must reproduce its
    original model values. A boost doubles coarse intervals once; it leaves
    the separate dense-reader ratio unchanged.
    """
    arguments = provider_arguments(compiled=compiled)
    provider = PhysicalProvider(**arguments)
    parameters = arguments["reference_parameters"]
    initial, _, _ = provider.get(parameters=parameters)
    baseline = provider._active_baseline
    sides = np.array([[0.1, 0.12, 0.15], [1.0, 1.2, 1.5]])
    redshifts = np.array([0.2, 1.2])
    original = initial.matter(sides=sides, redshifts=redshifts)
    angular_sides = np.array([[100.0, 120.0, 150.0]])
    bins = np.array([0, 1, 0], dtype=np.int32)
    original_projection = initial.projected(
        sides=angular_sides, bins=bins, parameters={"a1": 0.0},
    )
    shifted = dict(parameters)
    shifted["dz1"] = 0.01
    changed, _, _ = provider.get(parameters=shifted)
    assert changed is not initial
    assert provider._active_baseline is baseline
    changed_projection = changed.projected(
        sides=angular_sides, bins=bins, parameters={"a1": 0.0},
    )
    assert not np.array_equal(changed_projection, original_projection)
    returned, _, _ = provider.get(parameters=parameters)
    assert provider._active_baseline is baseline
    np.testing.assert_array_equal(
        returned.matter(sides=sides, redshifts=redshifts), original,
    )
    np.testing.assert_array_equal(
        returned.projected(sides=angular_sides, bins=bins, parameters={"a1": 0.0}),
        original_projection,
    )
    provider.accuracy = Accuracy(accuracy_boost=2.0)
    provider.get(parameters=parameters)
    refined = provider._active_baseline
    assert refined is not baseline
    for name in ("z", "logk", "logu", "v"):
        assert refined.native[name].size == 2 * (baseline.native[name].size - 1) + 1
    assert refined.native["boost"] == baseline.native["boost"]
    assert refined.record()["controls"]["fit_nz"] == 15
    np.testing.assert_array_equal(
        initial.matter(sides=sides, redshifts=redshifts), original,
    )
    provider.accuracy = Accuracy()
    restored, _, _ = provider.get(parameters=parameters)
    np.testing.assert_array_equal(
        restored.matter(sides=sides, redshifts=redshifts), original,
    )


@pytest.mark.parametrize("changed", ["z", "omega_m_z"])
def test_direct_factory_refuses_mismatched_recorded_background(compiled, changed):
    """A valid table shape does not establish that B and the background match."""
    tables, _ = supplied_tables()
    axes = grid()
    shape = tuple(axes[name].size for name in ("z", "logk", "logu", "v"))
    background_z = tables["z"].copy()
    density = tables["omega_m_z"].copy()
    if changed == "z":
        background_z[1] += 0.001
    else:
        density[1] += 0.001
    baseline = BaselineBispectrum(
        **axes,
        log_bispectrum=np.zeros(shape=shape),
        metadata={"test": "constant positive B"},
        background_z=background_z,
        background_omega_m_z=density,
    )
    with pytest.raises(ValueError, match="different background"):
        physics_model(compiled=compiled, tables=tables, baseline=baseline)
