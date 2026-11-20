"""Unrequested outputs must not change a retained physical prediction.

Adding an unused source triplet, ell query, Legendre order or aperture
filter moves some outputs between scalar tails and SIMD lanes. The
physical inputs and quadrature are unchanged, so the retained values must
match bitwise. These checks isolate arithmetic grouping from refinement.
"""

from theory.bihalofit.variance import reference_physics_model

import numpy as np
import pytest

from test_physics_model import compact_grid_settings, physical_inputs


def assert_same_bits(actual, expected):
    """Keep numerical failure diagnostics and also distinguish signed zeros."""
    np.testing.assert_array_equal(actual, expected)
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


@pytest.fixture(scope="module")
def lane_model(compiled):
    """Use the same controlled spectra, with short explicit LOS integration."""
    inputs = physical_inputs()
    inputs["los_z"] = np.array([0.15, 0.5, 1.0, 1.8])
    inputs["source_readers"] = False
    return reference_physics_model(factory=compiled.PhysicsModel, **inputs)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("component", [0, 1, 2])
def test_unused_source_triplet_preserves_projected_values(lane_model, adapter, component):
    """The residual is checked alone so a large lensing baseline cannot hide it."""
    settings = compact_grid_settings(reference_mode=0)
    settings["prepare_multipoles"] = 0
    parameters = {
        "a1": 0.5,
        "a2": 0.8,
        "bta": 0.6,
        "ct": 0.2,
    }
    bins = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int32)
    prepare = lane_model.projection_grids
    if adapter == "notebook":
        prepare = lane_model.projection_grids_notebook
    alone = prepare(bins=bins[:1].copy(), parameters=parameters, settings=settings)
    paired = prepare(bins=bins.copy(), parameters=parameters, settings=settings)
    scale = np.geomspace(start=25.0, stop=3000.0, num=50)
    triangles = np.column_stack((scale, 1.23 * scale, 1.48 * scale))
    expected = alone[0].projected(sides=triangles, component=component)
    actual = paired[0].projected(sides=triangles, component=component)
    assert np.any(expected != 0.0)
    assert_same_bits(actual=actual, expected=expected)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_extra_legendre_order_preserves_direct_multipoles(lane_model, adapter):
    """Lmax changes output count, while nmu fixes the same angular quadrature."""
    evaluate = lane_model.multipoles
    if adapter == "notebook":
        evaluate = lane_model.multipoles_notebook
    arguments = {
        "ell": np.array([83.0, 167.0, 331.0, 653.0]),
        "bins": np.array([0, 1, 0], dtype=np.int32),
        "parameters": {
            "a1": 0.5,
            "a2": 0.8,
            "bta": 0.6,
        },
        "nmu": 12,
    }
    three = evaluate(**arguments, lmax=2)
    four = evaluate(**arguments, lmax=3)
    assert np.any(three != 0.0)
    assert_same_bits(actual=four[:3], expected=three)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("source_readers", [False, True])
@pytest.mark.parametrize("query", ["one_to_two", "three_to_four", "mixed_extrapolation"])
def test_unused_ell_preserves_nla_power(compiled, adapter, source_readers, query):
    """A paired extrapolated query must not change its valid companion lane."""
    inputs = physical_inputs()
    inputs["source_readers"] = source_readers
    inputs["los_z"] = np.array([0.15, 0.5, 1.0, 1.8])
    model = reference_physics_model(factory=compiled.PhysicsModel, **inputs)
    evaluate = model.nla_power
    if adapter == "notebook":
        evaluate = model.nla_power_notebook
    ell = np.array([83.0, 167.0])
    count = 1
    if query == "three_to_four":
        ell = np.array([83.0, 167.0, 331.0, 653.0])
        count = 3
    elif query == "mixed_extrapolation":
        ell = np.array([83.0, 1.0e12])
    arguments = {
        "bins": np.array([0, 1], dtype=np.int32),
        "parameters": {"a1": 0.6, "eta1": -1.3},
    }
    alone = evaluate(ell=ell[:count].copy(), **arguments)
    extended = evaluate(ell=ell.copy(), **arguments)
    assert np.all(np.isfinite(extended))
    assert np.any(alone != 0.0)
    assert_same_bits(actual=extended[:count], expected=alone)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_unused_aperture_filter_preserves_full_integral(lane_model, adapter):
    """One extra radius triple changes lanes but no Fourier integration node."""
    evaluate = lane_model.aperture_full
    if adapter == "notebook":
        evaluate = lane_model.aperture_full_notebook
    ell = np.array([83.0, 167.0, 331.0, 653.0, 1259.0])
    phi = (np.arange(6) + 0.5) * (2.0 * np.pi / 6.0)
    filters = np.array([[0.002, 0.003, 0.004], [0.004, 0.003, 0.002]])
    arguments = {
        "ell": ell,
        "dell": np.array([65.0, 123.0, 245.0, 477.0, 917.0]),
        "phi": phi,
        "dphi": np.full(6, 2.0 * np.pi / 6.0),
        "bins": np.array([0, 1, 0], dtype=np.int32),
        "parameters": {
            "a1": 0.5,
            "a2": 0.8,
            "bta": 0.6,
        },
    }
    alone = evaluate(filters=filters[:1].copy(), **arguments)
    extended = evaluate(filters=filters.copy(), **arguments)
    assert np.any(alone != 0.0)
    assert_same_bits(actual=extended[:1], expected=alone)
