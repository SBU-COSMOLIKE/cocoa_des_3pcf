"""Keep an existing multipole unchanged when only its SIMD lane changes.

A new output query or a higher retained Legendre order does not change
the integrand of any existing coefficient. The compiler must therefore
use the same rounding for a scalar tail and a vector lane. These checks
isolate that requirement from physical resolution changes: all triangle
nodes, angular weights and interpolation cells remain fixed.
"""

import numpy as np
import pytest

from test_physics_model import compact_grid_settings, physical_model


def prepared_grid(model, adapter, settings):
    """Prepare the same ordered TATT source triplet through either adapter."""
    arguments = {
        "bins": np.asarray([0, 1, 0], dtype=np.int32),
        "parameters": {"a1": 0.4, "a2": 0.7, "bta": 0.3},
        "settings": settings,
    }
    if adapter == "notebook":
        return model.projection_grid_notebook(**arguments)
    return model.projection_grid(**arguments)


def read_multipoles(grid, adapter, ell):
    """Read coefficients without changing the retained projection grid."""
    if adapter == "notebook":
        return grid.multipoles_notebook(ell=ell)
    return grid.multipoles(ell=ell)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("reference_mode", [0, 1])
def test_added_query_keeps_existing_multipoles(
    physical_model, adapter, reference_mode
):
    """Appending one ell moves the third query from tail to vector lane."""
    settings = compact_grid_settings(reference_mode=reference_mode)
    grid = prepared_grid(model=physical_model, adapter=adapter, settings=settings)
    ell = np.asarray([130.0, 280.0, 610.0, 970.0])
    three = read_multipoles(grid=grid, adapter=adapter, ell=ell[:3].copy())
    four = read_multipoles(grid=grid, adapter=adapter, ell=ell)
    np.testing.assert_array_equal(
        three.view(np.uint64), four[:, :3, :3].view(np.uint64)
    )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("reference_mode", [0, 1])
def test_added_order_keeps_existing_multipoles(
    physical_model, adapter, reference_mode
):
    """Preparing L=3 must not change the L=2 angular integral."""
    settings = compact_grid_settings(reference_mode=reference_mode)
    settings["lmax"] = 2
    low = prepared_grid(model=physical_model, adapter=adapter, settings=settings)
    settings["lmax"] = 3
    high = prepared_grid(model=physical_model, adapter=adapter, settings=settings)
    # Four requested nodes keep the interpolation reader entirely in SIMD,
    # isolating the scalar tail in the earlier Legendre preparation.
    ell = np.asarray([130.0, 280.0, 610.0, 970.0])
    low_values = read_multipoles(grid=low, adapter=adapter, ell=ell)
    high_values = read_multipoles(grid=high, adapter=adapter, ell=ell)
    np.testing.assert_array_equal(
        low_values.view(np.uint64), high_values[:3].view(np.uint64)
    )
