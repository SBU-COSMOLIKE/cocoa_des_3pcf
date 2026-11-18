"""Check which halo coordinates belong to a large-area covariance.

The large-area quadratic halo response is a Fourier integral at zero
center momentum. Displaced shear interpolation and halo-center offsets
therefore cannot alter it. Cubic responses still need radial profiles.
These tests use small controlled arrays, not survey-accuracy settings.
"""

import numpy as np
import pytest

from test_physical_covariance import covariance_arguments, make_covariance_model


@pytest.fixture(scope="module")
def model(compiled):
    """Share the same immutable physical model across the array comparisons."""
    return make_covariance_model(compiled=compiled)


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
@pytest.mark.parametrize("selection", ["joint", "quadratic"])
def test_large_area_profile_lookup_settings_are_irrelevant(
    compiled, model, adapter_name, selection
):
    """Changing an unused profile reader leaves every covariance term exact."""
    adapter = getattr(compiled, adapter_name)
    inputs = covariance_arguments()
    if selection == "quadratic":
        for name in ("row_kind", "row_fields", "theta", "theta_weight"):
            inputs[name] = np.ascontiguousarray(inputs[name][1:])
    expected = adapter.physical_covariance(model=model, inputs=inputs)
    changed = dict(inputs)
    changed["profile_radius"] = np.asarray([0.0, 1.e-6, 0.01, 0.08])
    changed["profile_upsample"] = 3
    actual = adapter.physical_covariance(model=model, inputs=changed)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value.view(np.uint64), reference.view(np.uint64))


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
def test_large_area_quadratic_offsets_are_irrelevant(compiled, model, adapter_name):
    """Quadratic-only rows need no finite halo-offset Hankel quadrature.

    The independent long-mode window integral remains present and unchanged;
    the modified radius/dr here describe halo-center offsets, not that window.
    The comparison includes xi+, xi- and an unequal-radius Map2 row.
    """
    adapter = getattr(compiled, adapter_name)
    inputs = covariance_arguments()
    for name in ("row_kind", "row_fields", "theta", "theta_weight"):
        inputs[name] = np.ascontiguousarray(inputs[name][1:])
    expected = adapter.physical_covariance(model=model, inputs=inputs)
    changed = dict(inputs)
    changed["radius"] = np.asarray([1.e-7, 1.e-4, 0.02])
    changed["dr"] = np.asarray([1.e-7, 2.e-4, 0.01])
    actual = adapter.physical_covariance(model=model, inputs=changed)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value.view(np.uint64), reference.view(np.uint64))
