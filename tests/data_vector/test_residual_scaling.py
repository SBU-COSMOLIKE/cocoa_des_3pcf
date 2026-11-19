"""Check a signed residual after removing its positive radial amplitude.

The projected TATT correction can change sign while spanning many orders of
magnitude across Fourier scale. The optional representation divides it by
positive GGG at equal sides with the same perimeter. This file checks values
at the original physical PUV knots, a true residual zero, and ownership when
one retained transform plan alternates between interpolation conventions.
The expected values come from direct C line-of-sight evaluation, not another
prepared table. The model uses the established smooth artificial spectra;
no survey data or Boltzmann calculation enters these tests.

The wide-range comparison uses direct physical values as its reference;
the tolerance is an interpolation check for these controlled spectra.
"""

from des3pcf.variance import reference_physics_model

import numpy as np
import pytest

from test_grid_transform_reuse import (
    assert_bits,
    old_route,
    plan_and_weights,
    retained_route,
)
from test_physics_model import compact_grid_settings, physical_inputs


def alignment_parameters(residual=True):
    """Choose modest IA coefficients and a fixed nonlinear power convention."""
    return {
        "a1": 0.4,
        "eta1": 0.2,
        "a2": 0.7 if residual else 0.0,
        "eta2": -0.3,
        "bta": 0.6 if residual else 0.0,
        "ct": 0.2 if residual else 0.0,
        "power_mode": 2,
    }


def source_bins():
    """Keep unequal ordered bins so a scalar side sort cannot hide a change."""
    return np.asarray([0, 1, 0], dtype=np.int32)


def make_model(compiled, adapter):
    """Build the same immutable physical model through either public adapter."""
    inputs = physical_inputs()
    if adapter == "notebook":
        inputs["sources"] = np.asfortranarray(inputs["sources"])
        inputs["p_nonlinear"] = np.asfortranarray(inputs["p_nonlinear"])
        return reference_physics_model(factory=compiled.notebook.PhysicsModel, **inputs)
    return reference_physics_model(factory=compiled.production.PhysicsModel, **inputs)


def settings_for_test(reference_mode, scaling, multipoles=False):
    """Keep geometry fixed while choosing the signed residual representation."""
    settings = compact_grid_settings(reference_mode=reference_mode)
    settings["residual_scaling"] = scaling
    settings["prepare_multipoles"] = int(multipoles)
    settings["lmax"] = 3
    return settings


def make_grid(model, adapter, parameters, settings):
    """Call the borrowing or copying grid constructor with identical physics."""
    bins = source_bins()
    if adapter == "notebook":
        return model.projection_grid_notebook(
            bins=bins,
            parameters=parameters,
            settings=settings,
        )
    return model.projection_grid(
        bins=bins,
        parameters=parameters,
        settings=settings,
    )


def read_grid(grid, adapter, sides, component):
    """Read independent output arrays; notebook triangles use Fortran layout."""
    if adapter == "notebook":
        return grid.projected_notebook(
            sides=np.asfortranarray(sides),
            component=component,
        )
    return grid.projected(sides=sides, component=component)


def direct_values(model, sides, parameters, component):
    """Use physical radial quadrature without projected-table interpolation."""
    return model.projected(
        sides=sides,
        bins=source_bins(),
        parameters=parameters,
        component=component,
    )


def gravity_scale(model, sides):
    """Use positive GGG to measure absolute errors at signed residual zeros."""
    parameters = alignment_parameters(residual=False)
    parameters["a1"] = 0.0
    values = direct_values(
        model=model,
        sides=sides,
        parameters=parameters,
        component=1,
    )
    assert np.all(values > 0.0), "The positive GGG error scale was not reached"
    return values


def physical_puv_knots(grid, settings):
    """Select original knots strictly inside the physical triangle region.

    A PUV triple has sides p*(u,v,1-u-v). Physical closure requires each
    fraction below one half and u+v above one half. Selecting upper-middle
    shape knots satisfies these conditions without approaching the zero-side
    corner. Dense upsampling retains each original knot at a known stride.
    """
    stride = settings["table_boost"]
    if settings["reference_mode"]:
        stride = 1
    axes = grid.axes()
    perimeter = np.exp(axes["log_p"][::stride])
    shape = axes["pu"][::stride]
    middle = shape.size // 2
    specifications = (
        (1, middle, middle + 1),
        (perimeter.size // 2, middle + 1, middle + 2),
        (perimeter.size - 2, middle + 2, middle),
    )
    triangles = []
    for scale, first, second in specifications:
        fractions = np.asarray(
            [shape[first], shape[second], 1.0 - shape[first] - shape[second]]
        )
        assert np.all(fractions > 0.0)
        assert np.all(fractions < 0.5)
        triangles.append(perimeter[scale] * fractions)
    return np.asarray(triangles, dtype=np.float64)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("reference_mode", [0, 1])
def test_residual_scaling_preserves_physical_coarse_knots(
    compiled, adapter, reference_mode,
):
    """A change of representation reproduces the original sampled values."""
    compiled.set_omp_threads(nthreads=1)
    model = make_model(compiled=compiled, adapter=adapter)
    settings = settings_for_test(reference_mode=reference_mode, scaling=1)
    parameters = alignment_parameters()
    grid = make_grid(
        model=model,
        adapter=adapter,
        parameters=parameters,
        settings=settings,
    )
    sides = physical_puv_knots(grid=grid, settings=settings)
    expected = direct_values(
        model=model,
        sides=sides,
        parameters=parameters,
        component=2,
    )
    actual = read_grid(grid=grid, adapter=adapter, sides=sides, component=2)
    scale = gravity_scale(model=model, sides=sides)
    assert np.any(expected != 0.0), "The TATT residual was not reached"
    np.testing.assert_allclose(
        actual=actual / scale,
        desired=expected / scale,
        rtol=5.0e-11,
        atol=5.0e-11,
    )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_residual_scaling_keeps_both_signs_and_a_zero(compiled, adapter):
    """The scale stays positive while the physical residual passes through zero.

    At this perturbative order the residual is linear in A2 for fixed A1.
    Two direct evaluations determine an amplitude at which one physical
    knot has zero residual. Values on either side must have opposite signs;
    checking only a small numerical answer would not prove a crossing.
    """
    compiled.set_omp_threads(nthreads=1)
    model = make_model(compiled=compiled, adapter=adapter)
    settings = settings_for_test(reference_mode=0, scaling=1)
    parameters = alignment_parameters()
    initial = make_grid(
        model=model,
        adapter=adapter,
        parameters=parameters,
        settings=settings,
    )
    sides = physical_puv_knots(grid=initial, settings=settings)[1:2].copy()
    parameters["a2"] = 0.0
    at_zero = direct_values(
        model=model, sides=sides, parameters=parameters, component=2,
    )[0]
    parameters["a2"] = 1.0
    at_one = direct_values(
        model=model, sides=sides, parameters=parameters, component=2,
    )[0]
    assert at_one != at_zero, "A2 does not change the selected residual"
    crossing = -at_zero / (at_one - at_zero)
    displacement = 0.05 * max(1.0, abs(crossing))
    amplitudes = (
        crossing - displacement,
        crossing,
        crossing + displacement,
    )
    scale = gravity_scale(model=model, sides=sides)
    references = []
    for amplitude in amplitudes:
        parameters["a2"] = amplitude
        expected = direct_values(
            model=model, sides=sides, parameters=parameters, component=2,
        )
        references.append(expected[0])
        grid = make_grid(
            model=model,
            adapter=adapter,
            parameters=parameters,
            settings=settings,
        )
        actual = read_grid(grid=grid, adapter=adapter, sides=sides, component=2)
        np.testing.assert_allclose(
            actual=actual / scale,
            desired=expected / scale,
            rtol=5.0e-11,
            atol=5.0e-11,
        )
    assert references[0] * references[2] < 0.0
    assert abs(references[1] / scale[0]) < 1.0e-11


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("reference_mode", [0, 1])
def test_residual_scaling_pure_nla_bits(compiled, adapter, reference_mode):
    """Pure NLA allocates no residual scale and must keep every output bit."""
    compiled.set_omp_threads(nthreads=1)
    model = make_model(compiled=compiled, adapter=adapter)
    parameters = alignment_parameters(residual=False)
    grids = []
    for scaling in (0, 1):
        settings = settings_for_test(
            reference_mode=reference_mode,
            scaling=scaling,
            multipoles=True,
        )
        grids.append(make_grid(
            model=model,
            adapter=adapter,
            parameters=parameters,
            settings=settings,
        ))
    sides = np.asarray(
        [[190.0, 280.0, 360.0], [430.0, 170.0, 310.0]],
        dtype=np.float64,
    )
    ell = np.asarray([130.0, 280.0, 610.0], dtype=np.float64)
    for component in (0, 1, 2):
        ordinary = read_grid(
            grid=grids[0], adapter=adapter, sides=sides, component=component,
        )
        scaled = read_grid(
            grid=grids[1], adapter=adapter, sides=sides, component=component,
        )
        assert_bits(
            actual=scaled,
            expected=ordinary,
            context=f"Pure NLA changed projected component {component}",
        )
        if adapter == "notebook":
            expected = grids[0].multipoles_notebook(
                ell=ell, component=component,
            )
            actual = grids[1].multipoles_notebook(
                ell=ell, component=component,
            )
        else:
            expected = grids[0].multipoles(ell=ell, component=component)
            actual = grids[1].multipoles(ell=ell, component=component)
        assert_bits(
            actual=actual,
            expected=expected,
            context=f"Pure NLA changed multipole component {component}",
        )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_residual_scaling_workers_and_retained_query(compiled, adapter):
    """A retained geometric map reads fresh values and restores prior outputs.

    The raw and scaled tables share geometry, so a transform plan may keep
    its interpolation indices. Their physical arrays differ and must never
    be cached in that map. A/B/A calls at one, two and four workers check
    both this distinction and disjoint writes during grid preparation.
    """
    compiled.set_omp_threads(nthreads=1)
    model = make_model(compiled=compiled, adapter=adapter)
    parameters = alignment_parameters()
    grids = []
    references = []
    for scaling in (0, 1):
        settings = settings_for_test(
            reference_mode=0,
            scaling=scaling,
            multipoles=True,
        )
        grid = make_grid(
            model=model,
            adapter=adapter,
            parameters=parameters,
            settings=settings,
        )
        grids.append(grid)
        references.append(old_route(compiled=compiled, grid=grid))
    assert not np.array_equal(references[0], references[1])
    try:
        for workers in (1, 2, 4):
            compiled.set_omp_threads(nthreads=workers)
            settings = settings_for_test(
                reference_mode=0,
                scaling=1,
                multipoles=True,
            )
            threaded = make_grid(
                model=model,
                adapter=adapter,
                parameters=parameters,
                settings=settings,
            )
            plan, _, weights = plan_and_weights(
                compiled=compiled,
                workers=workers,
            )
            sequence = (grids[0], threaded, grids[0], grids[1], grids[0])
            reference_indices = (0, 1, 0, 1, 0)
            for grid, index in zip(sequence, reference_indices):
                actual = retained_route(
                    compiled=compiled,
                    adapter=adapter,
                    plan=plan,
                    grid=grid,
                    weights=weights,
                )
                assert_bits(
                    actual=actual,
                    expected=references[index],
                    context=f"Residual table restoration at {workers} workers",
                )
    finally:
        compiled.set_omp_threads(nthreads=1)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_residual_scaling_controls_wide_perimeter_tail(compiled, adapter):
    """Compare well-shaped high-p triangles with direct projected physics.

    All triangles stay away from squeezed and flattened boundaries. A huge
    signed error here is therefore not attributed to a poorly resolved
    corner. Only the perimeter count changes in the refinement pair. The
    positive GGG scale remains meaningful when the residual crosses zero.
    These proposed bounds await the first controlled execution of this test.
    """
    compiled.set_omp_threads(nthreads=1)
    model = make_model(compiled=compiled, adapter=adapter)
    parameters = alignment_parameters()
    configurations = (
        (0, 25),
        (1, 25),
        (1, 49),
    )
    grids = []
    for scaling, perimeter_count in configurations:
        settings = settings_for_test(reference_mode=0, scaling=scaling)
        settings.update(
            ell_min=0.001,
            ell_max=3.0e5,
            np=perimeter_count,
            nshape=33,
            table_boost=2,
        )
        grid = make_grid(
            model=model,
            adapter=adapter,
            parameters=parameters,
            settings=settings,
        )
        grids.append(grid)
    # Select original physical shape knots, but query between radial knots.
    # This removes shape interpolation from a test of the radial correction.
    axes = grids[0].axes()
    shape = axes["pu"][::2]
    templates = []
    target_pairs = (
        (1.0 / 3.0, 1.0 / 3.0),
        (0.30, 0.36),
        (0.27, 0.39),
    )
    for first_target, second_target in target_pairs:
        first_index = np.argmin(np.abs(shape - first_target))
        second_index = np.argmin(np.abs(shape - second_target))
        first = shape[first_index]
        second = shape[second_index]
        templates.append([first, second, 1.0 - first - second])
    templates = np.asarray(templates, dtype=np.float64)
    assert np.all(templates > 0.2)
    assert np.all(templates < 0.45)
    perimeters = np.asarray(
        [5100.0, 21000.0, 69000.0, 240000.0], dtype=np.float64,
    )
    sides = (perimeters[:, None, None] * templates[None, :, :]).reshape(-1, 3)
    expected = direct_values(
        model=model,
        sides=sides,
        parameters=parameters,
        component=2,
    )
    gravity = gravity_scale(model=model, sides=sides)
    errors = []
    for grid in grids:
        np.testing.assert_array_equal(grid.axes()["pu"], axes["pu"])
        actual = read_grid(grid=grid, adapter=adapter, sides=sides, component=2)
        assert np.all(np.isfinite(actual))
        errors.append(float(np.max(np.abs((actual - expected) / gravity))))
    message = f"Raw/scaled/refined maximum GGG-scaled errors: {errors}"
    assert errors[0] > 1.0e-3, message
    assert errors[1] < errors[0] / 4.0, message
    assert errors[2] < errors[1], message
    assert errors[2] < 0.1, message
