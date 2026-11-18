"""Check the optional PUV shape coordinate and retained-query ownership.

The production candidate concentrates sampling near squeezed triangles.
These tests evaluate actual well-conditioned physical coarse knots. Axis
endpoints are checked as coordinate definitions, not extreme-triangle accuracy. Direct radial projection supplies the reference. The tests do
not set an accuracy requirement for a survey mean between those knots.
"""

import numpy as np
import pytest

from test_grid_transform_reuse import (
    assert_bits,
    old_route,
    plan_and_weights,
    retained_route,
)
from test_residual_scaling import (
    alignment_parameters,
    direct_values,
    gravity_scale,
    make_grid,
    make_model,
    read_grid,
    settings_for_test,
)


CANDIDATE_STRETCH = 10.106


def shape_settings(stretch, multipoles=False):
    """Use a small table spanning the same Fourier bounds as the diagnosis."""
    settings = settings_for_test(
        reference_mode=0,
        scaling=1,
        multipoles=multipoles,
    )
    settings.update(
        ell_min=0.001,
        ell_max=3.0e5,
        np=7,
        nshape=17,
    )
    if stretch is not None:
        settings["shape_stretch"] = stretch
    return settings


def interior_knots(grid, settings):
    """Choose physical original knots with a bounded inverse-chart derivative.

    Each selected ratio has |dt/du|<5 in this chart. A represented ratio's
    one-ulp uncertainty therefore remains of order machine precision in t,
    unlike the extreme-edge diagnostic retained separately. The third side
    is also bounded away from zero and one half, keeping physical closure
    independent of a machine-scale nearly flattened triangle.
    """
    axes = grid.axes()
    stride = settings["table_boost"]
    shape = axes["pu"][::stride]
    perimeter = np.exp(axes["log_p"][::stride])
    middle = shape.size // 2
    indices = (
        (middle, middle + 1),
        (middle + 1, middle),
        (middle + 1, middle + 1),
    )
    minimum = shape[0]
    stretch = settings["shape_stretch"]
    tangent = np.tanh(stretch)
    triangles = []
    for first, second in indices:
        fractions = np.asarray(
            [shape[first], shape[second], 1.0 - shape[first] - shape[second]]
        )
        assert np.all(fractions > 0.01)
        assert np.all(fractions < 0.49)
        normalized = 2.0 * (fractions[:2] - minimum) / (0.5 - minimum) - 1.0
        sensitivity = 2.0 * tangent / (
            stretch * (0.5 - minimum) * (1.0 - (tangent * normalized) ** 2)
        )
        assert np.all(sensitivity < 5.0)
        triangles.append(perimeter[perimeter.size // 2] * fractions)
    return np.asarray(triangles, dtype=np.float64)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_candidate_shape_axes_and_interior_physical_knots(compiled, adapter):
    """The stretched axis is ordered and its interior knots retain physical values.

    The absolute error is scaled by positive GGG, which remains meaningful
    if a residual is close to zero. The selected coordinates have bounded
    inverse sensitivity. This is a
    knot-reconstruction check; it neither constrains extreme-edge triangles
    nor sets an accuracy tolerance for interpolated survey predictions.
    """
    compiled.set_omp_threads(nthreads=1)
    model = make_model(compiled=compiled, adapter=adapter)
    settings = shape_settings(stretch=CANDIDATE_STRETCH)
    parameters = alignment_parameters()
    grid = make_grid(
        model=model,
        adapter=adapter,
        parameters=parameters,
        settings=settings,
    )
    axes = grid.axes()
    minimum = min(1.0e-6, 0.25 * settings["ell_min"] / settings["ell_max"])
    assert axes["pu"][0] == minimum
    assert axes["pu"][-1] == 0.5
    assert np.all(np.isfinite(axes["pu"]))
    assert np.all(np.diff(axes["pu"]) > 0.0)
    sides = interior_knots(grid=grid, settings=settings)
    expected = direct_values(
        model=model,
        sides=sides,
        parameters=parameters,
        component=2,
    )
    actual = read_grid(grid=grid, adapter=adapter, sides=sides, component=2)
    scale = gravity_scale(model=model, sides=sides)
    assert np.any(expected != 0.0), "The interior TATT residual was not reached"
    np.testing.assert_allclose(
        actual=actual / scale,
        desired=expected / scale,
        rtol=2.0e-8,
        atol=2.0e-8,
    )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_omitted_shape_stretch_matches_explicit_default(compiled, adapter):
    """Omitting the new keyword selects exactly the current stretch-three path.

    This compares two current constructor choices. It does not assert that
    the new exact-endpoint handling has every bit of an older executable's
    inverse hyperbolic-tangent round trip.
    """
    compiled.set_omp_threads(nthreads=1)
    model = make_model(compiled=compiled, adapter=adapter)
    parameters = alignment_parameters()
    grids = []
    for stretch in (None, 3.0):
        settings = shape_settings(stretch=stretch, multipoles=True)
        grids.append(make_grid(
            model=model,
            adapter=adapter,
            parameters=parameters,
            settings=settings,
        ))
    first_axes = grids[0].axes()
    second_axes = grids[1].axes()
    for name in first_axes:
        assert_bits(
            actual=first_axes[name],
            expected=second_axes[name],
            context=f"Omitted stretch changed the {name} axis",
        )
    sides = np.asarray(
        [
            [190.0, 280.0, 360.0],
            [430.0, 170.0, 310.0],
        ]
    )
    for component in (0, 1, 2):
        assert_bits(
            actual=read_grid(
                grid=grids[0], adapter=adapter, sides=sides, component=component,
            ),
            expected=read_grid(
                grid=grids[1], adapter=adapter, sides=sides, component=component,
            ),
            context=f"Omitted stretch changed projected component {component}",
        )
    assert_bits(
        actual=old_route(compiled=compiled, grid=grids[0]),
        expected=old_route(compiled=compiled, grid=grids[1]),
        context="Omitted stretch changed prepared multipoles or aperture output",
    )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_shape_stretch_workers_and_retained_query(compiled, adapter):
    """A reused multipole query must read each changed shape table's values.

    The multipole interpolation axes stay fixed when only shape_stretch
    changes. Reusing geometric indices is correct; retaining the former
    physical multipoles would give the wrong result. Independent fresh-plan
    references distinguish those two cases, including a return to default.
    """
    compiled.set_omp_threads(nthreads=1)
    model = make_model(compiled=compiled, adapter=adapter)
    parameters = alignment_parameters()
    grids = []
    references = []
    for stretch in (3.0, CANDIDATE_STRETCH):
        settings = shape_settings(stretch=stretch, multipoles=True)
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
            settings = shape_settings(stretch=CANDIDATE_STRETCH, multipoles=True)
            threaded = make_grid(
                model=model,
                adapter=adapter,
                parameters=parameters,
                settings=settings,
            )
            plan, _, weights = plan_and_weights(compiled=compiled, workers=workers)
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
                    context=f"Changed-stretch table restoration at {workers} workers",
                )
    finally:
        compiled.set_omp_threads(nthreads=1)
