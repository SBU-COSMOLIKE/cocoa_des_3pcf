"""Check a uniform interpolation chart with nonuniform physical angles.

The optional coordinate x=log(PSI)+PSI/scale distributes table nodes across
small and large angles. A public log_psi array must still describe physical
angles, while interpolation cells use x. These checks distinguish those two
arrays, compare actual coarse-node multipoles with an independent angular
integral, and alternate coordinate definitions in a retained transform plan.
The controlled spectra come from the existing physical-model tests.

This private draft requires test_residual_scaling.py in the same test folder.
No native calculation has been run while preparing it.
"""

import math

import numpy as np
import pytest
from scipy.special import eval_legendre

from test_grid_transform_reuse import (
    assert_bits,
    old_route,
    plan_and_weights,
    retained_route,
)
from test_residual_scaling import (
    alignment_parameters,
    make_grid,
    make_model,
    read_grid,
    settings_for_test,
)


def mixed_settings(scale, reference_mode=0):
    """Keep physical projection choices fixed and change only the PSI chart."""
    settings = settings_for_test(
        reference_mode=reference_mode,
        scaling=0,
        multipoles=True,
    )
    settings["psi_scale"] = scale
    settings["npsi"] = 17
    return settings


def angular_reference(mu, values, lmax):
    """Integrate each linear B segment against P_L without prepared b_L values.

    A linear polynomial times degree L has degree L+1. The chosen Gauss
    rule therefore integrates that product exactly on each supplied interval.
    Half the absolute integral supplies an error scale that remains finite
    when a signed residual multipole is zero.
    """
    nodes, weights = np.polynomial.legendre.leggauss(deg=(lmax + 3) // 2)
    result = np.zeros(shape=lmax + 1, dtype=np.float64)
    absolute_integral = 0.0
    for interval in range(mu.size - 1):
        width = mu[interval + 1] - mu[interval]
        fraction = 0.5 * (1.0 + nodes)
        coordinate = mu[interval] + width * fraction
        first = values[interval]
        second = values[interval + 1]
        interpolated = first + fraction * (second - first)
        for order in range(lmax + 1):
            polynomial = eval_legendre(order, coordinate)
            result[order] += (
                (order + 0.5) * 0.5 * width
                * np.sum(weights * interpolated * polynomial)
            )
        amplitude = abs(first) + abs(second)
        if first * second < 0.0:
            absolute_integral += 0.5 * width * (
                abs(first) * (abs(first) / amplitude)
                + abs(second) * (abs(second) / amplitude)
            )
        else:
            absolute_integral += 0.5 * width * amplitude
    return result, 0.5 * absolute_integral


def multipoles(grid, adapter, ell, component):
    """Exercise the array conversion of the selected public reader."""
    if adapter == "notebook":
        backing = np.empty(shape=2 * ell.size, dtype=np.float64)
        backing[::2] = ell
        return grid.multipoles_notebook(
            ell=backing[::2],
            component=component,
        )
    return grid.multipoles(ell=ell, component=component)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_mixed_psi_axis_and_coarse_angular_integral(compiled, adapter):
    """Check that physical exports and chart coordinates describe one grid."""
    compiled.set_omp_threads(nthreads=1)
    model = make_model(compiled=compiled, adapter=adapter)
    settings = mixed_settings(scale=0.02)
    grid = make_grid(
        model=model,
        adapter=adapter,
        parameters=alignment_parameters(),
        settings=settings,
    )
    axes = grid.axes()
    logarithms = axes["log_psi"]
    assert np.all(np.diff(logarithms) > 0.0)
    expected_endpoints = np.asarray(
        [
            math.log(math.atan2(settings["ell_min"], settings["ell_max"])),
            math.log(math.pi / 4.0),
        ]
    )
    np.testing.assert_allclose(
        actual=logarithms[[0, -1]],
        desired=expected_endpoints,
        rtol=0.0,
        atol=4.0e-15,
    )
    physical_angles = np.exp(logarithms)
    coordinate = logarithms + physical_angles / settings["psi_scale"]
    spacing = np.diff(coordinate)
    np.testing.assert_allclose(
        actual=spacing,
        desired=spacing[0],
        rtol=3.0e-12,
        atol=3.0e-13,
    )
    assert not np.allclose(np.diff(logarithms), np.diff(logarithms)[0])
    stride = settings["multipole_boost"]
    mu = axes["mu"]
    for radial_index, angular_index in ((2, 5), (4, 13)):
        magnitude = np.exp(axes["log_ell"][stride * radial_index])
        angle = np.exp(logarithms[stride * angular_index])
        first = magnitude * np.cos(angle)
        second = magnitude * np.sin(angle)
        for ordered in ((first, second), (second, first)):
            ell = np.asarray(ordered, dtype=np.float64)
            closing = np.sqrt(
                (ell[0] - ell[1])**2
                + 2.0 * ell[0] * ell[1] * (1.0 - mu)
            )
            sides = np.column_stack(
                (
                    np.full(shape=mu.size, fill_value=ell[0]),
                    np.full(shape=mu.size, fill_value=ell[1]),
                    closing,
                )
            )
            for component in (1, 2):
                sampled = read_grid(
                    grid=grid,
                    adapter=adapter,
                    sides=sides,
                    component=component,
                )
                expected, scale = angular_reference(
                    mu=mu,
                    values=sampled,
                    lmax=settings["lmax"],
                )
                assert scale > 0.0
                actual = multipoles(
                    grid=grid,
                    adapter=adapter,
                    ell=ell,
                    component=component,
                )[:, 0, 1]
                np.testing.assert_allclose(
                    actual=actual / scale,
                    desired=expected / scale,
                    rtol=5.0e-10,
                    atol=5.0e-10,
                )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("reference_mode", [0, 1])
def test_mixed_psi_default_and_reference_bits(
    compiled, adapter, reference_mode,
):
    """Keep default bits and leave reference grids independent of the scale."""
    compiled.set_omp_threads(nthreads=1)
    model = make_model(compiled=compiled, adapter=adapter)
    explicit_settings = mixed_settings(scale=0.0, reference_mode=reference_mode)
    omitted_settings = dict(explicit_settings)
    del omitted_settings["psi_scale"]
    alternate_settings = dict(explicit_settings)
    if reference_mode:
        alternate_settings["psi_scale"] = 0.02
    settings_bank = (
        explicit_settings,
        omitted_settings,
        alternate_settings,
    )
    grids = []
    for settings in settings_bank:
        grids.append(make_grid(
            model=model,
            adapter=adapter,
            parameters=alignment_parameters(),
            settings=settings,
        ))
    ell = np.asarray([37.0, 190.0, 930.0], dtype=np.float64)
    first_axes = grids[0].axes()
    for grid in grids[1:]:
        current_axes = grid.axes()
        for name in first_axes:
            assert_bits(
                actual=current_axes[name],
                expected=first_axes[name],
                context=f"Default/reference physical axis changed: {name}",
            )
        for component in (0, 1, 2):
            expected = multipoles(
                grid=grids[0], adapter=adapter, ell=ell, component=component,
            )
            actual = multipoles(
                grid=grid, adapter=adapter, ell=ell, component=component,
            )
            assert_bits(
                actual=actual,
                expected=expected,
                context=f"Default/reference changed component {component}",
            )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_mixed_psi_retained_query_equal_physical_axes(compiled, adapter):
    """A new coordinate needs new weights even with identical physical axes.

    With two coarse angular nodes and no dense upsampling, both charts expose
    exactly the same physical endpoints and sample the same b_L values. Their
    interpolation weights between those endpoints differ. Comparing physical
    axes alone would wrongly reuse the first chart's weights for the second.
    A/B/A calls test this actual collision at each requested worker count.
    """
    compiled.set_omp_threads(nthreads=1)
    model = make_model(compiled=compiled, adapter=adapter)
    grids = []
    references = []
    for scale in (0.0, 0.02):
        settings = mixed_settings(scale=scale)
        settings["npsi"] = 2
        settings["multipole_boost"] = 1
        grid = make_grid(
            model=model,
            adapter=adapter,
            parameters=alignment_parameters(),
            settings=settings,
        )
        grids.append(grid)
        references.append(old_route(compiled=compiled, grid=grid))
    assert_bits(
        actual=grids[1].axes()["log_psi"],
        expected=grids[0].axes()["log_psi"],
        context="Two-node physical endpoints must coincide in both charts",
    )
    assert not np.array_equal(references[0], references[1])
    try:
        for workers in (1, 2, 4):
            compiled.set_omp_threads(nthreads=workers)
            settings = mixed_settings(scale=0.02)
            settings["npsi"] = 2
            settings["multipole_boost"] = 1
            threaded = make_grid(
                model=model,
                adapter=adapter,
                parameters=alignment_parameters(),
                settings=settings,
            )
            plan, _, weights = plan_and_weights(
                compiled=compiled,
                workers=workers,
            )
            sequence = (grids[0], threaded, grids[0], grids[1], grids[0])
            expected_indices = (0, 1, 0, 1, 0)
            for grid, index in zip(sequence, expected_indices):
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
                    context=f"Mixed PSI restoration at {workers} workers",
                )
    finally:
        compiled.set_omp_threads(nthreads=1)
