"""Check signed NLA tables against direct radial projections.

The controlled source model keeps only a short, high-redshift part of the
line-of-sight integral. Positive A1 can then make the local effective shape
kernel negative, so its cubic product gives a negative projected baseline.
This deliberately tests a numerical representation outside the old positive
logarithm's domain. It is not a calibrated survey source distribution.

No expected physical value comes from a prepared interpolation table. Direct
projection supplies the reference, and positive GGG supplies the error scale
at a baseline zero. The separate TATT residual is checked without claiming
that its coarse interpolation is scientifically accurate.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from test_grid_transform_reuse import (
    assert_bits,
    old_route,
    plan_and_weights,
    retained_route,
)
from test_physics_model import compact_grid_settings, physical_inputs


BINS = np.asarray([0, 0, 0], dtype=np.int32)


def signed_inputs():
    """Restrict the radial test integral where a strong NLA kernel is negative."""
    inputs = physical_inputs()
    inputs["los_z"] = np.asarray([1.0, 1.2, 1.4, 1.6])
    inputs["source_readers"] = False
    return inputs


def parameters(amplitude, residual=False):
    """Keep the power convention fixed while changing the effective shape field."""
    return {
        "a1": amplitude,
        "eta1": 5.0,
        "a2": 0.3 if residual else 0.0,
        "eta2": 0.0,
        "bta": 0.4 if residual else 0.0,
        "ct": 0.1 if residual else 0.0,
        "power_mode": 1,
    }


def grid_settings(reference_mode, signed=1, multipoles=False):
    """Use small tables; select the representation independently of its axes."""
    settings = compact_grid_settings(reference_mode=reference_mode)
    settings["signed_baseline"] = signed
    settings["prepare_multipoles"] = int(multipoles)
    settings["lmax"] = 3
    return settings


def prepare_grid(model, adapter, ia, settings):
    """Exercise both owning adapters without copying any prepared values."""
    if adapter == "notebook":
        return model.projection_grid_notebook(
            bins=BINS,
            parameters=ia,
            settings=settings,
        )
    return model.projection_grid(
        bins=BINS,
        parameters=ia,
        settings=settings,
    )


def projected(grid, adapter, sides, component):
    """The notebook adapter also receives a non-C-contiguous triangle array."""
    if adapter == "notebook":
        return grid.projected_notebook(
            sides=np.asfortranarray(sides),
            component=component,
        )
    return grid.projected(sides=sides, component=component)


def original_nodes(grid, settings):
    """Recover original RUV knots from either the original or dense axes."""
    stride = 1 if settings["reference_mode"] else settings["table_boost"]
    axes = grid.axes()
    triangles = []
    for ir, iu, iv in ((2, 4, 2), (3, 5, 3), (4, 6, 4)):
        radius = np.exp(axes["log_r"][stride * ir])
        ratio = np.exp(axes["log_u"][stride * iu])
        difference = axes["v"][stride * iv]
        triangles.append(
            [radius * (1.0 + ratio * difference), radius, radius * ratio]
        )
    return np.asarray(triangles, dtype=np.float64)


def direct(model, sides, amplitude):
    """Use the actual line-of-sight integral, with no prepared RUV table."""
    return model.projected(
        sides=sides,
        bins=BINS,
        parameters=parameters(amplitude=amplitude),
        component=1,
    )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("reference_mode", [0, 1])
def test_signed_baseline_negative_and_zero_crossing(compiled, adapter, reference_mode):
    """Signed values, including an actual zero crossing, survive coarse knots.

    The bounded root calculation only chooses a diagnostic IA amplitude; it
    does not replace any production quadrature. Both endpoint signs must be
    reached by the independently evaluated model before the test continues.
    """
    compiled.set_omp_threads(nthreads=1)
    model = compiled.production.PhysicsModel(**signed_inputs())
    settings = grid_settings(reference_mode=reference_mode)
    negative = prepare_grid(
        model=model,
        adapter=adapter,
        ia=parameters(amplitude=5.0),
        settings=settings,
    )
    sides = original_nodes(grid=negative, settings=settings)
    gravity = direct(model=model, sides=sides, amplitude=0.0)
    strong = direct(model=model, sides=sides, amplitude=5.0)
    assert np.all(gravity > 0.0), "The positive GGG error scale was not reached"
    assert np.all(strong < 0.0), "The controlled source model did not reach B_NLA<0"
    def scaled_baseline(amplitude):
        """Express the direct signed projection in positive GGG units."""
        value = direct(model=model, sides=sides[:1], amplitude=amplitude)
        return value[0] / gravity[0]

    zero = brentq(
        f=scaled_baseline,
        a=0.0,
        b=5.0,
        xtol=1.0e-13,
    )
    delta = min(0.02, zero / 4.0, (5.0 - zero) / 4.0)
    assert direct(model=model, sides=sides[:1], amplitude=zero - delta)[0] > 0.0
    assert direct(model=model, sides=sides[:1], amplitude=zero + delta)[0] < 0.0
    for amplitude in (5.0, zero - delta, zero, zero + delta):
        table = prepare_grid(
            model=model,
            adapter=adapter,
            ia=parameters(amplitude=amplitude),
            settings=settings,
        )
        expected = direct(model=model, sides=sides, amplitude=amplitude)
        actual = projected(grid=table, adapter=adapter, sides=sides, component=1)
        np.testing.assert_allclose(
            actual / gravity,
            expected / gravity,
            rtol=5.0e-11,
            atol=5.0e-11,
        )
        residual = projected(grid=table, adapter=adapter, sides=sides, component=2)
        assert_bits(
            actual=residual,
            expected=np.zeros_like(residual),
            context="Pure NLA must retain exactly zero prepared TATT residual",
        )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_signed_baseline_refinement_away_from_knots(compiled, adapter):
    """Refine all RUV coordinates together and use GGG to scale signed errors.

    A relative error divided by NLA would diverge near its zero. The positive
    gravity-only projection instead provides the same physical scale for
    both resolutions. The actual error is attached to an assertion failure;
    passing this small test is not a survey-accuracy determination.
    """
    compiled.set_omp_threads(nthreads=1)
    model = compiled.production.PhysicsModel(**signed_inputs())
    sides = np.asarray(
        [
            [100.0, 130.0, 170.0],
            [350.0, 210.0, 230.0],
            [800.0, 950.0, 1200.0],
        ]
    )
    expected = direct(model=model, sides=sides, amplitude=5.0)
    gravity = direct(model=model, sides=sides, amplitude=0.0)
    assert np.all(expected < 0.0)
    errors = []
    for count in (7, 13):
        settings = grid_settings(reference_mode=0)
        settings.update(ell_min=40.0, ell_max=2000.0, nr=count, nu=count, nv=count)
        table = prepare_grid(
            model=model,
            adapter=adapter,
            ia=parameters(amplitude=5.0),
            settings=settings,
        )
        actual = projected(grid=table, adapter=adapter, sides=sides, component=1)
        assert np.all(np.isfinite(actual))
        errors.append(float(np.max(np.abs((actual - expected) / gravity))))
    assert errors[1] < errors[0], f"GGG-scaled coarse/fine maximum errors: {errors}"


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_signed_baseline_preserves_default_and_tatt_residual(compiled, adapter):
    """Omitting the new setting keeps old bits; signed mode does not alter IA2."""
    compiled.set_omp_threads(nthreads=1)
    model = compiled.production.PhysicsModel(**physical_inputs())
    settings = grid_settings(reference_mode=0, signed=0)
    ia = parameters(amplitude=0.0, residual=True)
    ordinary = prepare_grid(model=model, adapter=adapter, ia=ia, settings=settings)
    omitted = dict(settings)
    del omitted["signed_baseline"]
    implicit = prepare_grid(model=model, adapter=adapter, ia=ia, settings=omitted)
    signed = dict(settings)
    signed["signed_baseline"] = 1
    alternative = prepare_grid(model=model, adapter=adapter, ia=ia, settings=signed)
    sides = np.asarray([[190.0, 280.0, 360.0], [430.0, 170.0, 310.0]])
    for component in (0, 1, 2):
        assert_bits(
            actual=projected(grid=ordinary, adapter=adapter, sides=sides,
                             component=component),
            expected=projected(grid=implicit, adapter=adapter, sides=sides,
                               component=component),
            context=f"Omitted signed_baseline changed component {component}",
        )
    assert_bits(
        actual=projected(grid=alternative, adapter=adapter, sides=sides, component=2),
        expected=projected(grid=ordinary, adapter=adapter, sides=sides, component=2),
        context="Signed baseline changed the independently prepared TATT residual",
    )


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_signed_baseline_workers_and_retained_query(compiled, adapter):
    """Reused geometric maps read fresh signed physics and restore old outputs."""
    model = compiled.production.PhysicsModel(**signed_inputs())
    ordinary = grid_settings(reference_mode=0, signed=0, multipoles=True)
    signed = grid_settings(reference_mode=0, signed=1, multipoles=True)
    grids = []
    references = []
    for amplitude, settings in ((0.0, ordinary), (5.0, signed), (0.0, signed)):
        compiled.set_omp_threads(nthreads=1)
        table = prepare_grid(
            model=model,
            adapter=adapter,
            ia=parameters(amplitude=amplitude),
            settings=settings,
        )
        grids.append(table)
        references.append(old_route(compiled=compiled, grid=table))
    assert not np.array_equal(references[0], references[1])
    try:
        for workers in (1, 2, 4):
            compiled.set_omp_threads(nthreads=workers)
            threaded = prepare_grid(
                model=model,
                adapter=adapter,
                ia=parameters(amplitude=5.0),
                settings=signed,
            )
            plan, _, weights = plan_and_weights(compiled=compiled, workers=workers)
            sequence = (grids[0], threaded, grids[2], grids[0], grids[1], grids[0])
            expected_indices = (0, 1, 2, 0, 1, 0)
            for table, expected_index in zip(sequence, expected_indices):
                actual = retained_route(
                    compiled=compiled,
                    adapter=adapter,
                    plan=plan,
                    grid=table,
                    weights=weights,
                )
                assert_bits(
                    actual=actual,
                    expected=references[expected_index],
                    context=f"Signed table/query restoration with {workers} workers",
                )
    finally:
        compiled.set_omp_threads(nthreads=1)
