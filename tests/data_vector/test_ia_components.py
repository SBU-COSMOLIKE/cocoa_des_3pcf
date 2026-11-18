"""Check physical IA operators independently and through the aperture integral.

The equilateral identity tests the physical torque-to-tensor conversion.
The closure tests compare with the pre-existing total calculation, including
the finite source-kernel splines and signed interpolation. Small grids test
algebra and state ownership; they are not survey accuracy settings.
"""

import numpy as np
import pytest

from test_physics_model import physical_inputs
from test_physics_setting_sequences import base_case
from test_transform_setting_sequences import configuration, make_plan, aperture_arguments


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_equilateral_operators(compiled, adapter):
    """Equation 51 gives ddE/P²=-C2/6-C1delta/2 for an equilateral triangle."""
    model = getattr(compiled, adapter).PhysicsModel(**physical_inputs())
    sides = np.array([[0.2, 0.2, 0.2]], dtype=np.float64)
    redshift = np.array([0.0])
    parameters = {"a1": 0.7, "a2": 0.8, "bta": 0.4, "zpivot": 0.0}
    power = model.power(k=np.array([0.2]), z=redshift, mode=1)[0]
    c1 = -0.7 * 0.0134 * 0.3
    c2 = 0.8 * 5.0 * 0.0134 * 0.3
    expected = {2: -0.4 * c1 / 2.0, 3: -c2 / 6.0}
    evaluate = model.ia
    if adapter == "notebook":
        evaluate = model.ia_notebook
    for operator in (2, 3):
        selected = dict(parameters, operator_component=operator)
        result = evaluate(
            sides=sides, redshifts=redshift, parameters=selected,
            remove_alignment=True,
        )
        np.testing.assert_allclose(result[0, 0], expected[operator] * power**2,
                                   rtol=2.0e-13)
    torque = evaluate(
        sides=sides, redshifts=redshift,
        parameters=dict(parameters, a1=0.0, operator_component=3),
        remove_alignment=True,
    )
    assert torque[0, 0] != 0.0
    np.testing.assert_array_equal(torque[0, 3:7], 0.0)


@pytest.mark.parametrize("scaled", [0, 1])
def test_components_close_after_interpolation_transform_and_aperture(compiled, scaled):
    """A reused transform must recover the total from signed pieces at 1/4 threads."""
    model = compiled.PhysicsModel(**physical_inputs())
    case = base_case()
    parameters = dict(case["ia"])
    bins = np.array([0, 1, 0], dtype=np.int32)
    settings = dict(case["grid"], signed_baseline=1, residual_scaling=scaled)
    transform, reached = configuration(case="baseline")
    plan, arguments = make_plan(compiled=compiled, settings=transform)
    ell = arguments["ell"]
    geometry = aperture_arguments(plan=plan, settings=transform)
    real, imag = compiled.production.aperture_weights(**geometry)
    weights = {"weights_real": real, "weights_imag": imag}
    grid = model.projection_grid(bins=bins, parameters=parameters, settings=settings)
    total = grid.multipoles(ell=ell)
    recovered = np.zeros_like(total)
    recovered_aperture = np.zeros(shape=transform["nfilter"])
    sides = np.array([[100., 150., 200.], [40., 60., 50.]])
    direct = model.projected(sides=sides, bins=bins, parameters=parameters)
    recovered_direct = np.zeros_like(direct)
    masks = (0, 4, 2, 1, 6, 3, 5, 7)
    for operator in (1, 2, 3, 4):
        for mask in masks:
            if operator != 1 and mask == 0:
                continue
            selected = dict(parameters, operator_component=operator, intrinsic_legs=mask)
            part = model.projection_grid(
                bins=bins, parameters=selected, settings=settings,
            )
            multipoles = part.multipoles(ell=ell)
            recovered += multipoles
            recovered_aperture += plan.aperture_from_multipoles(
                b_l=multipoles, **weights,
            )
            recovered_direct += model.projected(
                sides=sides, bins=bins, parameters=selected,
            )
    scale = np.max(np.abs(total))
    np.testing.assert_allclose(recovered, total, rtol=3.e-11, atol=scale * 3.e-14)
    np.testing.assert_allclose(recovered_direct, direct, rtol=3.e-13)
    # The transform is linear even when different signed components
    # overwrite its scratch buffers; the physical grid is never cached.
    total_aperture = plan.aperture_from_multipoles(b_l=total, **weights)
    np.testing.assert_allclose(recovered_aperture, total_aperture, rtol=5.e-11)
    compiled.set_omp_threads(nthreads=4)
    try:
        repeated = model.projection_grid(
            bins=bins, parameters=parameters, settings=settings,
        ).multipoles(ell=ell)
    finally:
        compiled.set_omp_threads(nthreads=1)
    np.testing.assert_array_equal(repeated, total)


def test_ordered_legs_and_nla_limits(compiled):
    """Move source labels and triangle legs together, then restore the initial state."""
    inputs = physical_inputs()
    sources = inputs["sources"]
    inputs["sources"] = np.vstack((sources, 0.3 * sources[0] + sources[1]))
    model = compiled.PhysicsModel(**inputs)
    sides = np.array([[120., 210., 260.]])
    bins = np.array([0, 1, 2], dtype=np.int32)
    parameters = {"a1": 0.8, "a2": 0.6, "bta": 0.4}
    original = model.projected(sides=sides, bins=bins, parameters=parameters)
    # Swapping physical legs 0 and 2 changes GGI (bit 2) to IGG (bit 0).
    for operator in (1, 2, 3):
        first = model.projected(
            sides=sides, bins=bins,
            parameters=dict(parameters, operator_component=operator, intrinsic_legs=4),
        )
        exchanged = model.projected(
            sides=np.ascontiguousarray(sides[:, [2, 1, 0]]),
            bins=np.array([2, 1, 0], dtype=np.int32),
            parameters=dict(parameters, operator_component=operator, intrinsic_legs=1),
        )
        np.testing.assert_allclose(exchanged, first, rtol=3.e-13)
    nla = dict(parameters, a2=0., bta=0.)
    for operator in (2, 3, 4):
        zero = model.projected(
            sides=sides, bins=bins, parameters=dict(nla, operator_component=operator),
        )
        np.testing.assert_array_equal(zero, 0.)
    negative_a1 = dict(parameters, a1=-parameters["a1"])
    for operator, parity in ((2, -1), (3, 1)):
        positive = model.projected(
            sides=sides, bins=bins, component=2,
            parameters=dict(parameters, operator_component=operator, intrinsic_legs=4),
        )
        negative = model.projected(
            sides=sides, bins=bins, component=2,
            parameters=dict(negative_a1, operator_component=operator, intrinsic_legs=4),
        )
        np.testing.assert_allclose(negative, parity * positive, rtol=3.e-13)
    repeated = model.projected(sides=sides, bins=bins, parameters=parameters)
    np.testing.assert_array_equal(repeated, original)

