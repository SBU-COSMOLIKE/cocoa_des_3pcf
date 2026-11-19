"""Check power interpolation at joins of Cocoa's redshift grid pieces.

The analytic nonlinear spectrum has a curved logarithm in redshift and a
power law in k. Its power-law shape removes wavenumber interpolation error
from this check. Refining the input redshifts then measures the error of
the prepared redshift interpolant, including both sides of a grid join.
"""

from des3pcf.variance import reference_physics_model

import numpy as np
import pytest

from test_physics_model import physical_inputs
from test_bihalofit_python import evaluate, query_triangles
from des3pcf.bihalofit import BiHalofit
from des3pcf.inputs import uniform_segments


def piecewise_inputs(refinement):
    """Sample one physical model more densely without moving the grid join.

    Redshifts below 0.4 have finer spacing, as in Cocoa's likelihoods.
    Shared endpoints occur once in the concatenated coordinate array.
    Background and source values are resampled from the same functions,
    so changing refinement changes no physical parameter.
    """
    inputs = physical_inputs()
    low = np.linspace(start=0.0, stop=0.4, num=20 * refinement + 1)
    high = np.linspace(start=0.4, stop=2.0, num=24 * refinement + 1)
    redshift = np.concatenate((low[:-1], high))
    wave = inputs["k"]
    evolution = np.exp(-1.1 * redshift - 0.3 * redshift**2)
    inputs["z"] = redshift
    inputs["p_nonlinear"] = 1.0e4 * evolution[:, None] * wave[None, :]**-1.5
    inputs["chi"] = 3000.0 * redshift / (1.0 + 0.35 * redshift)
    inputs["dz_dchi"] = (1.0 + 0.35 * redshift)**2 / 3000.0
    inputs["growth"] = 1.0 / (1.0 + redshift)
    inputs["sources"] = np.array(object=[
        redshift**2 * np.exp(-(redshift / 0.6)**1.5),
        redshift**2 * np.exp(-(redshift / 0.95)**1.5),
    ])
    inputs["source_readers"] = False
    inputs["table_boost"] = 16
    return inputs


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_piecewise_power_refinement(compiled, adapter):
    """Refinement reduces error at the join and at the physical endpoints.

    Natural end conditions need not match the analytic curvature at a
    segment boundary. Halving each coarse spacing should reduce that
    boundary error quadratically. The analytic formula is independent of
    both the C spline coefficients and its direct-index readers.
    """
    redshift = np.linspace(start=0.0, stop=2.0, num=2001)
    near_join = np.array(object=[
        np.nextafter(0.4, 0.0),
        0.4,
        np.nextafter(0.4, 1.0),
    ])
    redshift = np.concatenate((redshift, near_join))
    wave = np.full(shape=redshift.shape, fill_value=0.37)
    expected = 1.0e4 * wave**-1.5 * np.exp(-1.1 * redshift - 0.3 * redshift**2)
    errors = []
    factory = getattr(compiled, adapter).PhysicsModel
    for refinement in (1, 2, 4):
        model = reference_physics_model(
            factory=factory, **piecewise_inputs(refinement=refinement))
        actual = model.power(k=wave, z=redshift, mode=2)
        errors.append(np.max(np.abs(actual / expected - 1.0)))
        # Adjacent floating-point coordinates at the join must agree to
        # rounding accuracy; no cell may read across a missing interval.
        np.testing.assert_allclose(actual[-3:], expected[-3:], rtol=2.0e-13)
    assert errors[1] < errors[0] / 3.0
    assert errors[2] < errors[1] / 3.0
    assert errors[2] < 1.0e-5


def test_python_and_native_piecewise_fit(compiled):
    """Both languages consume the same piecewise growth preparation.

    Matter fractions are sampled at the fit anchors to isolate the spline
    convention from the physical background formula. The equation-level
    2e-11 tolerance is the same as the uniform-grid comparison.
    """
    inputs = piecewise_inputs(refinement=1)
    anchors = np.linspace(start=0.0, stop=2.0, num=33)
    matter = inputs["omega_m"] * (1.0 + anchors)**3
    omega = matter / (matter + inputs["omega_de"])
    supplied = {}
    for name in ("k", "p_linear0", "z", "growth", "ns", "sigma8"):
        supplied[name] = inputs[name]
    supplied["omega_m_redshift"] = anchors
    supplied["omega_m_z"] = omega
    python = BiHalofit(**supplied, fit_nz=33, table_boost=16)
    native = reference_physics_model(factory=compiled.production.PhysicsModel, **inputs)
    sides, redshift = query_triangles()
    for component in range(4):
        expected = native.matter(sides=sides, redshifts=redshift, component=component)
        actual = evaluate(model=python, sides=sides, redshift=redshift,
                          component=component)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=0.0)


def test_uniform_segment_validation():
    """Accept declared pieces, but reject irregular k and excess z pieces."""
    inputs = piecewise_inputs(refinement=1)
    assert uniform_segments(coordinates=inputs["z"], name="z") == [(0, 20), (20, 44)]
    logk = np.linspace(start=-5.0, stop=3.0, num=96)
    logk[32] += 0.01
    with pytest.raises(ValueError, match="more than 1 uniform pieces"):
        uniform_segments(coordinates=logk, name="log k", maximum=1)
    redshift = np.cumsum(np.arange(12, dtype=float))
    with pytest.raises(ValueError, match="more than 10 uniform pieces"):
        uniform_segments(coordinates=redshift, name="z")
