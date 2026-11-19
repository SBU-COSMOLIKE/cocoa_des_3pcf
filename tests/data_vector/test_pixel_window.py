"""Check three-leg smoothing, input ownership and its sharp angular support.

The direct multipole reference uses an independently written Legendre
sum with one interpolated W for each side. Constant-window tests cover
prepared tables, both IA components and both array adapters. Raw matter
and projected-bispectrum queries remain unsmoothed building blocks.
"""

from des3pcf.variance import reference_physics_model

import gc
import json
import yaml
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import eval_legendre

from des3pcf.pixel_window import checked_window, load_healpix_window
from test_physics_model import physical_inputs, compact_grid_settings
from test_grid_transform_reuse import plan_and_weights


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_constant_window_owned_and_grid_survives_model(compiled, adapter):
    """Three legs give W cubed; deleting the model cannot invalidate its grid."""
    inputs = physical_inputs()
    original = reference_physics_model(factory=getattr(compiled, adapter).PhysicsModel, **inputs)
    window = np.full(300001, .8)
    supplied = window
    if adapter == "notebook":
        storage = np.full(600002, .8)
        supplied = storage[::2]
    smoothed = reference_physics_model(factory=getattr(compiled, adapter).PhysicsModel, **inputs, pixel_window=supplied)
    supplied[:] = .2
    settings = compact_grid_settings(reference_mode=0)
    settings.update(signed_baseline=1)
    bins = np.array([0, 1, 0], dtype=np.int32)
    parameters = {"a1": .5, "a2": .8, "bta": .7}
    baseline = original.projection_grid(
        bins=bins, parameters=parameters, settings=settings,
    )
    grid = smoothed.projection_grid(bins=bins, parameters=parameters, settings=settings)
    ell = np.array([100., 220., 350.])
    before = grid.multipoles(ell=ell)
    del smoothed
    gc.collect()
    np.testing.assert_array_equal(x=before, y=grid.multipoles(ell=ell))
    for component in (0, 1, 2):
        expected = .8**3 * baseline.multipoles(ell=ell, component=component)
        actual = grid.multipoles(ell=ell, component=component)
        np.testing.assert_allclose(
            actual=actual, desired=expected, rtol=3.e-13, atol=1.e-32,
        )
    restored = reference_physics_model(factory=getattr(compiled, adapter).PhysicsModel, **inputs)
    restored_grid = restored.projection_grid(bins=bins, parameters=parameters,
                                             settings=settings)
    np.testing.assert_array_equal(x=baseline.multipoles(ell=ell),
                                  y=restored_grid.multipoles(ell=ell))


def test_direct_multipoles_use_window_on_all_three_legs(compiled):
    """The closure leg's angle-dependent window belongs inside the mu integral."""
    window = np.exp(-np.arange(1201) / 400.)
    model = reference_physics_model(factory=compiled.production.PhysicsModel, **physical_inputs(), pixel_window=window)
    raw = reference_physics_model(factory=compiled.production.PhysicsModel, **physical_inputs())
    ell = np.array([100., 330., 600.])
    bins = np.array([0, 1, 0], dtype=np.int32)
    parameters = {"a1": .4, "a2": .6, "bta": .3}
    nodes, weights = np.polynomial.legendre.leggauss(deg=24)
    polynomials = np.polynomial.legendre.legvander(x=nodes, deg=3)
    expected = np.empty((4, 3, 3))
    for first in range(3):
        for second in range(3):
            third = np.sqrt((ell[first] - ell[second])**2
                            + 2 * ell[first] * ell[second] * (1. - nodes))
            sides = np.column_stack((np.full(24, ell[first]),
                                     np.full(24, ell[second]), third))
            signal = raw.projected(sides=sides, bins=bins, parameters=parameters)
            np.testing.assert_array_equal(
                x=signal,
                y=model.projected(sides=sides, bins=bins, parameters=parameters),
            )
            factor = np.prod(np.interp(sides, np.arange(window.size), window,
                                       left=1., right=0.), axis=1)
            for order in range(4):
                expected[order, first, second] = (order + .5) * np.sum(
                    weights * polynomials[:, order] * signal * factor,
                )
    actual = model.multipoles(ell=ell, bins=bins, parameters=parameters, lmax=3, nmu=24)
    np.testing.assert_allclose(actual=actual, desired=expected, rtol=4.e-12, atol=1.e-32)


def test_retained_transform_reads_replaced_window_and_restores(compiled):
    """A plan may reuse geometry across windows but must refill physical values."""
    plan, arguments, weights = plan_and_weights(compiled=compiled, workers=1)
    settings = compact_grid_settings(reference_mode=0)
    settings["lmax"] = arguments["lmax"]
    values = []
    for amplitude in (1., .7, 1.):
        model = reference_physics_model(factory=compiled.production.PhysicsModel,
            **physical_inputs(), pixel_window=np.full(300001, amplitude),
        )
        grid = model.projection_grid(
            bins=np.array([0, 1, 0], dtype=np.int32), parameters={}, settings=settings,
        )
        actual = plan.aperture_from_grid(grid=grid, weights_real=weights[0],
                                         weights_imag=weights[1])
        reference_plan, _, _ = plan_and_weights(compiled=compiled, workers=1)
        expected = reference_plan.aperture_from_multipoles(
            b_l=grid.multipoles(ell=arguments["ell"]),
            weights_real=weights[0], weights_imag=weights[1],
        )
        np.testing.assert_array_equal(x=actual, y=expected)
        values.append(actual)
    np.testing.assert_allclose(actual=values[1], desired=.7**3 * values[0], rtol=3.e-12)
    np.testing.assert_array_equal(x=values[0], y=values[2])


def test_full_filter_direct_and_prepared_windows(compiled):
    """The full-filter callbacks apply the same three factors exactly once."""
    baseline = reference_physics_model(factory=compiled.production.PhysicsModel, **physical_inputs())
    model = reference_physics_model(factory=compiled.production.PhysicsModel, **physical_inputs(),
                                             pixel_window=np.full(300001, .8))
    bins = np.array([0, 1, 0], dtype=np.int32)
    parameters = {"a1": .5, "a2": .4, "bta": .7}
    arguments = {
        "ell": np.array([100., 220., 400.]),
        "dell": np.array([80., 160., 200.]),
        "phi": (np.arange(4) + .5) * np.pi / 2.,
        "dphi": np.full(4, np.pi / 2.),
        "filters": np.array([[.001, .002, .003]]),
    }
    direct = model.aperture_full(**arguments, bins=bins, parameters=parameters)
    expected = baseline.aperture_full(**arguments, bins=bins, parameters=parameters)
    np.testing.assert_allclose(actual=direct, desired=.8**3 * expected, rtol=3.e-14)
    settings = compact_grid_settings(reference_mode=0)
    settings["prepare_multipoles"] = 0
    grid = model.projection_grid(bins=bins, parameters=parameters, settings=settings)
    baseline_grid = baseline.projection_grid(bins=bins, parameters=parameters,
                                              settings=settings)
    np.testing.assert_allclose(actual=grid.aperture_full(**arguments),
                              desired=.8**3 * baseline_grid.aperture_full(**arguments),
                              rtol=3.e-14)


def test_prepared_query_does_not_interpolate_across_first_two_leg_cutoffs(compiled):
    """A discontinuous supplied upper extension must give exact zero outside it."""
    model = reference_physics_model(factory=compiled.production.PhysicsModel, **physical_inputs(),
                                             pixel_window=np.ones(301))
    settings = compact_grid_settings(reference_mode=0)
    grid = model.projection_grid(bins=np.array([0, 1, 0], dtype=np.int32),
                                parameters={}, settings=settings)
    ell = np.array([100., 300., np.nextafter(300., np.inf), 400.])
    values = grid.multipoles(ell=ell)
    assert np.any(values[:, :2, :2] != 0.)
    np.testing.assert_array_equal(x=values[:, 2:, :], y=0.)
    np.testing.assert_array_equal(x=values[:, :, 2:], y=0.)


def test_prepared_nonconstant_window_angular_integral(compiled):
    """Integrate the sampled W3*B linear segments independently at table nodes."""
    window = np.exp(-np.arange(5001) / 400.)
    model = reference_physics_model(factory=compiled.production.PhysicsModel, **physical_inputs(), pixel_window=window)
    settings = compact_grid_settings(reference_mode=1)
    grid = model.projection_grid(
        bins=np.array([0, 1, 0], dtype=np.int32),
        parameters={"a1": .5, "a2": .8, "bta": .7}, settings=settings,
    )
    axes = grid.axes()
    magnitude = np.exp(axes["log_ell"][3])
    psi = np.exp(axes["log_psi"][-3])
    ell = np.array([magnitude * np.cos(psi), magnitude * np.sin(psi)])
    actual = grid.multipoles(ell=ell)
    nodes, weights = np.polynomial.legendre.leggauss(deg=5)
    mu = axes["mu"]
    for first, second in ((0, 1), (1, 0)):
        third = np.sqrt((ell[first] - ell[second])**2
                        + 2. * ell[first] * ell[second] * (1. - mu))
        sides = np.column_stack((np.full(mu.size, ell[first]),
                                 np.full(mu.size, ell[second]), third))
        sampled = grid.projected(sides=sides)
        sampled *= np.interp(third, np.arange(window.size), window, right=0.)
        expected = np.zeros(settings["lmax"] + 1)
        for interval in range(mu.size - 1):
            lower, upper = mu[interval:interval + 2]
            coordinate = .5 * (lower + upper + (upper - lower) * nodes)
            fraction = (coordinate - lower) / (upper - lower)
            value = sampled[interval] + fraction * (sampled[interval + 1]
                                                    - sampled[interval])
            for order in range(settings["lmax"] + 1):
                expected[order] += (order + .5) * .5 * (upper - lower) * np.sum(
                    weights * value * eval_legendre(order, coordinate),
                )
        expected *= np.prod(np.interp(ell, np.arange(window.size), window, right=0.))
        np.testing.assert_allclose(actual=actual[:, first, second], desired=expected,
                                  rtol=2.e-10, atol=1.e-28)


def test_pinned_healpix_schema_and_default_truncation():
    """The official scalar table preserves its ulp-sized W0 excess and ell cutoff."""
    project = Path(__file__).resolve().parents[2]
    manifest = yaml.safe_load((project / "data/pixel_window_manifest.yaml").read_text())
    path = project / "data" / manifest["path"]
    if not path.is_file():
        pytest.fail("The bundled HEALPix FITS is missing from data/upstream_healpix.")
    values, metadata = load_healpix_window(path=path, nside=1024,
                                         sha256=manifest["sha256"])
    assert values.shape == (3072,)
    assert values[0] > 1.
    assert abs(values[0] - 1.) < 2.e-15
    np.testing.assert_allclose(actual=values[-1], desired=.6464505317040627, rtol=1.e-15)
    assert metadata["ell_max"] == 3071
    with pytest.raises(ValueError, match="SHA256"):
        load_healpix_window(path=path, nside=1024, sha256="0" * 64)


@pytest.mark.parametrize("values", [[1.], [1., np.nan], [1., -.1], [[1., 1.]]])
def test_invalid_windows_stop_in_python(values):
    """Malformed numerical windows fail before creating any physical C model."""
    with pytest.raises(ValueError):
        checked_window(values=values)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_unmatched_analytic_covariance_is_refused(adapter):
    """A smoothed mean cannot silently inherit the unsmoothed analytic covariance."""
    worker = Path(__file__).resolve().parents[1] / "refusal_worker.py"
    result = subprocess.run(args=[sys.executable, "-B", str(worker),
                                  "pixel_covariance", adapter],
                            capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "pixel-smoothed means require a separately matched covariance" in output
    assert "Traceback" not in output
    assert "runtime error:" not in output
