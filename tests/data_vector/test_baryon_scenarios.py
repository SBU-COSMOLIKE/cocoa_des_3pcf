"""Check physical feedback preparation and the installed paired BACCO API.

No survey projection is run here. The checks concentrate on mistakes that
can produce plausible curves: changed primordial normalization, a cold/total
amplitude substitution, silently lost triangles and changed input order.
"""

from contextlib import nullcontext
import importlib
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from des3pcf.baryon_scenarios import (
    _checked_grid, bfmt_model_input, bfmt_power_response,
)
from theory.bacco_bispectrum.emulator import (
    BaccoColdResponses, checked_bacco_triangles, installed_bacco_weights,
)


def model_input(tmp_path):
    """Supply one cosmology and its saved CAMB normalization without running it."""
    source = tmp_path / "external_modules/code/baryon_suppression/bfmt.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Only the file-presence check is used by these unit tests.\n")
    parameters = {
        "omega_m": .289, "omega_b": .0463, "h": .722, "ns": .959,
        "sigma8": .787, "w0": -1., "wa": 0., "mnu": .06,
        "num_massive_neutrinos": 3, "nnu": 3.046,
    }
    return {
        "cocoa_root": tmp_path, "parameters": parameters,
        "metadata": {"As": 1.85e-9, "omega_nu_h2": .000645,
                     "parameters": dict(parameters)},
        "options": {"baryon_model": 1, "spk_fb_model": 1},
        "point": {"fb_a_spk": .4, "fb_pow_spk": .3},
    }


def test_background_uses_saved_exact_neutrino_density(tmp_path):
    """The same saved density reconstructs total Omega_m without double counting."""
    supplied = model_input(tmp_path=tmp_path)
    result = bfmt_model_input(**supplied)
    fixed = result["params"]
    total = (fixed["ombh2"] + fixed["omch2"] + .000645) / .722**2
    np.testing.assert_allclose(total, .289, rtol=2.e-16)
    assert result["theory"]["camb"]["extra_args"]["num_massive_neutrinos"] == 3


@pytest.mark.parametrize("name", ["sigma8", "h", "mnu", "omega_b", "wa"])
def test_changed_cosmology_refuses_stale_amplitude(tmp_path, name):
    """A newly supplied cosmology cannot inherit an old CAMB amplitude silently."""
    supplied = model_input(tmp_path=tmp_path)
    supplied["parameters"][name] += .001
    with pytest.raises(ValueError, match="Regenerate CAMB metadata"):
        bfmt_model_input(**supplied)


def test_power_grid_validation_owns_its_arrays():
    """Preparation cannot retain a caller's mutable grid or accept reversed k."""
    z = np.linspace(0., 3., 4)
    k = np.geomspace(.01, 10., 5)
    copied_z, copied_k = _checked_grid(z=z, k=k)
    z[1] = 99.
    k[1] = 99.
    assert copied_z[1] == 1.
    assert copied_k[1] < 1.
    with pytest.raises(ValueError, match="increasing"):
        _checked_grid(z=copied_z, k=copied_k[::-1])


def test_actual_component_logger_detects_unity_fallback(tmp_path, monkeypatch):
    """A colored/differently named logger cannot hide bfmt's error-to-unity path."""
    import cobaya.model

    supplied = model_input(tmp_path=tmp_path)
    logger = logging.getLogger("colored-bfmt-regression")
    logger.setLevel(logging.CRITICAL)
    before = list(logger.handlers)
    request = {}

    def add_requirements(requirements):
        """Record the physical request so the unit conversion is tested too."""
        request.update(requirements["baryon_suppression"])

    def logposterior(params_values):
        """Imitate an upstream failure that nevertheless returns finite likelihood."""
        logger.error("emulator failed; using unity suppression")
        return SimpleNamespace(logpost=0.)

    model = SimpleNamespace(
        theory={"bfmt": SimpleNamespace(log=logger)},
        add_requirements=add_requirements, logposterior=logposterior,
        provider=SimpleNamespace(get_baryon_suppression=lambda: {
            z: np.ones(4) for z in request["z"]
        }),
    )
    monkeypatch.setattr(cobaya.model, "get_model",
                        lambda info_or_yaml_or_file: nullcontext(model))
    original_policy = np.geterr()
    with pytest.raises(RuntimeError, match="using unity"):
        bfmt_power_response(**supplied, z=np.arange(4.), k=np.arange(1., 5.))
    np.testing.assert_array_equal(request["k"], .722 * np.arange(1., 5.))
    assert logger.level == logging.CRITICAL
    assert logger.handlers == before
    assert np.geterr() == original_policy


def test_wrapper_restores_dmo_when_later_scenario_fails(tmp_path, monkeypatch):
    """An interrupted scenario sweep cannot leave its previous power installed."""
    import cosmolike_des_3pcf_feedback_wrappers as wrapper

    catalog = tmp_path / "scenarios.yaml"
    catalog.write_text('''format: des3pcf-bfmt-scenarios-v1
theory_options: {}
scenarios:
  - {name: first, options: {}, parameters: {}, paper: test}
  - {name: failed, options: {}, parameters: {}, paper: test}
''')
    table = {"k": np.arange(1., 5.), "z": np.arange(4.),
             "p_nonlinear": np.ones((4, 4))}
    rows = SimpleNamespace(kind=np.array(["xip"]), select=lambda indices: rows)
    provider = SimpleNamespace(get=lambda parameters: (None, table, {}))
    config = {"ia": {}, "parameters": {}}
    monkeypatch.setattr(wrapper.driver, "load_configuration",
                        lambda path, command: (config, path, "example hash"))
    monkeypatch.setattr(wrapper.driver, "compiled_extension", lambda config: None)
    monkeypatch.setattr(wrapper.driver, "rows_for_run", lambda config, path: rows)
    monkeypatch.setattr(wrapper.driver, "provider_for_run",
                        lambda **kwargs: (provider, {}))
    monkeypatch.setattr(wrapper.driver, "prediction_for_run", lambda **kwargs: None)
    installed = []

    def record_mean(prediction, tables, metadata, parameters, ia):
        """Represent installation of the supplied nonlinear power in the core."""
        installed.append(tables["p_nonlinear"].copy())
        return tables["p_nonlinear"].sum(axis=0)

    def first_succeeds_then_fails(**kwargs):
        """One model modifies power before the following model rejects its point."""
        if len(installed) > 1:
            raise RuntimeError("rejected next scenario")
        return np.full((4, 4), .8), {}

    monkeypatch.setattr(wrapper, "_two_point_mean", record_mean)
    monkeypatch.setattr(wrapper, "bfmt_power_response", first_succeeds_then_fails)
    with pytest.raises(RuntimeError, match="rejected next scenario"):
        wrapper.run_scenarios(configuration=tmp_path / "config.yaml",
                              catalog_path=catalog)
    np.testing.assert_array_equal(installed[1], .8 * table["p_nonlinear"])
    np.testing.assert_array_equal(installed[-1], table["p_nonlinear"])


def test_wrapper_passes_nonzero_shear_calibration(compiled):
    """The actual native xi calculation applies one calibration per leg."""
    import cosmolike_des_3pcf_feedback_wrappers as wrapper
    from native_inputs import init_two_point
    from test_pipeline import toy_tables
    from cobaya.likelihoods.des_3pcf.survey import forecast_rows

    tables, parameters = toy_tables()
    tables["p_linear"] = tables["p_nonlinear"].copy()
    init_two_point(core=compiled, nbin=2)
    rows = forecast_rows(nbin=2, aperture_radii=[], xi_edges=[.001, .002])
    ell, dell = compiled.quadrature(
        lower=30.0, upper=2000.0, count=24, logarithmic=True,
    )
    theta, theta_weight = compiled.xi_bin_weights(
        kind=rows.kind.tolist(), angles=rows.angles,
        count=3, pair_weight="uniform_area",
    )
    prediction = SimpleNamespace(
        compiled=compiled,
        provider=SimpleNamespace(nbin=2),
        two_point_core={"threads": 1},
        xi_ell=ell,
        xi_dell=dell,
        xi_theta=theta,
        xi_weight=theta_weight,
        xi_rows=rows,
        xi_sign=np.where(rows.kind == "xim", -1, 1).astype(np.int32),
    )
    baseline = wrapper._two_point_mean(
        prediction=prediction, tables=tables, metadata={},
        parameters=parameters, ia={"a1": .2},
    )
    changed = wrapper._two_point_mean(
        prediction=prediction, tables=tables, metadata={},
        parameters=dict(parameters, m1=.02, m2=-.01), ia={"a1": .2},
    )
    calibration = np.array([1.02, .99])
    factors = calibration[rows.bins[:, 0]] * calibration[rows.bins[:, 1]]
    np.testing.assert_allclose(changed, baseline*factors, rtol=3.e-15)
    restored = wrapper._two_point_mean(
        prediction=prediction, tables=tables, metadata={},
        parameters=parameters, ia={"a1": .2},
    )
    np.testing.assert_array_equal(restored, baseline)


@pytest.mark.parametrize("sides", [
    [[.1, .1, .21]], [[.1, .1, .2]], [[0., 1., 1.]],
    [[.001, .1, .1]], [[1., 1., np.nan]],
    [[.1, .1, .2 - 1.e-12]],  # Rounds to a degenerate float32 triangle.
])
def test_bad_triangles_are_rejected_before_upstream_row_filter(sides):
    """Invalid or float32-degenerate triangles never reach BACCO's silent mask."""
    with pytest.raises(ValueError):
        checked_bacco_triangles(sides=sides, k_range=[.01785654, 17.655489])


def test_triangle_row_order_survives_leg_sorting():
    """Sorting the three legs changes neither triangle count nor row identity."""
    actual = checked_bacco_triangles(
        sides=[[.3, .2, .25], [.8, 1., .6]], k_range=[.01785654, 17.655489],
    )
    np.testing.assert_array_equal(actual, np.asarray([[.2, .25, .3], [.6, .8, 1.]],
                                                   dtype=np.float32))


@pytest.fixture(scope="module")
def bacco():
    """Use installed paired weights only; a missing package cannot cause a download."""
    try:
        installed_bacco_weights()
    except FileNotFoundError:
        pytest.skip("Cocoa's optional paired BACCO weights are not installed.")
    return BaccoColdResponses(coordinates={
        "omega_cold": .288, "omega_baryon": .0463, "sigma8_cold": .79,
        "M_c": 14., "eta": -.3, "beta": -.22,
        "M1_z0_cen": 10.5, "theta_inn": -.86,
    })


def test_paired_bacco_permutation_and_single_triangle(bacco):
    """Single/batched calls preserve calibrated responses under leg permutation."""
    sides = np.asarray([[.2, .3, .4], [1., 1., 1.]])
    power, bispectrum = bacco.evaluate(z=.5, k=[.2, 1.], sides=sides)
    other_p, other_b = bacco.evaluate(z=.5, k=[.2, 1.], sides=sides[:, ::-1])
    np.testing.assert_array_equal(power, other_p)
    np.testing.assert_array_equal(bispectrum, other_b)
    one_p, one_b = bacco.evaluate(z=.5, k=[.2], sides=sides[:1])
    # Upstream neural-network arithmetic is float32 and changes matrix size
    # between these calls. Test agreement well below its percent calibration.
    np.testing.assert_allclose(one_p, power[:1], rtol=3.e-6)
    np.testing.assert_allclose(one_b, bispectrum[:1], rtol=3.e-6)
    assert np.all(bispectrum > 0.)


@pytest.mark.parametrize("z,k,sides", [
    (3.1, [.2], [[.2, .3, .4]]),
    (.5, [100.], [[.2, .3, .4]]),
    (.5, [.2], [[.001, .3, .3]]),
])
def test_paired_bacco_has_no_implicit_extrapolation(bacco, z, k, sides):
    """A diagnostic refuses unsupported redshift, power k or triangle sides."""
    with pytest.raises(ValueError):
        bacco.evaluate(z=z, k=k, sides=sides)


def test_figure_labels_check_connecting_segments():
    """A label between two sampled data points must not obscure their line."""
    from matplotlib import pyplot as plt

    plots = importlib.import_module("cosmolike_notebook_utils.3pcf")
    figure, axis = plt.subplots()
    axis.plot([0., 1.], [0., 1.], marker="o")
    label = axis.text(.5, .5, "triangle response", ha="center", va="center")
    with pytest.raises(ValueError, match="touches a plotted line"):
        plots.check_figure_labels(figure=figure)
    label.set_position((.2, .8))
    plots.check_figure_labels(figure=figure)
    plt.close(figure)


def test_figure_labels_ignore_undrawn_connections_and_clipped_segments():
    """Marker-only data and invisible off-axis segments do not obstruct text."""
    from matplotlib import pyplot as plt

    plots = importlib.import_module("cosmolike_notebook_utils.3pcf")
    figure, axis = plt.subplots()
    axis.plot([0., 1.], [0., 1.], marker="o", linestyle="None")
    axis.text(.5, .5, "no connecting line", ha="center", va="center")
    plots.check_figure_labels(figure=figure)
    axis.plot([.5, .5], [1.2, 2.0])
    axis.set_xlim(0., 1.)
    axis.set_ylim(0., 1.)
    axis.text(.5, 1.3, "clipped away", ha="center", va="center")
    plots.check_figure_labels(figure=figure)
    plt.close(figure)
