"""Check the Cobaya feedback-scenario route and the installed paired BACCO API.

The scenario route builds Cobaya models of EXAMPLE_EVALUATE1.yaml on the
xi-only dataset with the bfmt block and the likelihood key
external_baryon_suppression (see EXAMPLE_FEEDBACK_SCENARIOS.ipynb).
The checks concentrate on mistakes that can produce plausible curves: a
bfmt fallback to unity hidden by its logger, feedback state surviving into
the gravity-only prediction, a changed FLAMINGO continuation leaking below
its pivot, silently lost triangles and changed input order.
"""

import importlib
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from notebook_functions import notebook_functions

from theory.bacco_bispectrum.emulator import (
    BaccoColdResponses, checked_bacco_triangles, installed_bacco_weights,
)


PROJECT = Path(__file__).resolve().parents[2]
COCOA = PROJECT.parents[1]


def example_information():
    """Load EXAMPLE_EVALUATE1.yaml as the notebook does, with an absolute CAMB path.

    Returns:
      (information, point): the Cobaya input without sampler and output,
      and the evaluate point of the yaml's override block.
    """
    from cobaya.yaml import yaml_load_file

    information = yaml_load_file(str(PROJECT / "EXAMPLE_EVALUATE1.yaml"))
    point = dict(information["sampler"]["evaluate"]["override"])
    information.pop("sampler")
    information.pop("output")
    information["theory"]["camb"]["path"] = str(COCOA / "external_modules/code/CAMB")
    return information, point


def catalog_entry(name):
    """Return one scenario and the shared bfmt options of the project catalog."""
    catalog = yaml.safe_load((PROJECT / "configs/feedback_scenarios.yaml").read_text())
    for scenario in catalog["scenarios"]:
        if scenario["name"] == name:
            return scenario, catalog["theory_options"]
    raise KeyError(name)


def test_theory_messages_catch_a_silenced_logger():
    """A logger set to CRITICAL cannot hide bfmt's error-to-unity message.

    evaluate_scenario switches the bfmt logger to WARNING and attaches the
    collecting handler for one evaluation; this repeats those steps on a
    differently named logger, the way a colored Cobaya logger behaves.
    """
    wrapper = notebook_functions("EXAMPLE_FEEDBACK_SCENARIOS.ipynb")

    logger = logging.getLogger("colored-bfmt-regression")
    logger.setLevel(logging.CRITICAL)
    before = list(logger.handlers)
    messages = wrapper.TheoryMessages()
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(messages)
    try:
        logger.error("emulator failed; using unity suppression")
    finally:
        logger.removeHandler(messages)
        logger.setLevel(previous_level)
    assert messages.messages == ["emulator failed; using unity suppression"]
    assert logger.level == logging.CRITICAL
    assert logger.handlers == before


def test_scenario_information_switches_power_only_feedback():
    """A scenario input adds bfmt and the key on the xi-only dataset, copying."""
    wrapper = notebook_functions("EXAMPLE_FEEDBACK_SCENARIOS.ipynb")

    information, point = example_information()
    scenario, options = catalog_entry(name="FLAMINGO")
    reference = wrapper.scenario_information(
        information=information, scenario=None, theory_options=options,
    )
    changed = wrapper.scenario_information(
        information=information, scenario=scenario, theory_options=options,
    )
    likelihood = changed["likelihood"][wrapper.LIKELIHOOD]
    assert likelihood["data_file"] == "des_y3_xi1.dataset"
    assert likelihood["external_baryon_suppression"] is True
    assert changed["theory"]["bfmt"]["baryon_model"] == 3
    assert changed["params"]["jet_frac_flamingo"] == 0.0
    assert "bfmt" not in reference["theory"]
    assert "external_baryon_suppression" not in reference["likelihood"][wrapper.LIKELIHOOD]
    assert "bfmt" not in information["theory"]

    # A feedback parameter already in the example would be overwritten.
    information["params"]["jet_frac_flamingo"] = 0.5
    with pytest.raises(ValueError, match="already a parameter"):
        wrapper.scenario_information(
            information=information, scenario=scenario, theory_options=options,
        )


def test_failed_scenario_leaves_the_gravity_only_rows_exact():
    """A rejected bfmt point raises, and the DMO model then reproduces its bits.

    Every Cobaya model shares cosmolike's native globals, so a scenario
    model that sets P_DMO * R_P, or fails halfway, must not change the
    gravity-only xi rows evaluated afterward with Cobaya's cache off.
    """
    wrapper = notebook_functions("EXAMPLE_FEEDBACK_SCENARIOS.ipynb")

    information, point = example_information()
    scenario, options = catalog_entry(name="BACCO")
    reference_info = wrapper.scenario_information(
        information=information, scenario=None, theory_options=options,
    )
    reference = wrapper.evaluate_scenario(info=reference_info, point=point)
    first = reference["xi"].copy()

    # A working scenario changes the shared native power...
    working = wrapper.evaluate_scenario(
        info=wrapper.scenario_information(
            information=information, scenario=scenario, theory_options=options,
        ),
        point=point,
    )
    assert np.max(np.abs(working["xi"] / first - 1.0)) > 0.01
    working["model"].close()

    # ...and a point outside the BACCO training box is rejected by bfmt.
    rejected = dict(scenario, parameters=dict(scenario["parameters"]))
    rejected["parameters"]["M_c_baccoemu"] = 17.0
    with pytest.raises(RuntimeError):
        wrapper.evaluate_scenario(
            info=wrapper.scenario_information(
                information=information, scenario=rejected, theory_options=options,
            ),
            point=point,
        )

    reference["model"].logposterior(params_values=point, cached=False)
    likelihood = reference["likelihood"]
    restored = likelihood.last_datavector[likelihood.xi_indices]
    np.testing.assert_array_equal(restored, first)
    reference["model"].close()


def test_flamingo_tail_hold_changes_only_small_scales():
    """The held-tail bfmt keeps R_P below the pivot and holds it above."""
    wrapper = notebook_functions("EXAMPLE_FEEDBACK_SCENARIOS.ipynb")

    information, point = example_information()
    scenario, options = catalog_entry(name="FLAMINGO")
    responses = []
    for tail in (False, True):
        run = wrapper.evaluate_scenario(
            info=wrapper.scenario_information(
                information=information, scenario=scenario,
                theory_options=options, tail=tail,
            ),
            point=point,
        )
        responses.append(run["response"])
        k = 10.0**run["likelihood"].log10k_interp_2D
        run["model"].close()

    # Table nodes within one request cell (7.5 percent) of the pivot mix the
    # two sides after the h shift; away from them the responses are identical
    # below the pivot and constant above it.
    below = k < 0.9 * wrapper.TAIL_PIVOT
    above = k > 1.1 * wrapper.TAIL_PIVOT
    np.testing.assert_array_equal(responses[1][:, below], responses[0][:, below])
    held = responses[1][:, above]
    np.testing.assert_allclose(held, held[:, :1] * np.ones_like(held), rtol=1.0e-14)
    assert not np.allclose(responses[1][:, above], responses[0][:, above])


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
