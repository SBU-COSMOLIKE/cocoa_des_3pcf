"""Separate immutable matter preparation from each query's IA parameters.

The model owns power, background, source and optional four-dimensional
bispectrum tables. IA amplitudes and feedback amplitude belong to a query;
changing them must change the prediction without preparing those tables
again. Identity and build-generation checks measure that reuse directly.
The controlled inputs below test state handling, not survey convergence.
"""

import ctypes
import gc
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1] / "external_modules/code/cosmolike_core"
# The existing feedback fixture imports its neighboring helper by filename.
# Include that directory so this file also runs alone, outside full collection.
sys.path.insert(0, str(PROJECT / "tests/data_vector"))

from data_vector.test_feedback import paired_input
from data_vector.test_physics_model import compact_grid_settings
from data_vector.test_physics_model import physical_inputs
from test_supplied_baseline import baseline_input, install_bridge


# Unequal angular multipoles and alternating source bins prevent accidental
# triangle/source symmetries from removing a changed IA coefficient's effect.
ANGULAR_SIDES = np.array(object=[[120.0, 180.0, 250.0]])
# Matter wave numbers are in h/Mpc and lie inside the supplied response table.
MATTER_SIDES = np.array(object=[[0.3, 0.4, 0.5]])
TRIPLET = np.array(object=[0, 1, 0], dtype=np.int32)


def install_alignment(*, core, changed=None, linear_only=False):
    """This helper installs nonzero power-law intrinsic alignments.

    Every coefficient has a nonzero effect in the chosen test. Changing one
    coefficient therefore tests fresh physics as well as Python identity.

    Arguments:
        core = compiled module's process-global core bindings.
        changed = coefficient name to increase by 0.15, or None.
        linear_only = set quadratic and counterterm coefficients to zero.
    Returns:
        None. Updates the process-global intrinsic-alignment state.
    """
    values = {
        "a1": 0.4,
        "eta1": -0.2,
        "a2": 0.7,
        "eta2": 0.3,
        "bta": 0.9,
        "ct": 0.2,
    }
    if linear_only:
        values["a2"] = 0.0
        values["bta"] = 0.0
        values["ct"] = 0.0
    if changed is not None:
        values[changed] += 0.15
    core.init_IA(ia_model=1, ia_redshift_evolution=3)
    core.set_nuisance_ia(
        A1=np.array(object=[values["a1"], values["eta1"]]),
        A2=np.array(object=[values["a2"], values["eta2"]]),
        B_TA=np.array(object=[values["bta"], 0.0]),
    )
    core.set_nuisance_ia_ct(ct=values["ct"])


def install_state(*, compiled, supplied=False):
    """This helper installs a controlled background and optional absolute B.

    A complete state prevents reuse from an earlier test from hiding a missing
    dependency. The constant supplied bispectrum tests ownership rather than
    the accuracy of an empirical matter model.

    Arguments:
        compiled = extension selected by the test session.
        supplied = use a supplied constant B instead of the internal fit.
    Returns:
        The core binding after replacing its process-global input state.
    """
    inputs = physical_inputs()
    install_bridge(compiled=compiled, inputs=inputs, install_scalars=True)
    core = compiled.core
    if supplied:
        core.set_baseline_bispectrum(
            baseline=baseline_input(varying=False),
            omega_m_z=np.full(shape=inputs["z"].shape, fill_value=0.3),
            sigma8=0.8,
        )
    install_alignment(core=core)
    return core


def projected(*, model, parameters):
    """This helper reads one ordered, non-equilateral projected bispectrum.

    Keeping the source order fixed avoids averaging away an IA response.

    Arguments:
        model = immutable owner of matter, distance and source tables.
        parameters = query's intrinsic-alignment and feedback dictionary.
    Returns:
        Array of length one containing the projected angular bispectrum.
    """
    return model.projected(
        sides=ANGULAR_SIDES, bins=TRIPLET, parameters=parameters,
    )


@pytest.fixture(autouse=True)
def reset_after_case(compiled):
    """This fixture clears global inputs even after an assertion fails.

    Pytest resumes the code after yield during cleanup. A failure can therefore
    leave no source or response override for the following test.

    Arguments:
        compiled = extension selected by the test session.
    Returns:
        Yields no value, then resets the process-global core state.
    """
    yield
    compiled.core.initial_setup()


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("changed", ["a1", "eta1", "a2", "eta2", "bta", "ct"])
def test_alignment_changes_reuse_model(compiled, supplied, changed):
    """This test changes IA from A to B and back to A using one model.

    An unchanged owner alone could conceal an old query dictionary. Requiring
    a changed prediction at B and exact recovery at A tests both contracts.

    Arguments:
        compiled = extension selected by the test session.
        supplied = select constant supplied B or the internal matter fit.
        changed = one IA coefficient to vary while the other inputs stay fixed.
    Returns:
        None; assertions require model reuse and fresh query values.
    """
    core = install_state(compiled=compiled, supplied=supplied)
    first = core.model_from_core()
    generation = core.model_generation()
    parameters = core.ia_parameters_from_core()
    before = projected(model=first, parameters=parameters)
    install_alignment(core=core, changed=changed)
    current = core.ia_parameters_from_core()
    assert current != parameters
    assert core.model_from_core() is first
    assert core.model_generation() == generation
    after = projected(model=first, parameters=current)
    assert not np.array_equal(before, after)
    np.testing.assert_array_equal(
        projected(model=first, parameters=parameters), before,
    )
    install_alignment(core=core)
    assert core.ia_parameters_from_core() == parameters
    assert core.model_from_core() is first
    np.testing.assert_array_equal(
        projected(model=first, parameters=core.ia_parameters_from_core()),
        before,
    )


@pytest.mark.parametrize("changed", ["power_mode", "ct_mode", "suppress_only"])
def test_query_prescriptions_reuse_model(compiled, changed):
    """This test changes a query selector without rebuilding common tables.

    All three power prescriptions are already prepared. A selector chooses
    which table to read; it is not an input to their preparation.

    Arguments:
        compiled = extension selected by the test session.
        changed = power_mode, ct_mode or suppress_only selector.
    Returns:
        None; assertions require reuse and the expected observable response.
    """
    core = install_state(compiled=compiled, supplied=True)
    first = core.model_from_core()
    generation = core.model_generation()
    before_parameters = core.ia_parameters_from_core()
    before = projected(model=first, parameters=before_parameters)
    settings = {
        "power_mode": 1,
        "ct_mode": 0,
        "suppress_only": 0,
    }
    if changed == "power_mode":
        settings[changed] = 2
    else:
        settings[changed] = 1
    core.set_prescriptions(**settings)
    current = core.ia_parameters_from_core()
    assert current[changed] == settings[changed]
    assert core.model_from_core() is first
    assert core.model_generation() == generation
    after = projected(model=first, parameters=current)
    if changed == "suppress_only":
        # This selector caps the legacy feedback transfer. With fb=0 it
        # has no physical contribution, so equality is the expected result.
        np.testing.assert_array_equal(after, before)
    else:
        assert not np.array_equal(after, before)
    core.set_prescriptions(power_mode=1, ct_mode=0, suppress_only=0)
    assert core.model_from_core() is first
    np.testing.assert_array_equal(
        projected(model=first, parameters=core.ia_parameters_from_core()),
        before,
    )


@pytest.mark.parametrize("supplied", [False, True])
def test_feedback_amplitude_and_old_grid(compiled, supplied):
    """This test changes response amplitude and retains an old projected grid.

    In pure NLA every term uses the same constant B response, so multiplying
    the original projection by 0.6**amplitude gives an independent expectation.
    The grid stores an earlier projection and must survive resets unchanged.

    Arguments:
        compiled = extension selected by the test session.
        supplied = select constant supplied B or the internal matter fit.
    Returns:
        None; assertions check amplitude dependence and retained ownership.
    """
    core = install_state(compiled=compiled, supplied=supplied)
    install_alignment(core=core, linear_only=True)
    response = dict(paired_input().native)
    core.set_feedback(feedback=response, amplitude=0.0)
    first = core.model_from_core()
    generation = core.model_generation()
    parameters = core.ia_parameters_from_core()
    before = projected(model=first, parameters=parameters)
    grid = first.projection_grid(
        bins=TRIPLET, parameters=parameters,
        settings=compact_grid_settings(reference_mode=0),
    )
    grid_before = grid.projected(sides=ANGULAR_SIDES)
    for amplitude in (0.7, -0.3, 0.0):
        core.set_feedback(feedback=response, amplitude=amplitude)
        current = core.ia_parameters_from_core()
        assert current["feedback_amplitude"] == amplitude
        assert core.model_from_core() is first
        assert core.model_generation() == generation
        actual = projected(model=first, parameters=current)
        np.testing.assert_allclose(
            actual=actual, desired=before * 0.6**amplitude, rtol=5.0e-14,
        )
        np.testing.assert_array_equal(grid.projected(sides=ANGULAR_SIDES),
                                      grid_before)
    np.testing.assert_array_equal(actual, before)
    core.set_feedback(feedback={}, amplitude=0.0)
    assert core.model_from_core() is not first
    core.initial_setup()
    np.testing.assert_array_equal(projected(model=first, parameters=parameters),
                                  before)
    del first
    gc.collect()
    np.testing.assert_array_equal(grid.projected(sides=ANGULAR_SIDES), grid_before)


@pytest.mark.parametrize("changed", ["ns", "sigma8", "w0", "wa", "omega_m"])
def test_physical_scalars_still_rebuild(compiled, changed):
    """This test preserves model invalidation for physical scalar changes.

    Reducing the key to actual model dependencies must retain cosmological
    inputs. Older model owners must also keep their previous numerical arrays.

    Arguments:
        compiled = extension selected by the test session.
        changed = ns, sigma8, w0, wa or omega_m scalar.
    Returns:
        None; assertions require new owners and exact A/B/A recovery.
    """
    core = install_state(compiled=compiled)
    first = core.model_from_core()
    generation = core.model_generation()
    before = first.matter(sides=MATTER_SIDES, redshifts=np.array(object=[0.4]))
    settings = {
        "ns": 0.965,
        "sigma8": 0.8,
        "w0": -1.0,
        "wa": 0.0,
    }
    if changed == "omega_m":
        core.set_cosmological_parameters(
            omega_matter=0.31, omega_baryon=0.05, hubble=70.0,
            omega_nu_h2=0.0,
        )
    else:
        settings[changed] += 0.01
        core.set_scalars(**settings)
    assert core.model_from_core() is not first
    assert core.model_generation() == generation + 1
    np.testing.assert_array_equal(
        first.matter(sides=MATTER_SIDES, redshifts=np.array(object=[0.4])), before,
    )
    if changed == "omega_m":
        core.set_cosmological_parameters(
            omega_matter=0.3, omega_baryon=0.05, hubble=70.0,
            omega_nu_h2=0.0,
        )
    else:
        core.set_scalars(ns=0.965, sigma8=0.8, w0=-1.0, wa=0.0)
    restored = core.model_from_core()
    assert core.model_generation() == generation + 2
    np.testing.assert_array_equal(
        restored.matter(sides=MATTER_SIDES, redshifts=np.array(object=[0.4])), before,
    )


@pytest.mark.parametrize("changed", [
    "z", "logk", "logu", "v", "log_power", "log_bispectrum", "boost", "clamp",
])
def test_feedback_table_changes_still_rebuild(compiled, changed):
    """This test replaces each part of a prepared feedback definition.

    Unlike its amplitude, a table axis, value, boundary rule or upsampling
    factor affects table preparation and must produce a different owner.

    Arguments:
        compiled = extension selected by the test session.
        changed = response array name, boost or clamp control.
    Returns:
        None; assertions require new owners and exact A/B/A recovery.
    """
    core = install_state(compiled=compiled, supplied=True)
    response = dict(paired_input().native)
    core.set_feedback(feedback=response, amplitude=0.4)
    first = core.model_from_core()
    generation = core.model_generation()
    parameters = core.ia_parameters_from_core()
    before = projected(model=first, parameters=parameters)
    replacement = dict(response)
    if changed == "boost":
        replacement[changed] = 3
    elif changed == "clamp":
        replacement[changed] = 0
    elif changed == "logu":
        replacement[changed] = response[changed] * 1.01
    elif changed == "v":
        replacement[changed] = response[changed] * 0.99
    else:
        replacement[changed] = response[changed] + 0.01
    core.set_feedback(feedback=replacement, amplitude=0.4)
    second = core.model_from_core()
    assert second is not first
    assert core.model_generation() == generation + 1
    assert core.model_from_core() is second
    np.testing.assert_array_equal(projected(model=first, parameters=parameters),
                                  before)
    core.set_feedback(feedback=response, amplitude=0.4)
    restored = core.model_from_core()
    assert restored is not second
    np.testing.assert_array_equal(projected(model=restored, parameters=parameters),
                                  before)


# This helper uses the actual public C types. Python never invents a struct
# layout or replaces a native function. A native caller is allowed to send
# the bridge's borrowed arrays back with a changed upsampling factor; that
# call must finish copying all inputs before it frees the old allocation.
ALIAS_SOURCE = r"""
#include <stddef.h>
#include "cosmolike/3pcf/threepcf_core_bridge.h"

int resubmit_feedback_view(
  const int boost, // new dense-grid ratio; must differ from the installed one
  const int count, // capacity of the concatenated output array
  double* output // copied axes, log P response and log B response
)
{
  const physics_input_3pcf* input = threepcf_bridge_refresh();
  if (NULL == input
      || NULL == input->feedback
      || NULL == output) {
    return 1;
  }
  const feedback_input_3pcf values = *input->feedback;
  const int expected = values.nz + values.nk + values.nu + values.nv
    + values.nz*values.nk + values.nz*values.nk*values.nu*values.nv;
  if (count != expected
      || boost == values.boost) { return 2; }
  threepcf_bridge_set_feedback(values.z, values.nz, values.logk, values.nk,
    values.logu, values.nu, values.v, values.nv, values.log_power,
    values.log_bispectrum, boost, values.clamp, 0.3);
  input = threepcf_bridge_refresh();
  if (NULL == input
      || NULL == input->feedback) { return 3; }
  const feedback_input_3pcf* updated = input->feedback;
  if (boost != updated->boost) { return 4; }
  const int sizes[6] = {values.nz, values.nk, values.nu, values.nv,
    values.nz*values.nk, values.nz*values.nk*values.nu*values.nv};
  const double* arrays[6] = {updated->z, updated->logk, updated->logu,
    updated->v, updated->log_power, updated->log_bispectrum};
  int offset = 0;
  for (int field=0; field<6; field++) {
    for (int node=0; node<sizes[field]; node++) {
      output[offset++] = arrays[field][node];
    }
  }
  return 0;
}

int resubmit_source_view(
  const int count, // capacity of axis plus source-density outputs
  const double* sources, // replacement densities, one row per source bin
  double* output // copied source axis followed by all density rows
)
{
  const physics_input_3pcf* input = threepcf_bridge_refresh();
  if (NULL == input
      || NULL == input->source_z
      || NULL == sources
      || NULL == output) { return 1; }
  const int nsource = input->nsource;
  const int nbin = input->nbin;
  if (count != nsource + nbin*nsource) { return 2; }
  threepcf_bridge_set_sources(input->source_z, sources, nbin, nsource);
  input = threepcf_bridge_refresh();
  if (NULL == input) { return 3; }
  for (int node=0; node<nsource; node++) {
    output[node] = input->source_z[node];
  }
  for (int node=0; node<nbin*nsource; node++) {
    output[nsource + node] = input->sources[node];
  }
  return 0;
}

int resubmit_prefix_view(
  const int field, // 0 selects LOS nodes; 1 selects the pixel window
  const int count, // shorter valid prefix of the installed array
  double* output // copied prefix after setter replacement
)
{
  const physics_input_3pcf* input = threepcf_bridge_refresh();
  if (NULL == input
      || NULL == output
      || 2 > count) { return 1; }
  if (0 == field) {
    if (count >= input->nlos) { return 2; }
    threepcf_bridge_set_los(input->los_z, count);
  } else if (1 == field) {
    if (count >= input->nwindow) { return 2; }
    threepcf_bridge_set_pixel_window(input->window, count);
  } else { return 4; }
  input = threepcf_bridge_refresh();
  if (NULL == input) { return 3; }
  const double* values = 0 == field ? input->los_z : input->window;
  for (int node=0; node<count; node++) { output[node] = values[node]; }
  return 0;
}
"""


@pytest.fixture(scope="module")
def alias_helper(compiled, tmp_path_factory):
    """This fixture compiles a small caller using the public C types.

    The C compiler determines every struct offset. Python passes only scalars
    and array addresses, so a future public layout edit cannot cause a test
    to read an unrelated memory field through a guessed ctypes definition.

    Arguments:
        compiled = selected extension whose exported C functions are linked.
        tmp_path_factory = pytest helper creating a temporary build directory.
    Returns:
        ctypes library owner with three explicitly declared call signatures.
    Raises:
        FileNotFoundError if the configured C compiler is unavailable.
        CalledProcessError if compiling the observation helper fails.
    Side effects:
        Writes a small C source and shared library into pytest's temporary path.
    """
    compiler = os.environ.get("C_COMPILER", os.environ.get("CC", "cc"))
    command = shlex.split(s=compiler)
    if not command or shutil.which(cmd=command[0]) is None:
        raise FileNotFoundError(f"C compiler {compiler!r} is unavailable.")
    directory = tmp_path_factory.mktemp(basename="bridge_feedback_alias")
    source = directory / "caller.c"
    library = directory / "caller.so"
    source.write_text(ALIAS_SOURCE)
    command.extend(["-std=c99", "-O0", "-fPIC"])
    if sys.platform == "darwin":
        command.extend(["-dynamiclib", "-undefined", "dynamic_lookup"])
    else:
        command.append("-shared")
    command.extend(["-I", str(CORE), str(source), compiled.__file__,
                    "-o", str(library)])
    subprocess.run(args=command, check=True, capture_output=True, text=True)
    owner = ctypes.CDLL(name=str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    owner.resubmit_feedback_view.argtypes = [ctypes.c_int, ctypes.c_int, pointer]
    owner.resubmit_source_view.argtypes = [ctypes.c_int, pointer, pointer]
    owner.resubmit_prefix_view.argtypes = [ctypes.c_int, ctypes.c_int, pointer]
    owner.resubmit_feedback_view.restype = ctypes.c_int
    owner.resubmit_source_view.restype = ctypes.c_int
    owner.resubmit_prefix_view.restype = ctypes.c_int
    return owner


def test_native_feedback_view_survives_new_boost(compiled, alias_helper):
    """This test resubmits borrowed native feedback arrays with a new boost.

    A setter may receive the arrays it currently owns. It must finish copying
    them before releasing that allocation; changing the boost forces this
    replacement even when every table value is unchanged.

    Arguments:
        compiled = extension selected by the test session.
        alias_helper = compiled C caller that reads the public input structure.
    Returns:
        None; assertions compare every copied axis and response value exactly.
    """
    core = install_state(compiled=compiled, supplied=True)
    response = dict(paired_input().native)
    # Nonconstant rows expose copying the wrong bytes even when every value
    # remains finite. The helper inspects inputs, not a physical projection.
    for name in ("log_power", "log_bispectrum"):
        count = response[name].size
        response[name] = np.linspace(start=-0.2, stop=-0.01, num=count)
    core.set_feedback(feedback=response, amplitude=0.8)
    names = ("z", "logk", "logu", "v", "log_power", "log_bispectrum")
    # Concatenation keeps the same six-array order as the typed C helper.
    fields = []
    for name in names:
        fields.append(response[name])
    # NumPy requires the sequence argument positionally in this version.
    expected = np.concatenate(fields)
    actual = np.empty_like(prototype=expected)
    function = alias_helper.resubmit_feedback_view
    pointer = ctypes.POINTER(ctypes.c_double)
    # C calls are positional. The C compiler checks its own public struct
    # and setter call; this Python signature contains only scalars/one array.
    status = function(3, actual.size, actual.ctypes.data_as(pointer))
    assert status == 0, f"Native feedback resubmission failed with status {status}."
    np.testing.assert_array_equal(actual, expected)
    assert core.ia_parameters_from_core()["feedback_amplitude"] == 0.3
    # The helper consumed refresh to inspect the input. Reset before a
    # Python owner factory is used; NULL on a cache hit remains the contract.
    core.initial_setup()


def test_native_source_axis_survives_density_replacement(compiled, alias_helper):
    """This test changes source densities while borrowing the source axis.

    The incoming redshift pointer still belongs to the old stored input. Both
    incoming arrays must be copied before either old allocation is released.

    Arguments:
        compiled = extension selected by the test session.
        alias_helper = compiled C caller that reads the public input structure.
    Returns:
        None; assertions compare the axis and all replacement densities.
    """
    core = install_state(compiled=compiled)
    inputs = physical_inputs()
    core.set_sources(source_z=inputs["z"], sources=inputs["sources"])
    # A 10% density change forces replacement; this is a memory check only.
    replacement = np.ascontiguousarray(a=inputs["sources"] * 1.1)
    replacement = replacement.ravel()
    expected = np.concatenate((inputs["z"], replacement))
    actual = np.empty_like(prototype=expected)
    pointer = ctypes.POINTER(ctypes.c_double)
    status = alias_helper.resubmit_source_view(
        actual.size, replacement.ctypes.data_as(pointer),
        actual.ctypes.data_as(pointer),
    )
    assert status == 0, f"Native source resubmission failed with status {status}."
    np.testing.assert_array_equal(actual, expected)
    core.initial_setup()


@pytest.mark.parametrize("field", ["los", "window"])
def test_native_shorter_prefix_survives_replacement(compiled, alias_helper, field):
    """This test replaces a stored array with its own shorter prefix.

    Changing the count forces reallocation. The retained prefix is still a
    borrowed view of the previous allocation until the setter copies it.

    Arguments:
        compiled = extension selected by the test session.
        alias_helper = compiled C caller that reads the public input structure.
        field = los for integration nodes, or window for the pixel window.
    Returns:
        None; assertions compare every retained prefix value exactly.
    """
    core = install_state(compiled=compiled)
    if field == "los":
        selector = 0
        values = physical_inputs()["los_z"]
    else:
        selector = 1
        values = np.linspace(start=1.0, stop=0.4, num=17)
        core.set_pixel_window(window=values)
    expected = values[:-1]
    actual = np.empty_like(prototype=expected)
    pointer = ctypes.POINTER(ctypes.c_double)
    status = alias_helper.resubmit_prefix_view(
        selector, actual.size, actual.ctypes.data_as(pointer),
    )
    assert status == 0, f"Native prefix resubmission failed with status {status}."
    np.testing.assert_array_equal(actual, expected)
    core.initial_setup()


def test_simultaneous_python_builds_share_one_owned_model(compiled):
    """Global-state preparation and copying must be one serialized operation.

    Both threads request the same uncached model. The bridge keeps Python's
    lock while reading global power and copying its scratch arrays, so only
    one assembly occurs; native OpenMP work remains available within it.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    core = install_state(compiled=compiled)
    before = core.model_generation()
    ready = threading.Barrier(parties=2)

    def build():
        ready.wait(timeout=20)
        return core.model_from_core()

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(build)
        second = workers.submit(build)
        left = first.result(timeout=60)
        right = second.result(timeout=60)
    assert left is right
    assert core.model_generation() == before + 1
