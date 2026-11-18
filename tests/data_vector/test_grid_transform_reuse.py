"""Check which state a retained projected-grid interpolation map may reuse.

The direct grid-to-aperture entry keeps geometric interpolation indices
inside a transform plan. Cosmology, source weights and IA can change while
those indices remain valid, but the sampled physical multipoles must always
be read again. Changed interpolation axes instead require a new map, even
when their sizes and endpoints agree. These two cases need separate tests.

The reference is the existing two-step calculation: grid.multipoles followed
by plan.aperture_from_multipoles. Every state uses a fresh reference plan;
selected states also run that route alone in fresh Python interpreters.
The controlled model and aperture fixtures come from the existing tests.
This file adds no physical formula or survey-accuracy claim. Comparisons
include every float64 bit, and ordinary tests request at most four OpenMP
workers with serial BLAS.
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_physics_setting_sequences import base_case, physical_inputs
from test_transform_setting_sequences import (
    aperture_arguments,
    configuration,
    make_plan,
    notebook_view,
)


PROJECT = Path(__file__).resolve().parents[2]
STATES = (
    "baseline",
    "ia",
    "nla",
    "bins",
    "sources",
    "cosmology",
    "physical_grid_boost",
    "bounds",
    "grow",
    "shrink",
    "multipole_boost",
    "reference",
    "reference_interior",
)
FRESH_STATES = ("baseline", "ia", "cosmology", "reference_interior")


def grid_case(state):
    """Specify one controlled state without changing the transform geometry."""
    case = base_case()
    case["bins"] = [0, 1, 0]
    if state == "ia":
        case["ia"]["a2"] = 0.9
    elif state == "nla":
        case["ia"].update(a2=0.0, bta=0.0, ct=0.0)
    elif state == "bins":
        case["bins"] = [1, 0, 0]
    elif state in ("sources", "cosmology"):
        case["physical"]["variant"] = state
    elif state == "physical_grid_boost":
        case["grid"]["table_boost"] = 3
    elif state == "bounds":
        case["grid"].update(ell_min=0.5, ell_max=2.0e5)
    elif state == "grow":
        case["grid"].update(nell=7, npsi=11)
    elif state == "shrink":
        case["grid"].update(nell=4, npsi=5)
    elif state == "multipole_boost":
        case["grid"]["multipole_boost"] = 3
    elif state == "reference":
        case["grid"]["reference_mode"] = 1
    elif state == "reference_interior":
        # Both reference psi axes contain seven nodes and share endpoints;
        # moving the logarithmic/linear join changes only interior values.
        case["grid"].update(reference_mode=1, npsi_log=3, npsi_linear=4)
    elif state == "lmax_mismatch":
        case["grid"]["lmax"] = 4
    elif state != "baseline":
        raise ValueError(f"Unknown grid-transform state: {state}")
    return case


def prepare_grid(compiled, state):
    """Build an independent physical model and its owned projected table."""
    case = grid_case(state=state)
    inputs = physical_inputs(specification=case["physical"])
    model = compiled.production.PhysicsModel(**inputs)
    return model.projection_grid(
        bins=np.asarray(case["bins"], dtype=np.int32),
        parameters=case["ia"],
        settings=case["grid"],
    )


def plan_and_weights(compiled, workers):
    """Use ten Fourier nodes with the established three-side/angle selection.

    Returns the plan, its mutable caller arguments and a (real, imag)
    weight tuple of shape [3,4,27]. The aperture radii use the same radial
    bin representatives as the existing transform ownership tests.
    """
    settings, _ = configuration(case="baseline")
    # Ten is even but not divisible by four: the transform's final two
    # Fourier samples must follow the same arithmetic as its main blocks.
    settings["nell"] = 10
    settings["threads"] = workers
    plan, arguments = make_plan(compiled=compiled, settings=settings)
    weight_arguments = aperture_arguments(plan=plan, settings=settings)
    weights = compiled.production.aperture_weights(**weight_arguments)
    return plan, arguments, weights


def old_route(compiled, grid):
    """Evaluate a fresh two-step plan without calling the retained-map entry."""
    plan, arguments, weights = plan_and_weights(compiled=compiled, workers=1)
    multipoles = grid.multipoles(ell=arguments["ell"], component=0)
    return plan.aperture_from_multipoles(
        b_l=multipoles, weights_real=weights[0], weights_imag=weights[1],
    )


def retained_route(compiled, adapter, plan, grid, weights):
    """Use borrowed production weights or read-only strided notebook copies."""
    if adapter == "notebook":
        return compiled.notebook.aperture_from_grid(
            plan=plan,
            grid=grid,
            weights_real=notebook_view(values=weights[0]),
            weights_imag=notebook_view(values=weights[1]),
        )
    return plan.aperture_from_grid(
        grid=grid, weights_real=weights[0], weights_imag=weights[1],
    )


def assert_bits(actual, expected, context):
    """Keep numerical diagnostics while distinguishing signed-zero changes."""
    np.testing.assert_array_equal(
        x=actual.view(dtype=np.uint64),
        y=expected.view(dtype=np.uint64),
        err_msg=context,
    )


@pytest.fixture(scope="module")
def grid_bank(compiled):
    """Retain small independent grids; only the tested plan shares query state."""
    compiled.set_omp_threads(nthreads=1)
    grids = {}
    for state in STATES:
        grids[state] = prepare_grid(compiled=compiled, state=state)
    return grids


@pytest.fixture(scope="module")
def old_outputs(compiled, grid_bank):
    """Freeze the existing route's answer before retained-plan mutations."""
    outputs = {}
    for state in STATES:
        outputs[state] = old_route(compiled=compiled, grid=grid_bank[state]).copy()
    return outputs


def run_child(mode, name, output=None):
    """Run a reference or deliberate refusal in a separate serial interpreter.

    The child receives its thread limits before importing NumPy or the C
    extension. Output files, when requested, belong to pytest's temporary
    directory. A refusal exits its child process without terminating pytest.
    """
    arguments = [sys.executable, "-B", str(Path(__file__)), mode, name]
    if output is not None:
        arguments.append(str(output))
    environment = os.environ.copy()
    for environment_name in (
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS", "OMP_NUM_THREADS",
    ):
        environment[environment_name] = "1"
    environment["OMP_THREAD_LIMIT"] = "4"
    return subprocess.run(
        args=arguments, env=environment, capture_output=True, text=True,
        check=False, timeout=90,
    )


@pytest.fixture(scope="module")
def fresh_outputs(tmp_path_factory):
    """Use fresh old-route processes for representative physical and axis keys."""
    directory = tmp_path_factory.mktemp(basename="grid_transform_references")
    outputs = {}
    for state in FRESH_STATES:
        destination = directory / f"{state}.npy"
        result = run_child(mode="reference", name=state, output=destination)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs[state] = np.load(file=destination, allow_pickle=False)
    return outputs


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_grid_map_reuses_geometry_and_refills_physics(
    compiled, grid_bank, old_outputs, fresh_outputs, adapter, workers
):
    """A/B/A changes agree bitwise with fresh predictions, without restarting.

    The input ell array is overwritten before the first lazy query creation.
    The plan must therefore use its exact owned copy. Every returned aperture
    array is also overwritten; it must not alias future interpolation or FFT
    scratch. Physical and geometric changes are followed by the original
    grid, checking refill and map reconstruction in both directions.
    """
    try:
        plan, arguments, weights = plan_and_weights(compiled=compiled, workers=workers)
        arguments["ell"][:] = np.nan
        sequence = []
        for state in STATES:
            sequence.append(state)
            if state != "baseline":
                sequence.append("baseline")
        # Consecutive irregular grids have identical sizes and endpoints.
        # This transition detects keys that omit their interior coordinates.
        sequence.extend(("reference", "reference_interior", "reference", "baseline"))
        for state in sequence:
            actual = retained_route(
                compiled=compiled, adapter=adapter, plan=plan,
                grid=grid_bank[state], weights=weights,
            )
            context = f"{adapter}, {workers} workers, grid={state}"
            assert_bits(actual=actual, expected=old_outputs[state], context=context)
            if state in fresh_outputs:
                assert_bits(actual=actual, expected=fresh_outputs[state], context=context)
            actual[:] = np.nan
        # Retained plans may have greater worker capacity than the current
        # request. The query map has no per-worker storage to reinterpret.
        compiled.set_omp_threads(nthreads=1)
        restored = retained_route(
            compiled=compiled, adapter=adapter, plan=plan,
            grid=grid_bank["baseline"], weights=weights,
        )
        assert_bits(actual=restored, expected=old_outputs["baseline"],
                    context="Retained worker capacity after global request changes")
    finally:
        compiled.set_omp_threads(nthreads=1)


def test_grid_changes_reach_physical_values_and_exact_axes(grid_bank, old_outputs):
    """Check that each named transition reaches the state it is meant to test."""
    baseline = grid_bank["baseline"].axes()
    for state in ("ia", "nla", "bins", "sources", "cosmology", "physical_grid_boost"):
        axes = grid_bank[state].axes()
        for name in ("log_ell", "log_psi"):
            assert_bits(actual=axes[name], expected=baseline[name], context=state)
        assert not np.array_equal(
            a1=old_outputs[state].view(dtype=np.uint64),
            a2=old_outputs["baseline"].view(dtype=np.uint64),
        ), f"The {state} change did not reach the aperture prediction."
    for state in ("bounds", "grow", "shrink", "multipole_boost", "reference"):
        axes = grid_bank[state].axes()
        unchanged = True
        for name in ("log_ell", "log_psi"):
            unchanged = unchanged and np.array_equal(
                a1=axes[name].view(dtype=np.uint64),
                a2=baseline[name].view(dtype=np.uint64),
            )
        assert not unchanged, f"The {state} change did not reach query geometry."
    first = grid_bank["reference"].axes()["log_psi"]
    second = grid_bank["reference_interior"].axes()["log_psi"]
    assert first.shape == second.shape
    assert_bits(actual=first[[0, -1]], expected=second[[0, -1]],
                context="Irregular-axis endpoints must agree for this test")
    assert not np.array_equal(a1=first[1:-1], a2=second[1:-1])


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_grid_lmax_mismatch_stops_before_transform(adapter):
    """A different physical L extent is refused before writing plan scratch."""
    result = run_child(mode="refusal", name=adapter)
    message = result.stdout + result.stderr
    assert result.returncode == 1
    assert "projected-grid multipoles mismatch the transform plan" in message
    assert "runtime error:" not in message
    assert "Traceback" not in message
    assert "Mismatched grid returned" not in message


def child_entry():
    """Select one fresh reference or invalid-input call in this child process."""
    sys.path.insert(0, str(PROJECT / "interface"))
    import cosmolike_des_3pcf_interface as compiled

    compiled.set_omp_threads(nthreads=1)
    mode = sys.argv[1]
    name = sys.argv[2]
    if mode == "reference":
        grid = prepare_grid(compiled=compiled, state=name)
        result = old_route(compiled=compiled, grid=grid)
        np.save(file=sys.argv[3], arr=result, allow_pickle=False)
    elif mode == "refusal":
        grid = prepare_grid(compiled=compiled, state="lmax_mismatch")
        plan, _, weights = plan_and_weights(compiled=compiled, workers=1)
        retained_route(compiled=compiled, adapter=name, plan=plan,
                       grid=grid, weights=weights)
        print("Mismatched grid returned")
    else:
        raise ValueError(f"Unknown child operation: {mode}")


if __name__ == "__main__":
    child_entry()
