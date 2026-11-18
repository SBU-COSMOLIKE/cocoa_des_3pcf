"""Check ownership of the smallest angular-mode schedules bit for bit.

An angular mode M contributes to four natural shear components. The
ordered-leg transform calculates negative and positive M separately;
the historical transform reconstructs negative modes from transposes.
Modes M=+/-1 share some inverse transforms, so their scheduling differs
from the other modes. Cutoffs zero, one and two exercise the boundaries
between an empty mode pair, the shared pair and additional ordinary jobs.

These tests reuse the asymmetric multipole fixture and notebook adapter
helper from the existing transform tests. They compare execution order,
scratch ownership and component conventions, not a new physical formula.
The independent FFT and Gaussian-moment references remain in the original
transform tests. Every comparison here includes the float64 sign bit.
Ordinary checks request at most four OpenMP workers; BLAS remains serial
under the project's test environment.
"""

import numpy as np
import pytest

from test_transform import small_transform_inputs
from test_transform_setting_sequences import natural


def assert_component_bits(actual, expected, context):
    """Compare all real and imaginary component entries, including signed zero.

    Arguments:
      actual, expected = (real, imag) tuples, each array float64[4,3,3,3]
          for the shared small fixture's components, two sides and angles.
      context = description identifying the worker or reuse state on failure.
    Raises:
      AssertionError when any float64 bit pattern differs.
    """
    for part, name in enumerate(("real", "imag")):
        np.testing.assert_array_equal(
            x=actual[part].view(dtype=np.uint64),
            y=expected[part].view(dtype=np.uint64),
            err_msg=f"{context}: {name} natural components changed bits",
        )


@pytest.mark.parametrize("mmax", [0, 1, 2])
@pytest.mark.parametrize("ordered_legs", [False, True])
@pytest.mark.parametrize("convention", ["fastnc", "treecorr"])
@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_small_mode_schedule_and_reuse(
    compiled, mmax, ordered_legs, convention, adapter
):
    """Independent worker capacities and reused plans retain identical outputs.

    A separate serial plan supplies frozen copies of both input results.
    The three candidate plans are constructed for one, two and four workers,
    then interleaved while their input multipoles change and return. The
    notebook route supplies read-only strided arrays through its existing
    helper, exercising copies around the same retained C plan.

    Arguments:
      compiled = project extension supplied by the existing session fixture.
      mmax = angular cutoff zero, one or two; lmax remains three.
      ordered_legs = explicit signed modes or historical transpose assembly.
      convention = fastnc or TreeCorr component ordering and signs.
      adapter = borrowed production input or copied notebook input.
    Raises:
      AssertionError for a changed output bit or an input change with no effect.
    Side effects:
      Changes OpenMP's requested worker count and restores it to one even
      after a failed comparison. No files, timings or survey data are used.
    """
    ell, phi, indices, multipoles = small_transform_inputs()
    arguments = {
        "ell": ell,
        "phi": phi,
        "indices": indices,
        "lmax": multipoles.shape[0] - 1,
        "mmax": mmax,
        "npsi": 0,
        "projection": "X",
        "convention": convention,
        "ordered_legs": ordered_legs,
    }
    # Reversing only ell1 changes triangle dependence, rather than merely
    # scaling a cached answer. A copy retains the production layout contract.
    changed = np.ascontiguousarray(multipoles[:, ::-1, :])
    try:
        compiled.set_omp_threads(nthreads=1)
        reference_plan = compiled.production.TransformPlan(**arguments)
        references = []
        for values in (multipoles, changed):
            result = reference_plan.natural_components(b_l=values)
            # Own both reference arrays: later scratch reuse must not alter
            # the comparison target, even if an output-ownership bug appears.
            references.append((result[0].copy(), result[1].copy()))
        changed_output = False
        for part in range(2):
            changed_output = changed_output or not np.array_equal(
                a1=references[0][part].view(dtype=np.uint64),
                a2=references[1][part].view(dtype=np.uint64),
            )
        assert changed_output, "The changed multipoles must reach the shear outputs."

        plans = []
        for workers in (1, 2, 4):
            compiled.set_omp_threads(nthreads=workers)
            plan = compiled.production.TransformPlan(**arguments)
            plans.append((workers, plan))
            actual = natural(
                compiled=compiled, adapter=adapter, plan=plan, multipoles=multipoles,
            )
            assert_component_bits(
                actual=actual, expected=references[0],
                context=f"{adapter}, {workers} workers, independent plan",
            )

        for workers, plan in plans:
            actual = natural(
                compiled=compiled, adapter=adapter, plan=plan, multipoles=changed,
            )
            assert_component_bits(
                actual=actual, expected=references[1],
                context=f"{adapter}, {workers} workers, changed multipoles",
            )
        # Each retained plan owns its worker capacity. Reducing the global
        # request must not reinterpret that plan's scratch layout or M slots.
        compiled.set_omp_threads(nthreads=1)
        for workers, plan in reversed(plans):
            actual = natural(
                compiled=compiled, adapter=adapter, plan=plan, multipoles=multipoles,
            )
            assert_component_bits(
                actual=actual, expected=references[0],
                context=f"{adapter}, {workers} workers, original multipoles restored",
            )
    finally:
        compiled.set_omp_threads(nthreads=1)
