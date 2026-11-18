"""Compare long-lived transform/aperture state with fresh Python processes.

A notebook repeatedly changes grids and samples without restarting its C
extension. Agreement with a newly constructed object in that same process
alone would miss a shared stale table. Each reference below therefore comes
from a separate interpreter that performs only the requested calculation.
Every compared numerical element must agree bit for bit.

The audit distinguishes retained geometry from supplied data:

  State or input                 Change exercised                 Read checked
  FFT buffers/Mellin kernels      n, padding, orders, biases       two_bessel
  Coupling and geometric phases  L/M, npsi, phi, xy, conventions  four Gamma
  Selected inverse cells         index values and count           theta/Gamma
  Immutable plan geometry        caller arrays/dict overwritten   old Gamma
  Private transform scratch      new b_L then original b_L        Gamma/Map3
  Stateless aperture routines    side/filter/sample sizes/values  weights/Map3
  Worker capacity                create at 1/2; reuse at 2/1      all outputs
  Returned array ownership       metadata/output overwritten      later outputs

The notebook adapter copies strided, read-only arrays through Armadillo.
Its natural-components call uses the shared production.TransformPlan
constructor; there is no second notebook geometry implementation. Existing
physical-reference tests establish equations and signs. These tests isolate
state, allocation/refill coverage and array ownership on small grids.
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[2]


def configuration(case):
    """Return one declared transform geometry and the output it must affect.

    Each named scalar change is made relative to the same baseline. The
    large and small cases change allocation dimensions in opposite
    directions. No result-dependent refinement occurs in this test.
    """
    values = {
        "nell": 12,
        "lmax": 3,
        "mmax": 3,
        "indices": [2, 5, 8],
        "phi": [0.43, 1.11, 2.03],
        "ell_scale": 1.0,
        "nu1": 1.01,
        "nu2": 1.01,
        "xy": 1.0,
        "dlnt": 0.09,
        "padding": 0,
        "window": True,
        "window_width": 0.5,
        "npsi": 12,
        "dphi": 0.0,
        "projection": "centroid",
        "convention": "fastnc",
        "ordered_legs": True,
        "nfilter": 3,
        "nsample": 2,
        "filter_scale": 1.0,
        "order1": 2,
        "order2": 3,
        "query_shift": 0.0,
        "threads": 1,
    }
    changes = {
        "baseline": {},
        "grow": {
            "nell": 24,
            "lmax": 5,
            "mmax": 6,
            "indices": [2, 7, 12, 16],
            "phi": [0.3, 0.7, 1.1, 1.5, 2.1],
            "padding": 4,
            "nfilter": 4,
            "nsample": 3,
            "order1": 6,
            "order2": 7,
            "threads": 2,
        },
        "shrink": {
            "nell": 8,
            "lmax": 0,
            "mmax": 0,
            "indices": [3],
            "phi": [0.8],
            "npsi": 0,
            "nfilter": 1,
            "nsample": 1,
            "order1": 0,
            "order2": 0,
        },
        "ell_values": {"ell_scale": 1.17},
        "selected_values": {"indices": [1, 4, 7]},
        "selected_count": {"indices": [2, 7]},
        "phi_values": {"phi": [0.51, 1.23, 2.11]},
        "nu1": {"nu1": 0.81},
        "nu2": {"nu2": 1.34},
        "xy": {"xy": 1.23},
        "dlnt": {"dlnt": 0.15},
        "padding": {"padding": 2},
        "window": {"window": False},
        "window_width": {"window_width": 0.8},
        "npsi": {"npsi": 0},
        "dphi": {"dphi": 0.2},
        "projection": {"projection": "centroid_main"},
        "x_projection": {"projection": "X"},
        "convention": {"convention": "treecorr"},
        "ordered_legs": {"ordered_legs": False},
        "legendre_count": {"lmax": 5},
        "angular_count": {"mmax": 5},
        "filters": {"filter_scale": 1.1},
        "filter_count": {"nfilter": 4},
        "sample_count": {"nsample": 3},
        "bessel_orders": {"order1": 5, "order2": 4},
        "coupling_queries": {"query_shift": 0.04},
    }
    values.update(changes[case])
    reach = "real"
    if case in ("filters", "filter_count"):
        reach = "weights_real"
    elif case == "sample_count":
        reach = "samples"
    elif case == "bessel_orders":
        reach = "bessel"
    elif case == "coupling_queries":
        reach = "coupling"
    return values, reach


def notebook_view(values):
    """Create a read-only strided view with the same numerical axis order."""
    shape = list(values.shape)
    shape[-1] *= 2
    storage = np.empty(shape, dtype=values.dtype)
    view = storage[..., ::2]
    view[...] = values
    view.setflags(write=False)
    return view


def adapter_array(values, adapter):
    """Borrow C-order arrays in production and test copies in notebooks."""
    if adapter == "notebook":
        return notebook_view(values=values)
    return np.ascontiguousarray(values)


def make_plan(compiled, settings):
    """Construct an owned plan and retain its caller's mutable arguments."""
    ell = np.geomspace(start=20.0, stop=2000.0, num=settings["nell"])
    ell *= settings["ell_scale"]
    arguments = {
        "ell": ell,
        "phi": np.asarray(settings["phi"], dtype=np.float64),
        "indices": np.asarray(settings["indices"], dtype=np.int32),
    }
    for name in (
        "lmax", "mmax", "nu1", "nu2", "xy", "dlnt", "padding",
        "window", "window_width", "npsi", "dphi", "projection",
        "convention", "ordered_legs",
    ):
        arguments[name] = settings[name]
    compiled.set_omp_threads(nthreads=settings["threads"])
    plan = compiled.production.TransformPlan(**arguments)
    return plan, arguments


def multipole_values(settings, variant):
    """Supply signed ordered b_L values, changing more than their amplitude."""
    count = settings["nell"]
    first = np.linspace(start=0.1, stop=1.0, num=count)
    second = np.linspace(start=0.2, stop=1.3, num=count)
    output = np.empty((settings["lmax"] + 1, count, count))
    for degree in range(settings["lmax"] + 1):
        shape = np.exp(-first[:, None] - 0.7 * second[None, :])
        shape *= np.cos((degree + 1) * first[:, None])
        shape += 0.11 * np.sin((degree + 2) * second[None, :])
        if variant:
            shape += 0.07 * first[:, None] * second[None, :] ** 2
        output[degree] = 1.0e-12 * shape / (degree + 1)
    return output


def natural(compiled, adapter, plan, multipoles):
    """Call the chosen C++ route while keeping the C geometry identical."""
    if adapter == "notebook":
        return compiled.notebook.natural_components(
            plan=plan, b_l=notebook_view(values=multipoles),
        )
    return plan.natural_components(b_l=multipoles)


def aperture_arguments(plan, settings):
    """Give finite weights the same radial lower edges and chosen midpoint."""
    centers = plan.theta * np.exp(0.5 * settings["dlnt"])
    radius = float(np.median(centers)) * settings["filter_scale"]
    filters = np.empty((settings["nfilter"], 3))
    for index in range(settings["nfilter"]):
        filters[index] = radius * np.array([
            0.8 + 0.1 * index,
            1.0 + 0.07 * index,
            1.2 + 0.13 * index,
        ])
    return {
        "d2": centers,
        "d3": centers.copy(),
        "phi": np.asarray(settings["phi"], dtype=np.float64),
        "dlnd": settings["dlnt"],
        "dphi": 0.2,
        "filters": filters,
    }


def weights(compiled, adapter, arguments):
    """Apply notebook copying to every array, retaining scalar widths."""
    supplied = dict(arguments)
    for name in ("d2", "d3", "phi", "filters"):
        supplied[name] = adapter_array(values=arguments[name], adapter=adapter)
    return getattr(compiled, adapter).aperture_weights(**supplied)


def plan_snapshot(compiled, adapter, plan, settings):
    """Collect independent owned outputs after repeated samples/weight calls.

    The alternative b_L is followed by the original b_L, so incomplete
    refill or an uncleared padded cell cannot remain hidden. The returned
    dictionary contains copies; later scratch writes must not change them.
    """
    multipoles = multipole_values(settings=settings, variant=False)
    real, imag = natural(compiled=compiled, adapter=adapter, plan=plan,
                         multipoles=multipoles)
    result = {
        "theta": plan.theta.copy(),
        "theta_full": plan.theta_full.copy(),
        "real": np.array(real, copy=True),
        "imag": np.array(imag, copy=True),
    }
    alternate = multipole_values(settings=settings, variant=True)
    other = natural(compiled=compiled, adapter=adapter, plan=plan,
                    multipoles=alternate)
    result["alternate_real"] = np.array(other[0], copy=True)
    result["alternate_imag"] = np.array(other[1], copy=True)
    assert_bits(actual=real, expected=result["real"], context="old real output")
    assert_bits(actual=imag, expected=result["imag"], context="old imag output")
    restored = natural(compiled=compiled, adapter=adapter, plan=plan,
                       multipoles=multipoles)
    np.testing.assert_array_equal(restored[0], real)
    np.testing.assert_array_equal(restored[1], imag)
    arguments = aperture_arguments(plan=plan, settings=settings)
    real_weight, imag_weight = weights(compiled=compiled, adapter=adapter,
                                      arguments=arguments)
    result["weights_real"] = np.array(real_weight, copy=True)
    result["weights_imag"] = np.array(imag_weight, copy=True)
    triangles = real_weight.shape[2]
    numbers = np.arange(settings["nsample"] * 4 * triangles)
    gamma_real = np.sin(0.17 * numbers).reshape(-1, 4, triangles)
    gamma_imag = np.cos(0.09 * numbers).reshape(-1, 4, triangles)
    routine = getattr(compiled, adapter).aperture_apply
    result["samples"] = routine(
        wr=adapter_array(values=real_weight, adapter=adapter),
        wi=adapter_array(values=imag_weight, adapter=adapter),
        gr=adapter_array(values=gamma_real, adapter=adapter),
        gi=adapter_array(values=gamma_imag, adapter=adapter),
    )
    result["replacement_samples"] = routine(
        wr=adapter_array(values=0.91 * real_weight, adapter=adapter),
        wi=adapter_array(values=-1.03 * imag_weight, adapter=adapter),
        gr=adapter_array(values=gamma_real + 0.13, adapter=adapter),
        gi=adapter_array(values=gamma_imag - 0.07, adapter=adapter),
    )
    if settings["projection"] != "X":
        result["combined"] = plan.aperture_from_multipoles(
            b_l=multipoles,
            weights_real=np.ascontiguousarray(real_weight),
            weights_imag=np.ascontiguousarray(imag_weight),
        )
        result["combined_replacement"] = plan.aperture_from_multipoles(
            b_l=alternate,
            weights_real=np.ascontiguousarray(0.91 * real_weight),
            weights_imag=np.ascontiguousarray(-1.03 * imag_weight),
        )
    # Returned objects must outlive a subsequent call and may be edited by
    # their owner without corrupting either input or plan-owned storage.
    real[...] = 7.0
    imag[...] = -3.0
    repeated = natural(compiled=compiled, adapter=adapter, plan=plan,
                       multipoles=multipoles)
    np.testing.assert_array_equal(repeated[0], result["real"])
    np.testing.assert_array_equal(repeated[1], result["imag"])
    real_weight[...] = 17.0
    imag_weight[...] = -11.0
    repeated_weights = weights(compiled=compiled, adapter=adapter,
                               arguments=arguments)
    np.testing.assert_array_equal(repeated_weights[0], result["weights_real"])
    np.testing.assert_array_equal(repeated_weights[1], result["weights_imag"])
    return result


def snapshot(compiled, adapter, settings, plan=None):
    """Include stateless FFT/coupling calls around the retained plan route."""
    if plan is None:
        plan, _ = make_plan(compiled=compiled, settings=settings)
    result = plan_snapshot(compiled=compiled, adapter=adapter, plan=plan,
                           settings=settings)
    ell = np.geomspace(start=20.0, stop=2000.0, num=settings["nell"])
    ell *= settings["ell_scale"]
    logarithm = np.log(ell / 100)
    integrand = np.outer(np.cos(logarithm), np.exp(-0.2 * logarithm**2))
    result["bessel"] = getattr(compiled, adapter).two_bessel(
        ell=adapter_array(values=ell, adapter=adapter),
        integrand=adapter_array(values=integrand, adapter=adapter),
        order1=settings["order1"],
        order2=settings["order2"],
        nu1=settings["nu1"],
        nu2=settings["nu2"],
        xy=settings["xy"],
        dlnt=settings["dlnt"],
        padding=settings["padding"],
        window=settings["window"],
        window_width=settings["window_width"],
    )
    queries = np.array([0.21, 0.47, 0.9, 1.1]) + settings["query_shift"]
    result["coupling"] = getattr(compiled, adapter).legendre_coupling(
        psi=adapter_array(values=queries, adapter=adapter),
        lmax=settings["lmax"],
        mmax=settings["mmax"],
        npsi=settings["npsi"],
    )
    return result


def aperture_snapshot(compiled, adapter, case):
    """Exercise independent first/second-side lengths and sample/filter axes."""
    sizes = {
        "aperture_base": (2, 3, 3, 3, 2),
        "aperture_grow": (4, 5, 4, 4, 3),
        "aperture_shrink": (1, 2, 1, 1, 1),
        "aperture_values": (2, 3, 3, 3, 2),
        "aperture_widths": (2, 3, 3, 3, 2),
    }
    nd2, nd3, nphi, nfilter, nsample = sizes[case]
    filters = np.empty((nfilter, 3))
    for index in range(nfilter):
        filters[index] = (1 + 0.1 * index) * np.array([0.7, 0.9, 1.2])
    arguments = {
        "d2": np.geomspace(start=0.8, stop=1.5, num=nd2),
        "d3": np.geomspace(start=0.7, stop=1.7, num=nd3),
        "phi": np.linspace(start=0.4, stop=2.1, num=nphi),
        "dlnd": 0.12,
        "dphi": 0.2,
        "filters": filters,
    }
    for name in ("d2", "d3", "filters"):
        arguments[name] *= 1.0e-3
    if case == "aperture_values":
        arguments["d2"] *= 1.07
        arguments["d3"] *= 0.93
        arguments["phi"] += 0.08
        arguments["filters"][:, 1] *= 1.1
    elif case == "aperture_widths":
        arguments["dlnd"] = 0.18
        arguments["dphi"] = 0.31
    wr, wi = weights(compiled=compiled, adapter=adapter, arguments=arguments)
    result = {
        "weights_real": np.array(wr, copy=True),
        "weights_imag": np.array(wi, copy=True),
    }
    triangles = nd2 * nd3 * nphi
    numbers = np.arange(nsample * 4 * triangles, dtype=np.float64)
    real = np.sin(0.23 * numbers).reshape(nsample, 4, triangles)
    imag = np.cos(0.13 * numbers).reshape(nsample, 4, triangles)
    routine = getattr(compiled, adapter).aperture_apply
    supplied = {
        "wr": adapter_array(values=wr, adapter=adapter),
        "wi": adapter_array(values=wi, adapter=adapter),
        "gr": adapter_array(values=real, adapter=adapter),
        "gi": adapter_array(values=imag, adapter=adapter),
    }
    old_output = routine(**supplied)
    result["aperture"] = np.array(old_output, copy=True)
    supplied["gr"] = adapter_array(values=real + 0.17, adapter=adapter)
    supplied["gi"] = adapter_array(values=-0.91 * imag, adapter=adapter)
    result["new_samples"] = routine(**supplied)
    assert_bits(actual=old_output, expected=result["aperture"],
                context="aperture output survives new sample")
    old_output[:] = 100.0
    supplied["gr"] = adapter_array(values=real, adapter=adapter)
    supplied["gi"] = adapter_array(values=imag, adapter=adapter)
    restored = routine(**supplied)
    assert_bits(actual=restored, expected=result["aperture"],
                context="aperture output mutation is isolated")
    arguments["d3"] *= 1.03
    weights(compiled=compiled, adapter=adapter, arguments=arguments)
    assert_bits(actual=wr, expected=result["weights_real"],
                context="old weights survive another geometry")
    assert_bits(actual=wi, expected=result["weights_imag"],
                context="old weights survive another geometry")
    return result


def fresh_snapshot(adapter, case, directory):
    """Calculate exactly one case in an interpreter without previous plans."""
    output = directory / (adapter + "_" + case + ".npz")
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    process = subprocess.run(
        args=[sys.executable, "-B", str(Path(__file__)),
              "--snapshot", adapter, case, str(output)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "runtime error:" not in process.stderr
    with np.load(output, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def assert_snapshot(actual, expected, context):
    """Check every element, shape and dtype, naming the retained-state case."""
    assert set(actual) == set(expected), context
    for name in expected:
        assert actual[name].dtype == expected[name].dtype, (context, name)
        assert_bits(actual=actual[name], expected=expected[name],
                    context=f"{context}: {name}")


def assert_bits(actual, expected, context):
    """Compare float64 bit patterns, including the sign of an exact zero."""
    assert actual.shape == expected.shape, context
    assert actual.dtype == expected.dtype == np.dtype(np.float64), context
    first = np.ascontiguousarray(actual).view(np.uint64)
    second = np.ascontiguousarray(expected).view(np.uint64)
    np.testing.assert_array_equal(first, second, err_msg=context)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_transform_settings_against_fresh_process(compiled, adapter, tmp_path):
    """Grow/shrink and change retained geometry without restarting Python."""
    cases = (
        "baseline",
        "grow",
        "shrink",
        "ell_values",
        "selected_values",
        "selected_count",
        "phi_values",
        "nu1",
        "nu2",
        "xy",
        "dlnt",
        "padding",
        "window",
        "window_width",
        "npsi",
        "dphi",
        "projection",
        "x_projection",
        "convention",
        "ordered_legs",
        "legendre_count",
        "angular_count",
        "filters",
        "filter_count",
        "sample_count",
        "bessel_orders",
        "coupling_queries",
    )
    retained = None
    baseline = None
    try:
        for case in cases:
            settings, reach = configuration(case=case)
            reference = fresh_snapshot(adapter=adapter, case=case,
                                       directory=tmp_path)
            plan, _ = make_plan(compiled=compiled, settings=settings)
            actual = snapshot(compiled=compiled, adapter=adapter,
                              settings=settings, plan=plan)
            assert_snapshot(actual=actual, expected=reference,
                            context=adapter + ":" + case)
            if baseline is None:
                baseline = reference
                retained = plan
            else:
                # A change that never reaches any numerical output cannot
                # demonstrate cache invalidation, even if two runs agree.
                assert not np.array_equal(actual[reach], baseline[reach]), case
                original, _ = configuration(case="baseline")
                old = plan_snapshot(compiled=compiled, adapter=adapter,
                                    plan=retained, settings=original)
                for name in old:
                    assert_bits(actual=old[name], expected=baseline[name],
                                context="old plan: " + case)
    finally:
        compiled.set_omp_threads(nthreads=1)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_plan_ownership_and_thread_setting_sequences(
    compiled, adapter, tmp_path,
):
    """Interleave old plans, caller mutation and 1->2->1 worker changes."""
    base, _ = configuration(case="baseline")
    reference = fresh_snapshot(adapter=adapter, case="baseline",
                               directory=tmp_path)
    one_worker, arguments = make_plan(compiled=compiled, settings=base)
    two_settings = dict(base)
    two_settings["threads"] = 2
    two_workers, _ = make_plan(compiled=compiled, settings=two_settings)
    # The constructor copies numerical geometry and consumes scalar values.
    # Both arrays and the caller's options dictionary can subsequently change.
    arguments["ell"] *= 1.31
    arguments["phi"] += 0.07
    arguments["indices"][:] = [1, 3, 7]
    arguments["nu1"] = 1.73
    arguments["window"] = False
    arguments["lmax"] = 8
    public_theta = one_worker.theta
    public_full = one_worker.theta_full
    public_theta[:] = 100.0
    public_full[:] = 200.0
    try:
        for workers in (1, 2, 1):
            compiled.set_omp_threads(nthreads=workers)
            for plan in (two_workers, one_worker, two_workers):
                actual = snapshot(compiled=compiled, adapter=adapter,
                                  settings=base, plan=plan)
                assert_snapshot(actual=actual, expected=reference,
                                context=f"{adapter}: workers={workers}")
    finally:
        compiled.set_omp_threads(nthreads=1)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_aperture_setting_sequences(compiled, adapter, tmp_path):
    """Vary independent side/filter/sample dimensions and physical values."""
    cases = (
        "aperture_base",
        "aperture_grow",
        "aperture_shrink",
        "aperture_values",
        "aperture_widths",
    )
    baseline = None
    try:
        for case in cases:
            reference = fresh_snapshot(adapter=adapter, case=case,
                                       directory=tmp_path)
            for workers in (1, 2, 1):
                compiled.set_omp_threads(nthreads=workers)
                actual = aperture_snapshot(compiled=compiled, adapter=adapter,
                                           case=case)
                assert_snapshot(actual=actual, expected=reference,
                                context=f"{adapter}:{case}:{workers}")
            if baseline is None:
                baseline = reference
            else:
                assert not np.array_equal(actual["weights_real"],
                                          baseline["weights_real"]), case
    finally:
        compiled.set_omp_threads(nthreads=1)


def write_fresh_snapshot(adapter, case, output):
    """Import once, calculate one requested state, then exit the child."""
    sys.path.insert(0, str(PROJECT / "interface"))
    import cosmolike_des_3pcf_interface as compiled

    compiled.set_omp_threads(nthreads=1)
    if case.startswith("aperture_"):
        result = aperture_snapshot(compiled=compiled, adapter=adapter,
                                   case=case)
    else:
        settings, _ = configuration(case=case)
        result = snapshot(compiled=compiled, adapter=adapter, settings=settings)
    np.savez(file=output, **result)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--snapshot":
        raise SystemExit("Use --snapshot ADAPTER CASE OUTPUT.npz.")
    write_fresh_snapshot(adapter=sys.argv[2], case=sys.argv[3],
                         output=Path(sys.argv[4]))
