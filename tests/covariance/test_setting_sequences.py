"""Covariance setting changes in one Python process, checked bit for bit.

Each reference state is evaluated in a separate interpreter before it is
compared with an evolving notebook-style session. The small fixed rules
test state ownership and rebuilding, not integration accuracy or speed.
"""

from theory.bihalofit.variance import reference_physics_model

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss


def log_rule(lower, upper, count):
    """Return fixed positive nodes and weights for an integral over dx."""
    x, w = leggauss(deg=count)
    half = 0.5 * np.log(upper / lower)
    nodes = np.exp(0.5 * np.log(lower * upper) + half * x)
    return nodes, half * w * nodes


def model_inputs(version):
    """Two models differ in cosmology, source count and preparation sizes."""
    alternate = version == "b"
    wave = np.geomspace(1.0e-5, 1000.0, 72 if alternate else 64)
    z = np.linspace(0.0, 2.0, 29 if alternate else 25)
    linear = (2.3e4 if alternate else 2.0e4) * wave
    linear /= (1.0 + (wave / 0.15) ** 2) ** 1.8
    nonlinear = linear * (1.0 + (wave / 0.35) ** 2) ** 0.22
    sources = np.asarray([
        z**2 * np.exp(-(z / scale) ** 1.5)
        for scale in ([0.55, 0.85, 1.1] if alternate else [0.6, 0.95])
    ])
    omega_m = 0.32 if alternate else 0.3
    return {
        "k": wave,
        "p_linear0": linear,
        "z": z,
        "p_nonlinear": nonlinear[None, :] / (1.0 + z[:, None]) ** 2.4,
        "chi": 3000.0 * z / (1.0 + 0.35 * z),
        "dz_dchi": (1.0 + 0.35 * z) ** 2 / 3000.0,
        "growth": 1.0 / (1.0 + z),
        "sources": sources,
        "omega_m": omega_m,
        "omega_de": 1.0 - omega_m,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.965,
        "sigma8": 0.82 if alternate else 0.8,
        "los_z": np.asarray([0.2, 0.5, 1.0, 1.6] if alternate else [0.2, 0.7, 1.4]),
        "kernel_nz": 81 if alternate else 65,
    }


def base_inputs(size="base"):
    """Build a joint request whose dimensions can grow and shrink together.

    Every additional field in the larger request is used by an added row.
    The smaller request retains both an aperture and a shear observable.
    Theta nodes have normalized uniform-area weights within explicit edges.
    """
    sizes = {
        "base": (5, 8, 8, 18, 2, 5, 2, 96),
        "grow": (7, 11, 12, 26, 3, 7, 3, 112),
        "shrink": (4, 6, 6, 12, 1, 3, 1, 80),
    }
    nl, nw, nr, npf, nm, na, nt, nk = sizes[size]
    ell, dell = log_rule(12.0, 2400.0, nl)
    window, dwindow = log_rule(0.03, 700.0, nw)
    radius, dr = log_rule(1.0e-6, 0.015, nr)
    wave, dwave = log_rule(1.0e-5, 500.0, nk)
    mass, dmass = log_rule(1.0e11, 1.0e14, nm)
    phi = (np.arange(na) + 0.37) * (2.0 * np.pi / na)
    sources = [0, 1, 0, 1]
    radii = [0.002, 0.003, 0.0, 0.0]
    kinds = [1, 2, 3, 0]
    fields = [[0, 0, 1], [2, 3, 0], [2, 3, 0], [0, 1, 0]]
    if size == "grow":
        sources += [0, 1]
        radii += [0.003, 0.002]
        kinds += [1, 2]
        fields += [[5, 4, 0], [2, 2, 0]]
    elif size == "shrink":
        sources = [0, 1, 0]
        radii = [0.002, 0.003, 0.0]
        kinds = [1, 2, 0]
        fields = [[0, 0, 1], [2, 2, 0], [0, 1, 0]]
    theta = np.zeros(shape=(len(kinds), nt))
    theta_weight = np.zeros_like(theta)
    edges = np.zeros(shape=(len(kinds), 2))
    nodes, weights = leggauss(deg=nt)
    for row, kind in enumerate(kinds):
        if kind >= 2:
            edges[row] = [0.0015, 0.0035]
            theta[row] = 0.0025 + 0.001 * nodes
            theta_weight[row] = weights * theta[row]
            theta_weight[row] /= np.sum(theta_weight[row])
    return {
        "ell": ell,
        "dell": dell,
        "window_ell": window,
        "window_dell": dwindow,
        "phi": phi,
        "dphi": np.full(shape=na, fill_value=2.0 * np.pi / na),
        "radius": radius,
        "dr": dr,
        "profile_radius": np.concatenate(([0.0], np.geomspace(1.0e-7, 0.05, npf))),
        "profile_upsample": 3,
        "k": wave,
        "dk": dwave,
        "mass": mass,
        "dmass": dmass,
        "field_source": np.asarray(sources, dtype=np.int32),
        "field_radius": np.asarray(radii, dtype=np.float64),
        "row_kind": np.asarray(kinds, dtype=np.int32),
        "row_fields": np.asarray(fields, dtype=np.int32),
        "theta": theta,
        "theta_weight": theta_weight,
        "theta_edges": edges,
        "noise": np.diag([2.0e-9, 3.0e-9]),
        "area": 1.0e-5,
        "center_window": 1,
        "large_area": 1,
        "noise_mode": 0,
        "estimator": 0,
        "ia_model": 0,
        "st_a": 0.707,
        "st_p": 0.3,
        "st_norm": 0.3222,
        "delta_c": 1.686,
        "collapse_model": 0,
        "concentration_a": 10.14,
        "concentration_b": -0.081,
        "concentration_c": -1.01,
        "concentration_pivot": 2.0e12,
        "baryon_strength": 0.0,
    }


AXES = (
    "ell", "window_ell", "radius", "profile_radius", "k", "mass", "phi", "theta"
)
WEIGHTS = ("dell", "window_dell", "dr", "dk", "dmass", "dphi", "theta_weight")
SCALARS = {
    "area": 1.4e-5,
    "center_window": 2,
    "profile_upsample": 5,
    "st_a": 0.76,
    "st_p": 0.34,
    "st_norm": 0.35,
    "delta_c": 1.72,
    "collapse_model": 1,
    "concentration_a": 11.0,
    "concentration_b": -0.09,
    "concentration_c": -1.12,
    "concentration_pivot": 2.4e12,
    "baryon_strength": 0.7,
}
LABEL_STATES = ("noise", "source_labels", "aperture_radius", "row_fields", "row_kind")
SAME_SIZE = (
    tuple("axis_" + key for key in AXES)
    + tuple("weight_" + key for key in WEIGHTS)
    + tuple("scalar_" + key for key in SCALARS)
    + LABEL_STATES
    + ("finite", "analytic", "analytic_edges")
)
STATE_NAMES = ("base", "grow", "shrink", "pure_xi", "model_b") + SAME_SIZE


def state_inputs(name):
    """Apply one named change, leaving unrelated same-sized arrays intact."""
    values = base_inputs(size=name if name in ("grow", "shrink") else "base")
    if name.startswith("axis_"):
        key = name.removeprefix("axis_")
        if key == "phi":
            values[key] += 0.07 * (2.0 * np.pi / values[key].size)
        else:
            values[key] *= 1.035
    elif name.startswith("weight_"):
        key = name.removeprefix("weight_")
        if key == "theta_weight":
            values[key][1:3] = [0.6, 0.4]
        elif key == "dphi":
            values[key] *= np.linspace(0.9, 1.1, values[key].size)
            values[key] *= 2.0 * np.pi / np.sum(values[key])
        else:
            values[key] *= np.linspace(0.93, 1.09, values[key].size)
    elif name.startswith("scalar_"):
        key = name.removeprefix("scalar_")
        values[key] = SCALARS[key]
    elif name == "noise":
        values["noise"] *= 1.4
        values["noise"][0, 1] = values["noise"][1, 0] = 0.4e-9
    elif name == "source_labels":
        values["field_source"][0] = 1
    elif name == "aperture_radius":
        values["field_radius"][0] *= 1.12
    elif name == "row_fields":
        values["row_fields"][0] = [0, 1, 1]
    elif name == "row_kind":
        values["row_kind"][1] = 3
    elif name == "finite":
        values["large_area"] = 0
    elif name in ("analytic", "analytic_edges"):
        values["noise_mode"] = 1
        if name == "analytic_edges":
            # Hold the finite signal quadrature fixed to isolate the exact
            # NN edge dependence. Every node remains inside the annulus.
            values["theta_edges"][1:3] = [0.0014, 0.0036]
    elif name == "pure_xi":
        values["field_source"] = np.asarray([0, 1], dtype=np.int32)
        values["field_radius"] = np.zeros(shape=2)
        values["row_kind"] = np.asarray([2, 3], dtype=np.int32)
        values["row_fields"] = np.asarray([[0, 1, 0], [0, 1, 0]], dtype=np.int32)
        for key in ("theta", "theta_weight", "theta_edges"):
            values[key] = values[key][1:3].copy()
    elif name == "model_b":
        values["noise"] = np.diag([2.0e-9, 3.0e-9, 4.0e-9])
        values["field_source"][[1, 3]] = 2
    if name in ("axis_profile_radius", "scalar_profile_upsample"):
        # The large-area xi response has a direct Fourier zero mode; it
        # does not read interpolated endpoints. Exercise these controls in
        # the finite-window calculation where they change an actual result.
        values["large_area"] = 0
    return values


def evaluate(adapter, model, inputs):
    """Return named owned outputs and require finite scientific results."""
    mean, terms = adapter.physical_covariance(model=model, inputs=inputs)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(terms))
    return {"mean": mean, "terms": terms}


def assert_bits(actual, expected, context):
    """Compare all floating-point bits, including the sign bit of zero."""
    for key in expected:
        assert actual[key].shape == expected[key].shape, (context, key)
        assert actual[key].dtype == expected[key].dtype, (context, key)
        assert actual[key].tobytes() == expected[key].tobytes(), (context, key)


def origin_state(name):
    """Return a reference differing by exactly the setting under test."""
    if name == "analytic_edges":
        return "analytic"
    if name in ("axis_profile_radius", "scalar_profile_upsample"):
        return "finite"
    return "base"


def assert_reached(name, outputs, references):
    """A state that changes no output cannot demonstrate invalidation."""
    baseline = references[origin_state(name)]
    if name != "base":
        assert any(outputs[key].tobytes() != baseline[key].tobytes()
                   for key in outputs), name
    # These routes have a specific output family, independent of other
    # changed labels. Reaching it rules out a silent ignored option.
    selected = {
        "axis_profile_radius": 4,
        "scalar_profile_upsample": 4,
        "scalar_center_window": 5,
        "analytic": 0,
        "analytic_edges": 0,
    }
    if name in selected:
        term = selected[name]
        assert not np.array_equal(outputs["terms"][term], baseline["terms"][term]), name


@pytest.fixture(scope="module")
def fresh_references(tmp_path_factory):
    """Each adapter/state gets a new process with exactly one covariance call."""
    directory = tmp_path_factory.mktemp("covariance_fresh_states")
    references = {}
    for adapter_name in ("production", "notebook"):
        references[adapter_name] = {}
        for state in STATE_NAMES:
            destination = directory / f"{adapter_name}_{state}.npz"
            result = subprocess.run(
                args=[sys.executable, "-B", str(Path(__file__)),
                      adapter_name, state, str(destination)],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            with np.load(destination, allow_pickle=False) as archive:
                references[adapter_name][state] = {
                    key: archive[key].copy() for key in ("mean", "terms")
                }
        for state, outputs in references[adapter_name].items():
            assert_reached(state, outputs, references[adapter_name])
    return references


def dimension_signature(inputs):
    """Include every count that controls covariance workspace allocation."""
    keys = ("field_source", "row_kind", "ell", "radius", "window_ell",
            "profile_radius", "mass", "phi", "k")
    return tuple(inputs[key].size for key in keys) + (inputs["theta"].shape[1],)


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
def test_dimension_and_mode_round_trips(compiled, fresh_references, adapter_name):
    """Grow, shrink, remove cubic rows, and restore the original live model."""
    base = dimension_signature(state_inputs("base"))
    grown = dimension_signature(state_inputs("grow"))
    small = dimension_signature(state_inputs("shrink"))
    assert all(a < b for a, b in zip(base, grown))
    assert all(a > b for a, b in zip(base, small))
    model = reference_physics_model(factory=compiled.PhysicsModel, **model_inputs("a"))
    adapter = getattr(compiled, adapter_name)
    sequence = ("base", "grow", "shrink", "grow", "pure_xi", "finite",
                "analytic", "base", "pure_xi", "base")
    for name in sequence:
        result = evaluate(adapter, model, state_inputs(name))
        assert_bits(result, fresh_references[adapter_name][name], name)


def overwrite_same_shape(target, source):
    """Edit caller buffers in place, preserving every NumPy array identity."""
    for key, value in source.items():
        if isinstance(value, np.ndarray):
            assert target[key].shape == value.shape, key
            identity = id(target[key])
            np.copyto(target[key], value)
            assert id(target[key]) == identity
        else:
            target[key] = value


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
def test_individual_in_place_setting_changes(compiled, fresh_references, adapter_name):
    """Same-size edits must invalidate by value, not by pointer or dimensions."""
    model = reference_physics_model(factory=compiled.PhysicsModel, **model_inputs("a"))
    adapter = getattr(compiled, adapter_name)
    inputs = state_inputs("base")
    for name in SAME_SIZE + tuple(reversed(SAME_SIZE)) + ("base",):
        origin = origin_state(name)
        baseline = state_inputs(origin)
        requested = state_inputs(name)
        if name != "base":
            changed = [key for key in requested
                       if not np.array_equal(requested[key], baseline[key])]
            assert len(changed) == 1, (name, changed)
        overwrite_same_shape(inputs, baseline)
        prior = evaluate(adapter, model, inputs)
        assert_bits(prior, fresh_references[adapter_name][origin], origin)
        overwrite_same_shape(inputs, requested)
        result = evaluate(adapter, model, inputs)
        assert_bits(result, fresh_references[adapter_name][name], name)


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
def test_old_models_and_owned_arrays(compiled, fresh_references, adapter_name):
    """Old model objects, input buffers and returned arrays have separate lives."""
    constructor_a = model_inputs("a")
    constructor_b = model_inputs("b")
    first = reference_physics_model(factory=compiled.PhysicsModel, **constructor_a)
    second = reference_physics_model(factory=compiled.PhysicsModel, **constructor_b)
    # These remain valid arrays but no longer describe either owned model.
    for supplied in (constructor_a, constructor_b):
        supplied["p_linear0"] *= 1.07
        supplied["p_nonlinear"] *= 1.13
        supplied["sources"][0] = supplied["sources"][-1]
    adapter = getattr(compiled, adapter_name)
    originals = None
    for name, model in (("base", first), ("model_b", second), ("grow", first),
                        ("model_b", second), ("base", first)):
        output = evaluate(adapter, model, state_inputs(name))
        assert_bits(output, fresh_references[adapter_name][name], name)
        if originals is None:
            originals = output
        else:
            assert_bits(originals, fresh_references[adapter_name]["base"], "old output")
    inputs = state_inputs("base")
    if adapter_name == "notebook":
        # Two-dimensional arrays now have column-major storage. One-
        # dimensional arrays are strided views into independently owned pads.
        for key, value in list(inputs.items()):
            if isinstance(value, np.ndarray):
                if value.ndim == 2:
                    inputs[key] = np.asfortranarray(value)
                else:
                    backing = np.zeros(shape=2 * value.size, dtype=value.dtype)
                    backing[::2] = value
                    inputs[key] = backing[::2]
    owned = evaluate(adapter, first, inputs)
    overwrite_same_shape(inputs, state_inputs("noise"))
    assert_bits(owned, fresh_references[adapter_name]["base"], "input edit")
    changed = evaluate(adapter, first, inputs)
    assert_bits(changed, fresh_references[adapter_name]["noise"], "edited view")
    owned["mean"].fill(123.0)
    owned["terms"].fill(-456.0)
    restored = evaluate(adapter, first, state_inputs("base"))
    assert_bits(restored, fresh_references[adapter_name]["base"], "output edit")
    assert_bits(changed, fresh_references[adapter_name]["noise"], "other output")


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
def test_worker_count_changes(compiled, fresh_references, adapter_name):
    """Changing worker count preserves the prescribed integration order."""
    model = reference_physics_model(factory=compiled.PhysicsModel, **model_inputs("a"))
    adapter = getattr(compiled, adapter_name)
    try:
        for threads in (1, 2, 1, 2):
            compiled.set_omp_threads(nthreads=threads)
            for name in ("grow", "finite", "analytic", "shrink", "base"):
                actual = evaluate(adapter, model, state_inputs(name))
                assert_bits(actual, fresh_references[adapter_name][name], (threads, name))
    finally:
        compiled.set_omp_threads(nthreads=1)


def reference_worker(adapter_name, state, destination):
    """Load only the requested model and perform one isolated native call."""
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "interface"))
    import cosmolike_des_3pcf_interface as compiled

    compiled.set_omp_threads(nthreads=1)
    version = "b" if state == "model_b" else "a"
    model = reference_physics_model(factory=compiled.PhysicsModel, **model_inputs(version))
    output = evaluate(getattr(compiled, adapter_name), model, state_inputs(state))
    np.savez(destination, **output)


if __name__ == "__main__":
    reference_worker(adapter_name=sys.argv[1], state=sys.argv[2], destination=sys.argv[3])
