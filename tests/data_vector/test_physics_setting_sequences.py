"""Notebook state transitions compared bit for bit with fresh interpreters.

A sequence worker retains physical models and projection tables while changing
inputs, dimensions, IA choices, interpolation grids and worker counts. A fresh
worker evaluates each target configuration alone. Both call the same public
adapters; no function, cache or module is replaced by the test. Separate child
processes also isolate the core's deliberate fatal-input convention from pytest.

The cases follow cosmolike-dev/references/state_invalidation.md: dimensions
exercise allocation sizes; changed values exercise the contents of prepared
arrays; destruction followed by reconstruction exercises reset/lifetime; and
output shape/value assertions establish that each intended path was reached.
These isolated 3pcf objects have no initial_setup global reset. Their matching
reset operation is release of the Python owner, which frees its C arrays. The
fresh-worker comparison remains the same as the existing data-vector tests.
"""

import argparse
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[2]
FAMILIES = ("dimensions", "physical_values", "readers", "alignment", "grids",
            "threads_and_ownership")


def base_case():
    """Small physical grids retain every active calculation used below."""
    return {
        "label": "baseline",
        "physical": {
            "nk": 32,
            "nz": 13,
            "nlos": 7,
            "nbin": 2,
            "kernel_nz": 9,
            "fit_nz": 9,
            "table_boost": 2,
            "source_readers": False,
            "variant": "baseline",
        },
        "ia": {
            "a1": 0.3,
            "eta1": 0.2,
            "a2": 0.5,
            "eta2": -0.3,
            "bta": 0.4,
            "ct": 0.2,
            "fb": 0.0,
            "power_mode": 1,
            "ct_mode": 0,
            "suppress_only": 0,
        },
        "grid": {
            "ell_min": 1.0,
            "ell_max": 1.0e5,
            "epmu": 1.0e-7,
            "nr": 5,
            "nu": 5,
            "nv": 5,
            "np": 5,
            "nshape": 9,
            "npu_log": 2,
            "npu_linear": 4,
            "nell": 5,
            "npsi": 7,
            "npsi_log": 2,
            "npsi_linear": 5,
            "nmu_log": 2,
            "nmu_linear": 5,
            "lmax": 3,
            "reference_mode": 0,
            "table_boost": 2,
            "multipole_boost": 2,
            "prepare_multipoles": 1,
        },
        "bins": None,
        "threads": 1,
        "mutate": False,
    }


def changed_case(label, section, values):
    """Start from an independent configuration and change named settings."""
    case = base_case()
    case["label"] = label
    case[section].update(values)
    return case


def sequence_cases(family):
    """Every named transition has a visible numerical or shape reach check."""
    cases = [base_case()]
    if family == "dimensions":
        cases.append(changed_case("grow", "physical", {
            "nk": 48, "nz": 21, "nlos": 13, "nbin": 3,
            "kernel_nz": 17, "fit_nz": 13,
        }))
        cases.append(changed_case("shrink", "physical", {
            "nk": 20, "nz": 9, "nlos": 5, "nbin": 1,
            "kernel_nz": 5, "fit_nz": 5,
        }))
    elif family == "physical_values":
        for name in ("k", "z", "linear", "nonlinear", "distance",
                     "jacobian", "growth", "sources", "source_order",
                     "cosmology"):
            cases.append(changed_case(name, "physical", {"variant": name}))
    elif family == "readers":
        cases.append(changed_case("coarse_readers", "physical", {
            "table_boost": 1, "fit_nz": 5,
        }))
        cases.append(changed_case("source_readers", "physical", {
            "source_readers": True,
        }))
        cases.append(changed_case("dense_readers", "physical", {
            "table_boost": 4, "fit_nz": 17,
        }))
    elif family == "alignment":
        cases[0]["ia"].update(a2=0.0, bta=0.0, ct=0.0)
        cases[0]["label"] = "pure_nla"
        cases.append(base_case())
        changes = (
            ("a1", 0.6), ("eta1", 0.5), ("a2", 0.9), ("eta2", 0.4),
            ("bta", 0.8), ("ct", 0.5), ("c1rho", 0.015),
            ("zpivot", 0.8), ("power_mode", 0), ("power_mode", 2),
            ("ct_mode", 1), ("fb", 0.2),
        )
        for name, value in changes:
            cases.append(changed_case(f"{name}_{value}", "ia", {name: value}))
        cases.append(changed_case("capped_baryons", "ia", {
            "fb": 0.2, "suppress_only": 1,
        }))
        reordered = base_case()
        reordered["label"] = "ordered_bins"
        reordered["bins"] = [1, 0, 0]
        cases.append(reordered)
        cases.append(copy.deepcopy(cases[0]))
    elif family == "grids":
        for name, value in (
            ("nr", 7), ("nu", 7), ("nv", 7), ("np", 7), ("nshape", 13),
            ("nell", 7), ("npsi", 11), ("lmax", 4), ("table_boost", 3),
            ("multipole_boost", 3), ("ell_min", 0.1), ("ell_max", 2.0e5),
            ("epmu", 1.0e-5), ("nmu_log", 3), ("nmu_linear", 7),
            ("prepare_multipoles", 0),
        ):
            cases.append(changed_case(f"{name}_{value}", "grid", {name: value}))
        cases.append(changed_case("source_grid", "grid", {"reference_mode": 1}))
        for name, value in (("npu_log", 3), ("npu_linear", 6),
                            ("npsi_log", 3), ("npsi_linear", 7)):
            cases.append(changed_case(f"source_{name}", "grid", {
                "reference_mode": 1, name: value,
            }))
    elif family == "threads_and_ownership":
        for count in (2, 1, 3, 1):
            case = base_case()
            case["label"] = f"threads_{count}"
            case["threads"] = count
            cases.append(case)
        case = base_case()
        case["label"] = "mutated_inputs_and_results"
        case["mutate"] = True
        cases.append(case)
    else:
        raise ValueError(f"Unknown setting family: {family}")
    cases.append(copy.deepcopy(cases[0]))
    cases[-1]["label"] = "return_to_first"
    rebuilt = copy.deepcopy(cases[0])
    rebuilt["label"] = "destroy_and_rebuild"
    rebuilt["release_all"] = True
    cases.append(rebuilt)
    return cases


def physical_inputs(specification):
    """Return smooth test tables in the model's declared h-based units.

    Individual table perturbations test input identity; they are not a
    replacement cosmological prediction. Every source still vanishes at
    z=0, every grid stays ordered, and the background scalars remain flat.
    """
    k = np.geomspace(start=1.0e-5, stop=1.0e3, num=specification["nk"])
    z = np.linspace(start=0.0, stop=2.0, num=specification["nz"])
    linear = 2.0e4 * k / (1.0 + (k / 0.15)**2)**1.8
    nonlinear = (linear[None, :] * (1.0 + (k[None, :] / 0.35)**2)**0.22
                 / (1.0 + z[:, None])**2.4)
    sources = np.empty(shape=(specification["nbin"], z.size))
    for source in range(specification["nbin"]):
        sources[source] = z**2 * np.exp(-(z / (0.55 + 0.3 * source))**1.5)
    inputs = {
        "k": k,
        "p_linear0": linear,
        "z": z,
        "p_nonlinear": nonlinear,
        "chi": 3000.0 * z / (1.0 + 0.35 * z),
        "dz_dchi": (1.0 + 0.35 * z)**2 / 3000.0,
        "growth": 1.0 / (1.0 + z),
        "sources": sources,
        "omega_m": 0.3,
        "omega_de": 0.7,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.965,
        "sigma8": 0.8,
        "los_z": np.linspace(start=0.02, stop=1.9, num=specification["nlos"]),
        "kernel_nz": specification["kernel_nz"],
        "fit_nz": specification["fit_nz"],
        "table_boost": specification["table_boost"],
        "source_readers": specification["source_readers"],
    }
    variant = specification["variant"]
    if variant == "k":
        inputs["k"] = k * np.exp(0.03 * np.sin(np.arange(k.size)))
    elif variant == "z":
        inputs["z"] = z + 0.025 * np.sin(np.pi * z / 2.0)
    elif variant == "linear":
        inputs["p_linear0"] *= 1.0 + 0.12 * np.tanh(np.log(k / 0.2))
    elif variant == "nonlinear":
        inputs["p_nonlinear"] *= 1.0 + 0.1 / (1.0 + (k[None, :] / 0.4)**2)
    elif variant == "distance":
        inputs["chi"] *= 1.04
        inputs["dz_dchi"] /= 1.04
    elif variant == "jacobian":
        inputs["dz_dchi"] *= 1.0 + 0.06 * z
    elif variant == "growth":
        inputs["growth"] /= (1.0 + z)**0.12
    elif variant == "sources":
        inputs["sources"] *= 1.0 + 0.2 * z[None, :]
    elif variant == "source_order":
        inputs["sources"] = sources[::-1].copy()
    elif variant == "cosmology":
        inputs.update(omega_m=0.31, omega_de=0.69, w0=-0.9, wa=0.1,
                      ns=0.98, sigma8=0.83)
    return inputs


def notebook_buffers(inputs):
    """Give the notebook adapter negative-stride, separately owned views."""
    converted = {}
    for name, value in inputs.items():
        if isinstance(value, np.ndarray):
            shape = tuple(2 * length for length in value.shape)
            storage = np.empty(shape=shape, dtype=value.dtype)
            slices = tuple(slice(None, None, -2) for _ in value.shape)
            view = storage[slices]
            view[...] = value
            converted[name] = view
        else:
            converted[name] = value
    return converted


def evaluate(model, grid, case, adapter):
    """Evaluate paths whose dimensions and values expose the tested state."""
    method_suffix = "_notebook" if adapter == "notebook" else ""
    sides = np.asarray([[0.13, 0.21, 0.27], [2.0, 3.0, 4.0]])
    redshift = np.asarray([0.2, 0.8])
    angular = np.asarray([[90.0, 140.0, 190.0], [3.0e4, 4.0e4, 5.0e4]])
    last = case["physical"]["nbin"] - 1
    bins = np.asarray(case["bins"] if case["bins"] is not None
                      else [0, last, 0], dtype=np.int32)
    ia = dict(case["ia"])
    record = {}
    for mode in (0, 1, 2):
        record[f"power_{mode}"] = model.power(
            k=np.asarray([0.007, 0.4, 12.0]),
            z=np.asarray([0.1, 0.6, 1.4]),
            mode=mode,
        )
    for component in range(4):
        record[f"matter_{component}"] = getattr(model, "matter" + method_suffix)(
            sides=sides, redshifts=redshift, component=component, fb=ia["fb"]
        )
    for subtract in (False, True):
        record[f"ia_{subtract}"] = getattr(model, "ia" + method_suffix)(
            sides=sides, redshifts=redshift, parameters=ia,
            remove_alignment=subtract,
        )
    for component in (0, 1, 2):
        record[f"projected_{component}"] = getattr(
            model, "projected" + method_suffix
        )(sides=angular, bins=bins, parameters=ia, component=component)
        record[f"grid_{component}"] = getattr(
            grid, "projected" + method_suffix
        )(sides=angular, component=component)
        if case["grid"]["prepare_multipoles"]:
            record[f"multipoles_{component}"] = getattr(
                grid, "multipoles" + method_suffix
            )(ell=np.asarray([80.0, 190.0]), component=component)
    ell = np.asarray([80.0, 190.0])
    phi = (np.arange(4) + 0.5) * np.pi / 2.0
    aperture = {
        "ell": ell,
        "dell": np.asarray([70.0, 130.0]),
        "phi": phi,
        "dphi": np.full(shape=4, fill_value=np.pi / 2.0),
        "filters": np.asarray([[0.005, 0.007, 0.006]]),
    }
    record["aperture_direct"] = getattr(
        model, "aperture_full" + method_suffix
    )(**aperture, bins=bins, parameters=ia)
    record["aperture_grid"] = getattr(grid, "aperture_full" + method_suffix)(
        **aperture
    )
    nla = {"a1": ia["a1"], "eta1": ia["eta1"]}
    record["nla_power"] = getattr(model, "nla_power" + method_suffix)(
        ell=ell, bins=np.asarray([0, last], dtype=np.int32), parameters=nla
    )
    for name, values in model.radial().items():
        record["radial_" + name] = values
    for name, values in grid.axes().items():
        record["axis_" + name] = values
    for name, values in record.items():
        assert values.size > 0 and np.isfinite(values).all(), name
    assert record["ia_True"].shape == (2, 19)
    assert record["radial_lens"].shape == (last + 1, case["physical"]["nlos"])
    if ia["a2"] == 0.0 and ia["bta"] == 0.0 and ia["ct"] == 0.0:
        assert np.all(record["projected_2"] == 0.0)
        assert np.all(record["grid_2"] == 0.0)
    else:
        assert np.any(record["projected_2"] != 0.0)
    if case["grid"]["prepare_multipoles"]:
        assert record["multipoles_0"].shape == (case["grid"]["lmax"] + 1, 2, 2)
    return record


def equal_bits(first, second):
    """Compare shape, dtype and every byte, including signed floating zero."""
    assert set(first) == set(second)
    for name in first:
        assert first[name].shape == second[name].shape, name
        assert first[name].dtype == second[name].dtype, name
        assert first[name].tobytes(order="C") == second[name].tobytes(order="C"), name


def worker(family, adapter, output, index):
    """Retain old objects only in the sequence leg; a fresh leg sees one case."""
    sys.path.insert(0, str(PROJECT / "interface"))
    import cosmolike_des_3pcf_interface as compiled

    cases = sequence_cases(family)
    selected = list(range(len(cases))) if index is None else [index]
    models = {}
    retained = None
    for step in selected:
        case = cases[step]
        if case.get("release_all", False):
            # Remove every Python owner, including the retained model/grid.
            # gc.collect finishes collection before the next C allocations.
            models.clear()
            retained = None
            model = None
            inputs = None
            gc.collect()
        compiled.set_omp_threads(nthreads=case["threads"])
        key = json.dumps(case["physical"], sort_keys=True)
        if key not in models:
            inputs = physical_inputs(case["physical"])
            if adapter == "notebook":
                inputs = notebook_buffers(inputs)
                model = compiled.notebook.PhysicsModel(**inputs)
            else:
                model = compiled.production.PhysicsModel(**inputs)
            models[key] = (model, inputs)
        model, inputs = models[key]
        last = case["physical"]["nbin"] - 1
        bins = np.asarray(case["bins"] if case["bins"] is not None
                          else [0, last, 0], dtype=np.int32)
        parameters = copy.deepcopy(case["ia"])
        settings = copy.deepcopy(case["grid"])
        method = "projection_grid"
        if adapter == "notebook":
            method = "projection_grid_notebook"
        grid = getattr(model, method)(bins=bins, parameters=parameters, settings=settings)
        result = evaluate(model=model, grid=grid, case=case, adapter=adapter)
        if case["mutate"]:
            for value in inputs.values():
                if isinstance(value, np.ndarray):
                    value[...] = np.nan
            bins[:] = 1000
            parameters.clear()
            settings.clear()
            for values in model.radial().values():
                values[...] = np.nan
            for values in grid.axes().values():
                values[...] = np.nan
            for values in evaluate(model, grid, case, adapter).values():
                values[...] = np.nan
            equal_bits(result, evaluate(model, grid, case, adapter))
        if retained is None:
            retained = (model, grid, copy.deepcopy(case), result)
        else:
            old_model, old_grid, old_case, old_result = retained
            equal_bits(old_result, evaluate(old_model, old_grid, old_case, adapter))
            del old_model, old_grid, old_case, old_result
        destination = output / f"step_{step:03d}.npz" if index is None else output
        np.savez(file=destination, **result)
        # The first grid remains alive through retained; other completed
        # grids are destroyed before the next grid allocation is requested.
        del grid
    print(f"reached {family} {adapter} steps={len(selected)}")


def run_child(arguments):
    """Return only after a worker succeeds and prints its reach marker."""
    result = subprocess.run(
        args=[sys.executable, "-B", str(Path(__file__)), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    message = result.stdout + result.stderr
    assert result.returncode == 0, message
    assert "reached " in result.stdout, message
    assert "runtime error:" not in message


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("family", FAMILIES)
def test_physical_state_sequence_matches_fresh_process(tmp_path, adapter, family):
    """Every compared state must match its own clean-interpreter result."""
    reused = tmp_path / "reused"
    reused.mkdir()
    common = ["--family", family, "--adapter", adapter]
    run_child(common + ["--output", str(reused)])
    cases = sequence_cases(family)
    previous = None
    first = None
    for step, case in enumerate(cases):
        fresh = tmp_path / f"fresh_{step:03d}.npz"
        run_child(common + ["--index", str(step), "--output", str(fresh)])
        with np.load(file=reused / f"step_{step:03d}.npz") as stored:
            actual = {name: stored[name] for name in stored.files}
        with np.load(file=fresh) as stored:
            expected = {name: stored[name] for name in stored.files}
        equal_bits(actual, expected)
        if first is None:
            first = actual
        if (case["label"] in ("return_to_first", "destroy_and_rebuild")
                or family == "threads_and_ownership"):
            equal_bits(actual, first)
        elif previous is not None:
            # A changed setting must reach a calculation or an actual saved
            # axis, not merely appear in a configuration dictionary.
            common_outputs = set(actual).intersection(previous)
            changed = set(actual) != set(previous)
            for name in common_outputs:
                if (actual[name].shape != previous[name].shape
                        or actual[name].tobytes() != previous[name].tobytes()):
                    changed = True
            assert changed, case["label"]
        previous = actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--adapter", required=True, choices=("production", "notebook"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", type=int)
    arguments = parser.parse_args()
    worker(family=arguments.family, adapter=arguments.adapter,
           output=arguments.output, index=arguments.index)
