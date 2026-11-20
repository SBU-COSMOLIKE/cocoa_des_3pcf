"""Extend two saved cosmological stencils without recomputing their old nodes.

The default calculation adds only +/-Delta/4 for Omega_m and Hubble h.
--larger-step also adds +/-4Delta, completing a derivative with spacing
2Delta. This requires exactly four or eight new physical means. Existing
Delta and Delta/2 vectors are read as arrays, never installed into caches.
All derived Fisher ratios replace these two columns of the fixed full
Delta/2 baseline; they are diagnostics, not complete refined forecasts.

Run without --execute first to inspect coordinates and dependency checks.
The default project is discovered from the current directory or this file's
location after installation under tests/studies. --project is explicit.
"""

import argparse
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    # A command-line child owns its pools. Importing these array helpers in
    # pytest must not change the caller's existing thread environment.
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "4"

import numpy as np


# The two selected physical parameters are matter density and Hubble h.
COLUMNS = ("omega_m", "h")
# Integers label offsets in units of the original absolute Delta divided
# by four. Existing Delta/Delta/2 archives therefore supply all even units.
SAVED_UNITS = {
    "h": (-8, -4, 0, 4, 8),
    "h2": (-4, -2, 0, 2, 4),
}
# Each row gives the five physical coordinates for a complete named stencil.
STENCIL_UNITS = {
    "h4": (-2, -1, 0, 1, 2),
    "h2": (-4, -2, 0, 2, 4),
    "h": (-8, -4, 0, 4, 8),
    "two_h": (-16, -8, 0, 8, 16),
}
# These factors convert the declared absolute Delta to each derivative step.
STEP_FACTORS = {
    "h4": 0.25,
    "h2": 0.5,
    "h": 1.0,
    "two_h": 2.0,
}


def new_point_plan(parameters, steps, bounds, larger_step=False):
    """Validate the entire coordinate union before requesting any prediction.

    Arguments:
      parameters = full physical reference dictionary, including both columns.
      steps = absolute Delta values in the original Fisher configuration.
      bounds = declared physical bounds; both requested columns require them.
      larger_step = also calculate the outer points +/-4Delta.
    Returns:
      Ordered records with name, integer quarter-step unit, and physical point.
    Raises:
      ValueError for nonfinite, collapsed or out-of-bounds coordinates.
    Side effects:
      None; neither the input dictionaries nor a physical model are modified.
    """
    units = [-1, 1]
    if larger_step:
        units.extend([-16, 16])
    plan = []
    for name in COLUMNS:
        reference = float(parameters[name])
        step = float(steps[name])
        if not np.isfinite(reference) or not np.isfinite(step) or step <= 0.0:
            raise ValueError(f"{name} needs a finite reference and positive step.")
        if name not in bounds:
            raise ValueError(f"Explicit bounds are required for {name}.")
        lower, upper = bounds[name]
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid bounds for {name}.")
        union = sorted(set(SAVED_UNITS["h"] + SAVED_UNITS["h2"] + tuple(units)))
        previous = None
        for unit in union:
            coordinate = reference + unit * (step / 4.0)
            if not np.isfinite(coordinate) or not lower <= coordinate <= upper:
                raise ValueError(f"{name} quarter-step unit {unit} leaves its bounds.")
            if previous is not None and coordinate <= previous:
                raise ValueError(f"{name} quarter-step coordinates collapse in float64.")
            previous = coordinate
        for unit in units:
            point = dict(parameters)
            point[name] = reference + unit * (step / 4.0)
            plan.append({"name": name, "quarter_step_unit": unit, "point": point})
    return plan


def reuse_nodes(stencil_h, stencil_h2):
    """Return owned vectors indexed by quarter-step units from two saved stencils.

    Shared coordinates must agree bit for bit, including signed zeros. This
    establishes actual reuse of the baseline's saved evaluations, not a new
    prediction that happens to be close to one of them.

    Arguments:
      stencil_h, stencil_h2 = saved finite [5,ndata] arrays in stencil order.
    Returns:
      Owned vectors indexed by their integer quarter-step coordinates.
    Raises:
      ValueError for malformed arrays or disagreeing shared coordinates.
    """
    first = np.asarray(a=stencil_h, dtype=np.float64)
    second = np.asarray(a=stencil_h2, dtype=np.float64)
    if (first.ndim != 2 or first.shape[0] != 5 or first.shape[1] == 0
            or second.shape != first.shape
            or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second))):
        raise ValueError("Saved stencils need finite matching [5,ndata] arrays.")
    nodes = {}
    for label, values in (("h", first), ("h2", second)):
        for unit, value in zip(SAVED_UNITS[label], values):
            if unit in nodes and not np.array_equal(
                nodes[unit].view(np.uint64), value.view(np.uint64),
            ):
                raise ValueError(f"Saved shared coordinate {unit} differs bitwise.")
            nodes[unit] = value.copy()
    return nodes


def evaluate_missing(plan, predict, ndata):
    """Call the ordinary predictor exactly once for each explicitly new point.

    Arguments:
      plan = validated new-point records from new_point_plan.
      predict = ordinary callable accepting a full physical parameter dictionary.
      ndata = required output vector length.
    Returns:
      A dictionary for each selected parameter, indexed by quarter-step unit.
      No center, saved node, artificial state key or cache entry is created.
    Raises:
      ValueError for a duplicate coordinate or invalid prediction vector.
    Side effects:
      Calls predict once per requested new point; its normal model state changes.
    """
    result = {name: {} for name in COLUMNS}
    for record in plan:
        name = record["name"]
        unit = record["quarter_step_unit"]
        if unit in result[name] or unit in SAVED_UNITS["h"] + SAVED_UNITS["h2"]:
            raise ValueError("The new-point plan repeats or reuses an old coordinate.")
        values = np.array(
            object=predict(dict(record["point"])), dtype=np.float64, copy=True,
        )
        if values.shape != (ndata,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid mean at {name}, quarter-step unit {unit}.")
        result[name][unit] = values
    return result


def assemble_stencil(nodes, new_values, label):
    """Assemble [-2,-1,0,1,2] at one named spacing from saved and new vectors.

    Arguments:
      nodes, new_values = dictionaries indexed by quarter-step integers.
      label = h4, h2, h or two_h, selecting the spacing and five coordinates.
    Returns:
      Owned finite [5,ndata] array; no prediction is requested.
    Raises:
      ValueError for a missing coordinate or attempted replacement of old data.
    """
    if label not in STENCIL_UNITS:
        raise ValueError("Choose h4, h2, h or two_h for the stencil spacing.")
    combined = dict(nodes)
    for unit, value in new_values.items():
        if unit in combined:
            raise ValueError("A new prediction cannot replace a saved baseline node.")
        combined[unit] = np.array(object=value, dtype=np.float64, copy=True)
    values = []
    for unit in STENCIL_UNITS[label]:
        if unit not in combined:
            raise ValueError(f"Missing quarter-step unit {unit} for {label}.")
        values.append(combined[unit])
    result = np.asarray(a=values, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError("Assembled stencil is not a finite [5,ndata] array.")
    return result


def project_directory(requested):
    """Find the project without embedding a machine-specific workspace path.

    Arguments:
      requested = explicit project path, or None to inspect known locations.
    Returns:
      Absolute path to a project containing scripts/run_3pcf.py.
    Raises:
      ValueError when none of the declared locations contains that project.
    """
    if requested is not None:
        candidates = [Path(requested)]
    else:
        candidates = [
            Path.cwd(),
            Path.cwd() / "Cocoa/projects/des_3pcf",
            Path(__file__).resolve().parents[2],
        ]
    for candidate in candidates:
        if (candidate / "scripts/run_3pcf.py").is_file():
            return candidate.resolve()
    raise ValueError("Pass --project with the des_3pcf project directory.")


def run(configuration, output, project, larger_step=False, execute=False):
    """Validate a saved baseline, then optionally evaluate four/eight new means.

    Arguments:
      configuration = baseline Fisher YAML, with its saved result and inputs.
      output = new or empty directory for stencils.npz and summary.json.
      project = absolute des_3pcf project directory.
      larger_step = additionally complete the two-Delta stencil.
      execute = False validates and prints the plan without constructing a model.
    Returns:
      JSON-compatible study record with identities and explicit interpretation.
    Raises:
      ValueError for changed files, invalid rows/covariance, bounds or stencils.
      FileExistsError when the destination already contains a result.
    Side effects:
      Reads saved inputs and limits loaded BLAS pools to one. If execute is True,
      runs the normal provider/predictor with at most four OpenMP workers and
      saves the new vectors. The baseline and physical implementation stay fixed.
    """
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project.parents[1] / "external_modules/code/cosmolike_core"))
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project.parents[1] / "external_modules/code/cosmolike_core"))
    core = project.parents[1] / "external_modules/code/cosmolike_core"
    sys.path.insert(0, str(core))
    import run_3pcf as driver
    from des3pcf.provenance import file_digest, prediction_identity
    from des3pcf.provenance import verify_prediction_identity
    from cosmolike_notebook_utils.fisher import retained_stencil_noise, predict_fom_noise
    from cosmolike_notebook_utils.fisher import fisher_matrix, figure_of_merit
    from scipy.linalg import solve_triangular
    from threadpoolctl import threadpool_limits

    blas_limit = threadpool_limits(limits=1, user_api="blas")
    config, path, config_digest = driver.load_configuration(
        path=configuration, command="fisher",
    )
    if config["fisher"].get("precision", {}).get("mode", "inverse") != "inverse":
        raise ValueError(
            "This diagnostic requires the supplied fixed covariance inverse."
        )
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory for this step study.")
    baseline_path = driver.resolve_path(path, config["outputs"]["fisher"])
    baseline_digest = file_digest(path=baseline_path)
    with np.load(file=baseline_path, allow_pickle=False) as archive:
        arrays = {}
        for key in archive.files:
            arrays[key] = archive[key].copy()
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if metadata["configuration_sha256"] != config_digest:
        raise ValueError("Baseline and current configuration bytes differ.")
    names = arrays["parameter_names"].tolist()
    if names != list(config["fisher"]["steps"]):
        raise ValueError("Baseline and configuration parameter orders differ.")
    columns = [names.index(name) for name in COLUMNS]
    rows = driver.rows_for_run(config=config, path=path)
    for key, expected in (
        ("row_labels", rows.labels),
        ("row_kind", rows.kind),
        ("row_bins", rows.bins),
        ("row_angles_radians", rows.angles),
    ):
        if not np.array_equal(arrays[key], expected):
            raise ValueError(f"Current {key} differs from the baseline.")
    covariance = driver.read_matrix(
        settings=config["covariance"], path=path, rows=rows, positive=True,
    )
    if not np.array_equal(covariance, arrays["fiducial_covariance"]):
        raise ValueError("Current covariance differs from the baseline.")
    reference = dict(config["ia"])
    reference.update(config["parameters"])
    plan = new_point_plan(
        parameters=reference,
        steps=config["fisher"]["steps"],
        bounds=config["fisher"].get("bounds", {}),
        larger_step=larger_step,
    )
    saved = {}
    for name, column in zip(COLUMNS, columns):
        saved[name] = reuse_nodes(
            stencil_h=arrays["stencil_values_h"][column],
            stencil_h2=arrays["stencil_values_h2"][column],
        )
        if not np.array_equal(
            saved[name][0].view(np.uint64), arrays["mean"].view(np.uint64),
        ):
            raise ValueError("Saved stencil center differs from the baseline mean.")
    second_extension = None
    core_settings = config["mean"]["two_point"]
    if core_settings["mode"] == "core_tatt" and np.any(rows.kind != "map3"):
        directory = driver.resolve_path(path, core_settings["directory"])
        second_extension = directory / (core_settings["module"] + ".so")
    third_extension = project / "interface/cosmolike_des_3pcf_interface.so"

    def snapshot():
        """Hash current disk dependencies; loaded native memory is not inspected."""
        return {
            "baseline_sha256": file_digest(path=baseline_path),
            "configuration_sha256": file_digest(path=path),
            "study_sha256": file_digest(path=Path(__file__)),
            "diagnostic_sha256": file_digest(
                path=core / "cosmolike_notebook_utils/fisher.py",
            ),
            "prediction_identity": prediction_identity(
                config=config,
                configuration_path=path,
                project=project,
                three_point_path=third_extension,
                two_point_path=second_extension,
            ),
        }

    beginning = snapshot()
    if (beginning["baseline_sha256"] != baseline_digest
            or beginning["configuration_sha256"] != config_digest):
        raise ValueError("Baseline or configuration bytes changed during loading.")
    verify_prediction_identity(
        recorded=metadata.get("prediction_identity"),
        current=beginning["prediction_identity"],
        description="Saved baseline for the cosmological step check",
    )
    report = {
        "format": "des3pcf-cosmology-step-study-v1",
        "configuration": str(path),
        "baseline": str(baseline_path),
        "identity": beginning,
        "selected_parameters": list(COLUMNS),
        "new_mean_count": len(plan),
        "new_point_plan": plan,
        "executed": False,
        "interpretation": (
            "Only Omega_m and Hubble h columns are replaced in the fixed full "
            "Delta/2 Fisher. These are step diagnostics, not full new forecasts."
        ),
    }
    if not execute:
        print(json.dumps(report, indent=2, allow_nan=False))
        return report
    # The driver sets the declared one-to-four OpenMP workers. BLAS remains
    # at one. The provider prepares each new cosmology normally; saved means
    # stay in Python arrays and are never injected into provider state.
    compiled = driver.compiled_extension(config=config)
    provider, _ = driver.provider_for_run(compiled=compiled, config=config, path=path)
    prediction = driver.prediction_for_run(
        compiled=compiled, config=config, path=path, rows=rows, provider=provider,
    )
    third_extension = Path(compiled.__file__)
    if prediction.two_point_core is not None:
        second_extension = Path(prediction.compiled.__file__)
    verify_prediction_identity(
        recorded=beginning,
        current=snapshot(),
        description="Loaded extensions before the cosmological step check",
    )
    new_values = evaluate_missing(plan=plan, predict=prediction, ndata=rows.values.size)
    expected_count = 8 if larger_step else 4
    if sum(len(values) for values in new_values.values()) != expected_count:
        raise RuntimeError("The new-point evaluation count is inconsistent.")
    factor = np.linalg.cholesky(a=covariance)
    inverse_factor = solve_triangular(factor, np.eye(N=rows.values.size), lower=True)
    precision = inverse_factor.T @ inverse_factor
    full_information = arrays["fisher_h2"]
    pair = ["w0", "wa"]
    target = [names.index(name) for name in pair]
    base_fom = figure_of_merit(fisher=full_information, names=names, pair=pair)
    steps = np.asarray(a=[config["fisher"]["steps"][name] for name in COLUMNS])
    labels = ["h4", "h2", "h"]
    if larger_step:
        labels.append("two_h")
    output_arrays = {
        "parameter_names": np.asarray(a=names),
        "selected_columns": np.asarray(a=columns),
        "absolute_base_steps": steps,
        "row_labels": rows.labels,
    }
    report["cases"] = []
    for label in labels:
        selected_values = []
        for name in COLUMNS:
            selected_values.append(assemble_stencil(
                nodes=saved[name], new_values=new_values[name], label=label,
            ))
        values = np.asarray(a=selected_values)
        spacing = steps * STEP_FACTORS[label]
        diagnostic = retained_stencil_noise(
            stencil_values=values, steps=spacing, invcov=precision,
        )
        derivatives = arrays["derivatives_h2"].copy()
        changes = {}
        for position, column in enumerate(columns):
            # Use the shared estimator's derivative, not a second numerical
            # formula. Its accumulation order can differ by roundoff from the
            # original forecast; the reused h2 case displays that control.
            derivative = diagnostic["ddv"][:, position]
            derivatives[:, column] = derivative
            baseline_column = arrays["derivatives_h2"][:, column]
            norm = np.linalg.norm(solve_triangular(factor, baseline_column, lower=True))
            change = solve_triangular(factor, derivative-baseline_column, lower=True)
            changes[names[column]] = float(np.linalg.norm(change)/norm)
        information = fisher_matrix(
            derivatives=derivatives,
            names=names,
            covariance=covariance,
            priors=config["fisher"]["priors"],
        )
        merit_ratio = None
        rank_status = "resolved"
        try:
            merit = figure_of_merit(fisher=information, names=names, pair=pair)
            merit_ratio = float(merit/base_fom)
        except ValueError as error:
            rank_status = str(error)
        additions = np.zeros(shape=len(names))
        additions[columns] = diagnostic["a"]
        prediction_noise = predict_fom_noise(
            F=full_information, i=target[0], j=target[1], a=additions,
        )
        record = {
            "step_label": label,
            "absolute_steps": spacing.tolist(),
            "selected_derivative_replacement_fom_ratio": merit_ratio,
            "selected_derivative_replacement_status": rank_status,
            "column_fractional_covariance_norm_change_from_h2": changes,
            "sigma": diagnostic["sigma"].tolist(),
            "a": diagnostic["a"].tolist(),
            "diagonal_noise_inflation_ratio": float(prediction_noise["ratio"]),
            "first_order_noise_dlnfom": float(prediction_noise["dlnFoM"]),
        }
        report["cases"].append(record)
        output_arrays[f"stencil_values_{label}"] = values
        output_arrays[f"selected_derivatives_{label}"] = derivatives[:, columns]
    report["executed"] = True
    report["fixed_full_h2_fom"] = float(base_fom)
    verify_prediction_identity(
        recorded=beginning,
        current=snapshot(),
        description="Cosmological step check during evaluation",
    )
    output_arrays["metadata_json"] = np.asarray(a=json.dumps(report, sort_keys=True))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stencils.npz").open("xb") as stream:
        np.savez_compressed(stream, **output_arrays)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    """Parse explicit execution settings and run the requested study.

    Arguments:
      None; options come from the process command line.
    Returns:
      None; run writes or prints the study record.
    Raises:
      Propagates configuration, identity and numerical validation errors.
    Side effects:
      Reads files; only --execute starts predictions and saves their arrays.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--larger-step", action="store_true")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    run(
        configuration=arguments.configuration,
        output=arguments.output,
        project=project_directory(requested=arguments.project),
        larger_step=arguments.larger_step,
        execute=arguments.execute,
    )


if __name__ == "__main__":
    main()
