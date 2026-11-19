"""Screen numerical settings using one saved default-accuracy Fisher base.

The full derivative base fixes the marginalization weights of the w0–wa
ellipse. Each candidate then evaluates only selected fourth-difference
stencils. The shared cosmolike_notebook_utils.fisher functions estimate
noise and its expected information; this driver supplies project predictions,
absolute parameter steps, exact caches and saved provenance.

Example from the project directory:
  python scripts/run_fisher_accuracy.py configs/des_y3_tatt_demo_fisher.yaml \
    --plan configs/des_y3_fisher_accuracy.yaml --output runs/tatt_accuracy

The baseline Fisher result and physical tables must already exist. A saved
candidate is reused only when its baseline, resolved settings and compiled
module agree. Full derivative checks use the generated candidate YAML with
the ordinary run_3pcf.py fisher command. No timing experiment is performed.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1] / "external_modules/code/cosmolike_core"
sys.path.insert(0, str(PROJECT / "python"))
sys.path.insert(0, str(CORE))
import run_3pcf as driver
from des3pcf.accuracy import accuracy_settings
from des3pcf.forecast import ExactMeanCache, figure_of_merit, parameter_covariance
from des3pcf.forecast import fisher_matrix
from des3pcf.fisher_accuracy import retained_stencil_noise, make_offset_predictor
from des3pcf.fisher_accuracy import get_fom_noise_weights, get_stencil_noise
from des3pcf.fisher_accuracy import predict_fom_noise
from des3pcf.fisher_accuracy import accuracy_cases_for_request
from des3pcf.fisher_accuracy import load_retained_noise_case
from des3pcf.fisher_accuracy import current_noise_case_identity
from des3pcf.provenance import prediction_identity
from des3pcf.provenance import verify_prediction_identity


def digest_file(path):
    """Return the SHA256 of one explicit input or compiled-module file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absolute_configuration(config, path):
    """Resolve the input paths before writing a candidate beside its results.

    Arguments:
      config = validated joint configuration; path = its original YAML path.
    Returns:
      independent configuration with absolute physical input paths.
    """
    result = deepcopy(config)
    locations = [
        (result["source"], "path"),
        (result["physics"], "tables"),
        (result["rows"], "two_point"),
        (result["rows"], "aperture"),
        (result["mean"]["two_point"], "directory"),
        (result["covariance"], "path"),
    ]
    for section, key in locations:
        value = section.get(key)
        if value is not None:
            section[key] = str(driver.resolve_path(configuration_path=path, value=value))
    return result


def select_columns(names, contributions, pair, fraction, maximum):
    """Rank measured baseline contributions while always checking the FoM pair.

    The cap can prevent reaching the requested fraction. The returned actual
    fraction records that outcome; it never becomes a bound on omitted noise
    at another numerical setting.

    Arguments:
      names = full parameter order; contributions = nonnegative baseline terms.
      pair = two FoM names; fraction = desired share in (0,1]; maximum = cap.
    Returns:
      selected column indices and their measured baseline contribution share.
    """
    if not 0.0 < fraction <= 1.0 or not 2 <= maximum <= len(names):
        raise ValueError("Selection needs 0<fraction<=1 and 2<=maximum<=nparameter.")
    selected = []
    for name in pair:
        selected.append(names.index(name))
    total = float(np.sum(contributions))
    ordering = np.argsort(-contributions, kind="stable")
    for index in ordering:
        retained = float(np.sum(contributions[selected]))
        if total > 0.0 and retained >= fraction * total:
            break
        if len(selected) == maximum:
            break
        if int(index) not in selected:
            selected.append(int(index))
    captured = None
    if total > 0.0:
        captured = float(np.sum(contributions[selected]) / total)
    return selected, captured


def run(configuration, plan_path, output, only=None):
    """Measure selected noise stencils with fixed default-accuracy weights.

    Arguments:
      configuration = baseline Fisher YAML whose result already exists.
      plan_path = explicit candidate/column selection YAML.
      output = study directory; existing consistent cases may be loaded.
      only = optional list of candidate names to execute in this invocation.
    Returns:
      JSON-serializable study summary, also written to summary.yaml.
    Side effects:
      Runs one case at a time using the configured maximum of four workers;
      writes candidate configurations, stencils and provenance.
    """
    config, path, config_digest = driver.load_configuration(
        path=configuration, command="fisher",
    )
    if config["fisher"].get("precision", {}).get("mode", "inverse") != "inverse":
        raise ValueError("This noise study requires the declared fixed covariance inverse.")
    if any(value != 1.0 for value in config["accuracy"].values()):
        raise ValueError("Build the expensive base with all global/stage boosts equal to one.")
    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    if plan["format"] != "des3pcf-fisher-accuracy-v1":
        raise ValueError("The accuracy plan has an unsupported format.")
    if only is not None:
        available = {candidate["name"] for candidate in plan["candidates"]}
        unknown = set(only) - available
        if unknown:
            raise ValueError(f"Unknown requested accuracy cases: {sorted(unknown)}")
    divisor = float(plan["step_divisor"])
    if divisor not in (1.0, 2.0):
        raise ValueError("step_divisor must select the saved h or h/2 stencil.")
    label = "h" if divisor == 1.0 else "h2"
    baseline_path = driver.resolve_path(path, config["outputs"]["fisher"])
    baseline_digest = digest_file(path=baseline_path)
    with np.load(file=baseline_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata["configuration_sha256"] != config_digest:
            raise ValueError("Baseline Fisher result belongs to different configuration bytes.")
        arrays = {}
        for name in archive.files:
            arrays[name] = archive[name].copy()
    two_point_path = None
    if (config["mean"]["two_point"]["mode"] == "core_tatt"
            and np.any(arrays["row_kind"] != "map3")):
        two_point = config["mean"]["two_point"]
        directory = driver.resolve_path(path, two_point["directory"])
        two_point_path = directory / (two_point["module"] + ".so")
    current_identity = prediction_identity(
        config=config,
        configuration_path=path,
        project=PROJECT,
        three_point_path=PROJECT / "interface/cosmolike_des_3pcf_interface.so",
        two_point_path=two_point_path,
    )
    verify_prediction_identity(
        recorded=metadata.get("prediction_identity"),
        current=current_identity,
        description="Baseline Fisher",
    )
    names = arrays["parameter_names"].tolist()
    rows = driver.rows_for_run(config=config, path=path)
    for key, current in (("row_labels", rows.labels), ("row_kind", rows.kind),
                         ("row_bins", rows.bins), ("row_angles_radians", rows.angles)):
        if not np.array_equal(arrays[key], current):
            raise ValueError(f"Current {key} differs from the baseline; rebuild its Fisher result.")
    current_covariance = driver.read_matrix(
        settings=config["covariance"], path=path, rows=rows, positive=True,
    )
    if not np.array_equal(current_covariance, arrays["fiducial_covariance"]):
        raise ValueError("Current covariance differs from the baseline; rebuild its Fisher result.")
    pair = plan["pair"]
    pair_indices = [names.index(pair[0]), names.index(pair[1])]
    information = arrays[f"fisher_{label}"]
    # Apply the same rank criterion used for reported project FoMs before
    # passing the matrix to the shared core's diagnostic functions.
    parameter_covariance(fisher=information, names=names)
    steps = np.empty(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        steps[index] = config["fisher"]["steps"][name] / divisor
    covariance = arrays["fiducial_covariance"]
    precision = np.linalg.solve(covariance, np.eye(covariance.shape[0]))
    precision = 0.5 * (precision + precision.T)
    baseline_noise = retained_stencil_noise(
        stencil_values=arrays[f"stencil_values_{label}"],
        steps=steps, invcov=precision,
    )
    weights = get_fom_noise_weights(
        F=information, i=pair_indices[0], j=pair_indices[1], eps=steps,
    )
    contributions = weights["R2"] * weights["sigma2"] * baseline_noise["a"]
    selected, captured = select_columns(
        names=names, contributions=contributions, pair=pair,
        fraction=float(plan["selection_fraction"]), maximum=int(plan["maximum_columns"]),
    )
    baseline_selected = np.zeros(len(names))
    baseline_selected[selected] = baseline_noise["a"][selected]
    baseline_prediction = predict_fom_noise(
        F=information, i=pair_indices[0], j=pair_indices[1], a=baseline_selected,
    )
    baseline_full_prediction = predict_fom_noise(
        F=information, i=pair_indices[0], j=pair_indices[1], a=baseline_noise["a"],
    )
    omitted = []
    for index, name in enumerate(names):
        if index not in selected:
            omitted.append(index)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Physical files and code are already shared with the baseline identity.
    # These additional files define the diagnostic, rather than the mean.
    dependencies = [
        CORE / "cosmolike_notebook_utils/fisher.py",
        Path(__file__).resolve(),
        Path(plan_path).resolve(),
    ]
    dependency_hashes = {}
    for filename in dependencies:
        dependency_hashes[str(filename)] = digest_file(path=filename)
    report = {
        "format": "des3pcf-fisher-accuracy-result-v1",
        "configuration_sha256": config_digest,
        "plan_sha256": digest_file(path=plan_path),
        "prediction_identity": current_identity,
        "dependency_sha256": dependency_hashes,
        "baseline_result_sha256": baseline_digest,
        "baseline_result": str(baseline_path),
        "parameter_names": names,
        "pair": pair,
        "absolute_steps": steps.tolist(),
        "selected_columns": selected,
        "selected_parameters": [names[index] for index in selected],
        "baseline_contribution_fraction": captured,
        "omitted_parameters": [names[index] for index in omitted],
        "omitted_baseline_contribution_sum": float(np.sum(contributions[omitted])),
        "baseline_full_noise_prediction": {
            "first_order_dlnfom": baseline_full_prediction["dlnFoM"],
            "noise_inflation_ratio": baseline_full_prediction["ratio"],
        },
        "baseline_noise_a": baseline_noise["a"].tolist(),
        "baseline_noise_sigma": baseline_noise["sigma"].tolist(),
        "R2": weights["R2"].tolist(),
        "sigma2": weights["sigma2"].tolist(),
        "baseline_contributions": contributions.tolist(),
        "baseline_fom": float(figure_of_merit(information, names=names, pair=pair)),
        "evaluations_per_candidate": 1 + 4 * len(selected),
        "baseline_distinct_evaluations": metadata["cached_mean_count"],
        "interpretation": "Fixed-base noise surrogate: the baseline Fisher can itself contain numerical error. This is neither an exact candidate/base FoM prediction nor a de-noised Fisher correction. Omitted columns and coherent errors require full derivative checks.",
        "cases": [],
    }
    compiled = driver.compiled_extension(config=config)
    reference = dict(config["ia"])
    reference.update(config["parameters"])
    cases = accuracy_cases_for_request(
        candidates=plan["candidates"], only=only, output=output,
    )
    for candidate in cases:
        name = candidate["name"]
        if not name.replace("_", "").isalnum():
            raise ValueError("Candidate names must contain letters, digits or underscores.")
        requested = only is None or name in only
        destination = output / f"{name}_noise.npz"
        numerical_core_keys = {
            "accuracy_boost", "integration_accuracy", "fpt_internal_boost", "ell_internal",
        }
        unknown = set(candidate.get("two_point", {})) - numerical_core_keys
        if unknown:
            raise ValueError(f"Candidate {name} changes non-resolution two-point keys: {sorted(unknown)}.")
        case = absolute_configuration(config=config, path=path)
        controls = dict(config["accuracy"])
        controls.update(candidate.get("accuracy", {}))
        case["accuracy"] = accuracy_settings(value=controls).record()
        case["mean"]["two_point"].update(candidate.get("two_point", {}))
        case["outputs"]["fisher"] = str(output / f"{name}_full_fisher.npz")
        case["outputs"]["evaluate"] = str(output / f"{name}_mean.npz")
        case_path = output / f"{name}.yaml"
        content = yaml.safe_dump(case, sort_keys=False)
        if case_path.exists() and case_path.read_text(encoding="utf-8") != content:
            raise ValueError(f"{case_path} has different settings; choose a new study directory.")
        case_path.write_text(content, encoding="utf-8")
        case, _, case_digest = driver.load_configuration(path=case_path, command="fisher")
        destination = output / f"{name}_noise.npz"
        identity = {
            "baseline_result_sha256": baseline_digest,
            "configuration_sha256": case_digest,
            "dependency_sha256": dependency_hashes,
            "prediction_identity": current_identity,
            "selected_columns": selected,
            "absolute_steps": steps.tolist(),
        }
        if destination.exists():
            saved = load_retained_noise_case(path=destination, identity=identity)
            additions = saved["a"]
            sigma = saved["sigma"]
            samples = saved["stencil_values"]
            evaluation_count = saved["evaluation_count"]
        else:
            if not requested:
                raise RuntimeError("An unrequested case lost its saved archive.")
            provider, _ = driver.provider_for_run(compiled=compiled, config=case, path=case_path)
            rows = driver.rows_for_run(config=case, path=case_path)
            prediction = driver.prediction_for_run(
                compiled=compiled, config=case, path=case_path, rows=rows, provider=provider,
            )
            cached = ExactMeanCache(predict=prediction)
            shifted = make_offset_predictor(predict=cached, reference=reference, names=names)
            additions = np.zeros(len(names))
            sigma = np.full(len(names), np.nan)
            derivatives = np.empty((rows.values.size, len(selected)))
            samples = np.empty((len(selected), 5, rows.values.size))
            for position, column in enumerate(selected):
                diagnostic = get_stencil_noise(
                    dv=shifted, index=column, h=steps[column], CV=np.zeros(len(names)),
                    AccuracyBoost=1.0, invcov=precision,
                )
                additions[column] = diagnostic["a"]
                sigma[column] = diagnostic["sigma"]
                derivatives[:, position] = diagnostic["ddv"]
                for node, offset in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
                    coordinates = np.zeros(len(names))
                    coordinates[column] = offset * steps[column]
                    samples[position, node] = shifted(param=coordinates, AccuracyBoost=1.0)
                print(json.dumps({"candidate": name, "column": names[column],
                                  "a": diagnostic["a"]}), flush=True)
            evaluation_count = len(cached.values)
            second_extension = None
            if prediction.two_point_core is not None:
                second_extension = prediction.two_point_core.core.__file__
            completed_identity = current_noise_case_identity(
                config=case,
                configuration_path=case_path,
                project=PROJECT,
                three_point_path=compiled.__file__,
                two_point_path=second_extension,
                baseline_path=baseline_path,
                dependency_paths=dependencies,
                selected_columns=selected,
                steps=steps,
            )
            verify_prediction_identity(
                recorded=identity,
                current=completed_identity,
                description=f"Accuracy case {name} during evaluation",
            )
            np.savez_compressed(
                file=destination, a=additions, sigma=sigma, derivatives=derivatives,
                stencil_values=samples, evaluation_count=evaluation_count,
                identity_json=json.dumps(identity, sort_keys=True),
            )
        predicted = predict_fom_noise(
            F=information, i=pair_indices[0], j=pair_indices[1], a=additions,
        )
        # The same evaluated vectors also expose coherent changes in the
        # selected derivatives. Use the project's original accumulation
        # order, then replace only those columns in the default base. This
        # hybrid calculation is a diagnostic, not a full finer-grid Fisher.
        hybrid_derivatives = arrays[f"derivatives_{label}"].copy()
        for position, column in enumerate(selected):
            values = samples[position]
            derivative = values[0] - 8.0 * values[1] + 8.0 * values[3] - values[4]
            hybrid_derivatives[:, column] = derivative / (12.0 * steps[column])
        hybrid_information = fisher_matrix(
            derivatives=hybrid_derivatives, names=names, covariance=covariance,
            priors=config["fisher"]["priors"],
        )
        hybrid_ratio = None
        hybrid_status = "resolved"
        try:
            hybrid_fom = figure_of_merit(fisher=hybrid_information, names=names, pair=pair)
            hybrid_ratio = float(hybrid_fom / report["baseline_fom"])
        except ValueError as error:
            hybrid_status = str(error)
        record = {
            "name": name,
            "configuration": str(case_path),
            "evaluations": evaluation_count,
            "first_order_dlnfom": predicted["dlnFoM"],
            "noise_inflation_ratio": predicted["ratio"],
            "predicted_ratio_to_default": predicted["ratio"] / baseline_prediction["ratio"],
            "selected_derivative_replacement_fom_ratio": hybrid_ratio,
            "selected_derivative_replacement_status": hybrid_status,
            "measured_columns": report["selected_parameters"],
            "a": additions.tolist(),
        }
        report["cases"].append(record)
        (output / "summary.yaml").write_text(
            yaml.safe_dump(report, sort_keys=False), encoding="utf-8",
        )
        print(json.dumps(record, allow_nan=False), flush=True)
    return report


def main():
    """Run the requested subset of a declared accuracy-screening plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--only", nargs="+")
    arguments = parser.parse_args()
    run(configuration=arguments.configuration, plan_path=arguments.plan,
        output=arguments.output, only=arguments.only)


if __name__ == "__main__":
    main()
