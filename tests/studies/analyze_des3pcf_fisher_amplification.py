"""Inspect historical Fisher arrays with independent QR and SVD algebra.

No physical model or CAMB calculation is requested. Archived input identities
are compared with each other, not with today's physical code or source files.
The current array-helper code and every input archive are hashed separately.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        os.environ[variable] = "1"

import numpy as np
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits


def digest(path):
    """Return the exact SHA256 of one small archived input or analysis file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_arrays(path):
    """Read owned arrays and refuse a file changed while it was being loaded."""
    before = digest(path=path)
    with np.load(file=path, allow_pickle=False) as archive:
        arrays = {}
        for name in archive.files:
            arrays[name] = archive[name].copy()
    if digest(path=path) != before:
        raise ValueError(f"Archive changed during reading: {path}")
    return arrays, {"path": str(path), "sha256": before}


def augmented(jacobian, covariance, prior):
    """Stack C-whitened derivatives and Gaussian-prior rows to give F=A.T@A."""
    factor = np.linalg.cholesky(a=covariance)
    whitened = solve_triangular(a=factor, b=jacobian, lower=True)
    prior_rows = np.diag(v=np.sqrt(np.diag(v=prior)))
    return np.vstack(tup=(whitened, prior_rows))


def marginal(array, target):
    """Project away nuisance columns before forming target information.

    Normalizing columns changes neither their span nor the QR projector.
    Returns the marginalized information and surviving response columns.
    """
    nuisance = []
    for index in range(array.shape[1]):
        if index not in target:
            nuisance.append(index)
    columns = array[:, nuisance]
    scales = np.linalg.norm(x=columns, axis=0)
    basis, _ = np.linalg.qr(a=columns/scales[None, :], mode="reduced")
    residual = array[:, target] - basis @ (basis.T @ array[:, target])
    return residual.T @ residual, residual


def metrics(array, names, target, forecast):
    """Compare nuisance QR with the project's scaled inverse, without repair."""
    fisher = array.T @ array
    scales = np.sqrt(np.diag(v=fisher))
    normalized = fisher/scales[:, None]/scales[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(a=normalized)
    singular = np.linalg.svd(a=array/scales[None, :], compute_uv=False)
    effective, residual = marginal(array=array, target=target)
    target_eigenvalues, target_eigenvectors = np.linalg.eigh(a=effective)
    covariance = forecast.parameter_covariance(fisher=fisher, names=names)
    merit_qr = np.sqrt(np.linalg.det(a=effective))
    merit_inverse = forecast.figure_of_merit(
        fisher=fisher, names=names, pair=["w0", "wa"],
    )
    largest = np.argsort(a=np.abs(eigenvectors[:, 0]))[::-1][:8]
    weak = {}
    for index in largest:
        weak[names[index]] = float(eigenvectors[index, 0])
    variance = np.diag(v=covariance)[target]
    correlation = covariance[target[0], target[1]]/np.sqrt(np.prod(a=variance))
    weak_target = target_eigenvectors[:, 0]/target_eigenvectors[1, 0]
    record = {
        "fom_qr": float(merit_qr),
        "fom_inverse": float(merit_inverse),
        "qr_inverse_relative_difference": float(merit_qr/merit_inverse-1),
        "normalized_fisher_eigenvalue_min": float(eigenvalues[0]),
        "normalized_fisher_eigenvalue_max": float(eigenvalues[-1]),
        "normalized_fisher_condition": float(eigenvalues[-1]/eigenvalues[0]),
        "augmented_singular_ratio": float(singular[-1]/singular[0]),
        "rank_refusal_eigenvalue_ratio": 128*len(names)*np.finfo(float).eps,
        "weak_normalized_coordinates": weak,
        "target_effective_information": effective.tolist(),
        "target_information_eigenvalues": target_eigenvalues.tolist(),
        "weak_target_wa_unit_direction": weak_target.tolist(),
        "target_sigma": np.sqrt(variance).tolist(),
        "target_correlation": float(correlation),
    }
    return record, effective, residual


def partial_ratio(array, reference_merit, target):
    """Return the FoM of this artificial column substitution relative to the base."""
    effective, _ = marginal(array=array, target=target)
    return float(np.sqrt(np.linalg.det(a=effective))/reference_merit)


def candidate_record(base, state, candidate, names, selected, forecast):
    """Inspect the measured column replacement and its target-space response.

    Arguments:
      base = saved full baseline arrays; state = whitened h and h2 arrays.
      candidate = saved selected-column stencil archive; names = full order.
      selected = candidate's column indices; forecast = array-only helpers.
    Returns:
      Metrics for a partial substitution. Unselected derivatives remain at h2.
    """
    target = [names.index("w0"), names.index("wa")]
    identity = json.loads(str(candidate["identity_json"].item()))
    steps = np.asarray(a=identity["absolute_steps"])
    updated = base["derivatives_h2"].copy()
    for position, column in enumerate(selected):
        value = candidate["stencil_values"][position]
        # Retain the archive runner's arithmetic when rebuilding its hybrid.
        derivative = value[0]-8*value[1]+8*value[3]-value[4]
        updated[:, column] = derivative/(12*steps[column])
    array = augmented(
        jacobian=updated,
        covariance=base["fiducial_covariance"],
        prior=base["fisher_prior"],
    )
    current = state["h2"]
    reference, effective0, residual0 = metrics(
        array=current, names=names, target=target, forecast=forecast,
    )
    result, effective1, residual1 = metrics(
        array=array, names=names, target=target, forecast=forecast,
    )
    result["fom_ratio"] = result["fom_qr"]/reference["fom_qr"]
    factor = np.linalg.cholesky(a=effective0)
    relative = solve_triangular(a=factor, b=effective1, lower=True)
    relative = solve_triangular(a=factor, b=relative.T, lower=True).T
    values, vectors = np.linalg.eigh(a=0.5*(relative+relative.T))
    directions = solve_triangular(a=factor.T, b=vectors, lower=False)
    result["target_generalized_information_ratios"] = values.tolist()
    result["target_generalized_wa_unit_directions"] = (
        directions/directions[1, :]
    ).T.tolist()
    result["changed_column_relative_norms"] = {}
    for column in selected:
        difference = np.linalg.norm(x=array[:, column]-current[:, column])
        original = np.linalg.norm(x=current[:, column])
        result["changed_column_relative_norms"][names[column]] = float(
            difference/original,
        )
    # Complete the baseline's weak target displacement with nuisance values
    # that minimize its total data/prior response. This identifies a weak
    # response in the actual whitened experiment, not just raw coordinates.
    _, vectors = np.linalg.eigh(a=effective0)
    direction = vectors[:, 0]/vectors[1, 0]
    nuisance = []
    for index in range(len(names)):
        if index not in target:
            nuisance.append(index)
    coefficients = np.zeros(shape=len(names))
    coefficients[target] = direction
    coefficients[nuisance] = np.linalg.lstsq(
        a=current[:, nuisance], b=-current[:, target]@direction, rcond=None,
    )[0]
    change = (array-current)@coefficients
    response = current@coefficients
    original_norm = np.linalg.norm(x=response)
    change_norm = np.linalg.norm(x=change)
    result["baseline_weak_displacement"] = {}
    for index, name in enumerate(names):
        result["baseline_weak_displacement"][name] = float(coefficients[index])
    result["baseline_weak_response_norm"] = float(original_norm)
    result["change_on_baseline_weak_displacement_norm"] = float(change_norm)
    result["weak_displacement_change_relative"] = float(change_norm/original_norm)
    result["weak_response_change_cosine"] = None
    if change_norm > 0:
        result["weak_response_change_cosine"] = float(
            np.dot(a=response, b=change)/(original_norm*change_norm),
        )
    groups = [
        ("w0wa", ["w0", "wa"]),
        ("h", ["h"]),
        ("omega_m", ["omega_m"]),
        ("h_omega_m", ["h", "omega_m"]),
    ]
    result["partial_replacement_ratios"] = {}
    for label, members in groups:
        columns = [names.index(name) for name in members]
        if not all(column in selected for column in columns):
            continue
        partial = current.copy()
        partial[:, columns] = array[:, columns]
        result["partial_replacement_ratios"][label] = partial_ratio(
            array=partial, reference_merit=reference["fom_qr"], target=target,
        )
    xi = base["two_point_indices"]
    result["raw_xi_changed_max"] = float(np.max(
        a=np.abs(updated[xi]-base["derivatives_h2"][xi]),
    ))
    result["projected_target_relative_changes"] = {}
    result["target_residual_fraction_of_full_columns"] = {}
    for position, column in enumerate(target):
        norm = np.linalg.norm(x=residual0[:, position])
        change = np.linalg.norm(x=residual1[:, position]-residual0[:, position])
        result["projected_target_relative_changes"][names[column]] = float(change/norm)
        result["target_residual_fraction_of_full_columns"][names[column]] = float(
            norm/np.linalg.norm(x=current[:, column]),
        )
    delta_step = state["h"]-current
    delta_grid = array-current
    result["column_grid_step_cosine"] = {}
    for column in selected:
        first = delta_step[:, column]
        second = delta_grid[:, column]
        denominator = np.linalg.norm(x=first)*np.linalg.norm(x=second)
        value = None
        if denominator > 0:
            value = float(np.dot(a=first, b=second)/denominator)
        result["column_grid_step_cosine"][names[column]] = value
    # SVD is independent of forming the Fisher Gram matrix or its inverse.
    scales = np.linalg.norm(x=array, axis=0)
    _, singular, transpose = np.linalg.svd(
        a=array/scales[None, :], full_matrices=False,
    )
    covariance = ((transpose.T/singular**2)@transpose)/scales[:, None]/scales[None, :]
    pair_covariance = covariance[np.ix_(target, target)]
    result["fom_svd"] = float(1/np.sqrt(np.linalg.det(a=pair_covariance)))
    return result


def inspect_model(project, model, forecast):
    """Read one complete baseline and currently completed compatible candidates."""
    path = project/"chains"/f"y3_{model}_measured_edge_demo_fisher.npz"
    base, input_record = read_arrays(path=path)
    metadata = json.loads(str(base["metadata_json"].item()))
    names = base["parameter_names"].tolist()
    target = [names.index("w0"), names.index("wa")]
    record = {"input": input_record, "candidates": {}}
    states = {}
    for label in ("h", "h2"):
        array = augmented(
            jacobian=base[f"derivatives_{label}"],
            covariance=base["fiducial_covariance"],
            prior=base["fisher_prior"],
        )
        result, _, _ = metrics(array=array, names=names, target=target, forecast=forecast)
        result["stored_fom"] = float(forecast.figure_of_merit(
            fisher=base[f"fisher_{label}"], names=names, pair=["w0", "wa"],
        ))
        difference = np.max(a=np.abs(array.T@array-base[f"fisher_{label}"]))
        original = np.max(a=np.abs(base[f"fisher_{label}"]))
        result["rebuilt_stored_fisher_max_relative"] = float(difference/original)
        states[label] = array
        record[label] = result
    directory = project/"chains"/f"{model}_accuracy"
    summary_path = directory/"summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record["summary_sha256"] = digest(path=summary_path)
        # Read only archives already declared complete by the summary; an
        # unrelated process may currently be writing another candidate.
        for case in summary["cases"]:
            filename = directory/f"{case['name']}_noise.npz"
            candidate, candidate_input = read_arrays(path=filename)
            identity = json.loads(str(candidate["identity_json"].item()))
            if identity["baseline_result_sha256"] != input_record["sha256"]:
                raise ValueError("Candidate belongs to different baseline bytes.")
            if identity["prediction_identity"] != metadata["prediction_identity"]:
                raise ValueError("Candidate and baseline used different physical inputs.")
            steps = []
            for name in names:
                steps.append(metadata["configuration"]["fisher"]["steps"][name]/2)
            if not np.array_equal(a1=identity["absolute_steps"], a2=steps):
                raise ValueError(
                    "This substitution audit requires h2 candidate stencils."
                )
            result = candidate_record(
                base=base,
                state=states,
                candidate=candidate,
                names=names,
                selected=identity["selected_columns"],
                forecast=forecast,
            )
            result["input"] = candidate_input
            result["archived_identity"] = identity
            record["candidates"][case["name"]] = result
    record["single_column_h_into_h2_ratios"] = {}
    for column, name in enumerate(names):
        partial = states["h2"].copy()
        partial[:, column] = states["h"][:, column]
        record["single_column_h_into_h2_ratios"][name] = partial_ratio(
            array=partial, reference_merit=record["h2"]["fom_qr"], target=target,
        )
    return record


def paths_from_arguments(description):
    """Resolve --project and --output for a portable saved-array inspection."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.project is None:
        candidates = [
            Path(__file__).resolve().parents[2],
            Path.cwd(),
            Path.cwd()/"Cocoa/projects/des_3pcf",
        ]
        for candidate in candidates:
            if (candidate/"scripts/run_3pcf.py").is_file():
                arguments.project = candidate
                break
        if arguments.project is None:
            parser.error("Pass --project with the des_3pcf project directory.")
    project = arguments.project.resolve()
    output = arguments.output
    if output is None:
        output = project/"chains/fisher_array_diagnostics"
    return project, output.resolve()


def main():
    """Write a historical array-only report; do not import a physical interface."""
    project, output = paths_from_arguments(description=__doc__)
    sys.path.insert(0, str(project/"scripts"))
    from cosmolike_notebook_utils import fisher as forecast

    limit = threadpool_limits(limits=1, user_api="blas")
    report = {}
    for model in ("nla", "tatt"):
        report[model] = inspect_model(project=project, model=model, forecast=forecast)
    report["inspection"] = {
        "interpretation": "Historical arrays; no new physical prediction or timing.",
        "script_sha256": digest(path=Path(__file__)),
        "array_helper_sha256": digest(path=Path(forecast.__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output/"amplification.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(destination)


if __name__ == "__main__":
    main()
