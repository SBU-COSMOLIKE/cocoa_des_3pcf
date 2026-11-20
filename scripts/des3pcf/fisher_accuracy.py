"""Read and identify DES numerical-accuracy results saved by its run scripts.

The Fisher estimator and array calculations live in the shared Fisher module.
This file owns only the project's filenames, configuration and reuse checks.
"""

import json
from pathlib import Path
import numpy as np
from .provenance import file_digest, prediction_identity


def accuracy_cases_for_request(candidates, only, output):
    """Visit requested cases plus earlier archives in the current plan.

    Existence selects a case for inspection, not scientific reuse. Every
    retained file must subsequently pass load_retained_noise_case with its
    complete expected baseline, configuration, and runtime identity.
    """
    output = Path(output)
    result = []
    for candidate in candidates:
        requested = only is None or candidate["name"] in only
        saved = output / f"{candidate['name']}_noise.npz"
        if requested or saved.is_file():
            result.append(candidate)
    return result


def load_retained_noise_case(path, identity):
    """Read a completed case only after exact dependency-identity validation.

    No value is recovered from summary.json: that file can omit an earlier
    case after a partial invocation. The archive is the authoritative saved
    stencil, and a stale archive raises instead of silently joining a scan.
    """
    with np.load(file=path, allow_pickle=False) as saved:
        recorded = json.loads(str(saved["identity_json"].item()))
        if recorded != identity:
            raise ValueError(
                f"{path} has different inputs; choose a new study directory."
            )
        return {
            "a": saved["a"].copy(),
            "sigma": saved["sigma"].copy(),
            "stencil_values": saved["stencil_values"].copy(),
            "evaluation_count": int(saved["evaluation_count"].item()),
        }


def current_noise_case_identity(
    config, configuration_path, project, three_point_path, two_point_path,
    baseline_path, dependency_paths, selected_columns, steps,
):
    """Read the current complete identity before saving a new accuracy case.

    Physical files, both native libraries, and Python/CAMB implementations
    use the same fingerprint as a full Fisher result. The diagnostic also
    depends on its exact baseline archive, configuration, plan, and shared
    Fisher helper. Comparing this result with the beginning identity detects
    replacements during candidate evaluation; it cannot inspect code already
    loaded into memory or changes restored between the two reads.

    Arguments:
      config = resolved numerical candidate configuration.
      configuration_path = candidate YAML path whose exact bytes are recorded.
      project = project root containing the prediction implementation.
      three_point_path, two_point_path = loaded extension paths; the second
          is None when the mean does not call the ordinary two-point core.
      baseline_path = saved Fisher archive used to interpret this candidate.
      dependency_paths = accuracy plan and diagnostic implementation files.
      selected_columns = ordered parameter-column indices evaluated here.
      steps = absolute finite-difference steps in the full parameter order.
    Returns:
      JSON-compatible identity, with no physical model evaluation.
    """
    dependencies = {}
    for filename in dependency_paths:
        dependencies[str(Path(filename).resolve())] = file_digest(path=filename)
    return {
        "baseline_result_sha256": file_digest(path=baseline_path),
        "configuration_sha256": file_digest(path=configuration_path),
        "dependency_sha256": dependencies,
        "prediction_identity": prediction_identity(
            config=config,
            configuration_path=configuration_path,
            project=project,
            three_point_path=three_point_path,
            two_point_path=two_point_path,
        ),
        "selected_columns": list(selected_columns),
        "absolute_steps": np.asarray(steps, dtype=np.float64).tolist(),
    }
