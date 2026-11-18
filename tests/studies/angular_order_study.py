"""Check five angular-order candidates against one saved Fisher mean.

Default: validate provenance and print a plan without evaluating a model.
With --execute, run five sequential ordinary CLI evaluate calls, with
BLAS fixed to one and the baseline's one-to-four OpenMP workers. The input
must be the explicit completed baseline Fisher configuration; all existing
accuracy overrides, source choices, rows and covariance are preserved.
These are fiducial mean checks, not derivative or FoM convergence tests.

Example from the project directory, after the baseline Fisher has finished:
  python tests/studies/angular_order_study.py \
    runs/tatt_accuracy/triangle_1p5.json --output runs/tatt_angular_orders
Repeat the same command with --execute to request physical calculations.
The installed script discovers its project from tests/studies/. An explicit
--project overrides that location. The default explanatory note is its
sibling angular_order_validation.md. Use a separate output directory and
baseline configuration for NLA. Every invocation defaults to preflight.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# Command-line execution sets pools before importing numerical libraries.
# Importing these array/configuration helpers must not alter a caller's pools.
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
    os.environ["OMP_THREAD_LIMIT"] = "4"

import numpy as np


# Each entry declares its name, L/M cutoffs, doubled prepared-mu flag,
# and exact-coupling flag. The last three entries isolate both controls
# separately and then combine them at the same L50/M50 reference.
CASES = (
    (
        "orders_L50_M20",
        50,
        20,
        False,
        False,
    ),
    (
        "orders_L50_M50",
        50,
        50,
        False,
        False,
    ),
    (
        "orders_L50_M50_angular",
        50,
        50,
        True,
        False,
    ),
    (
        "orders_L50_M50_coupling",
        50,
        50,
        False,
        True,
    ),
    (
        "orders_L50_M50_guard",
        50,
        50,
        True,
        True,
    ),
)
ROW_KEYS = (
    "row_labels",
    "row_kind",
    "row_bins",
    "row_angles_radians",
)


def project_directory(requested):
    """Find the installed project, or validate an explicit project directory.

    The normal location is tests/studies/angular_order_study.py; walking
    three parents reaches its project root. Current-directory alternatives
    support invocation from the project directory or the checkout root.
    """
    if requested is not None:
        candidates = [Path(requested)]
    else:
        candidates = [
            Path(__file__).resolve().parent.parent.parent,
            Path.cwd(),
            Path.cwd() / "Cocoa/projects/des_3pcf",
        ]
    for candidate in candidates:
        if ((candidate / "scripts/run_3pcf.py").is_file()
                and (candidate / "python/des3pcf/provenance.py").is_file()):
            return candidate.resolve()
    raise ValueError("Pass --project with the des_3pcf project directory.")


def validate_threads(threads):
    """Require one through four integer OpenMP workers before any child starts."""
    if (isinstance(threads, (bool, np.bool_))
            or not isinstance(threads, (int, np.integer))
            or not 1 <= threads <= 4):
        raise ValueError("Baseline threads must be an integer from one through four.")


def json_bytes(record):
    """Serialize a study record deterministically, retaining dictionary order."""
    text = json.dumps(obj=record, indent=2, allow_nan=False)
    text += "\n"
    return text.encode("utf-8")


def identical_file(path, content, write=False):
    """Refuse conflicting bytes; create an absent file only during execution."""
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(f"Conflicting existing file: {path}")
    elif write:
        with path.open(mode="xb") as stream:
            stream.write(content)


def load_arrays(path):
    """Read owned arrays and explicit JSON metadata without pickled objects."""
    arrays = {}
    with np.load(file=path, allow_pickle=False) as archive:
        for key in archive.files:
            arrays[key] = archive[key].copy()
    metadata_array = arrays.pop("metadata_json")
    metadata_text = str(metadata_array.item())
    metadata = json.loads(s=metadata_text)
    return arrays, metadata


def check_mean(arrays, rows):
    """Check finite means and the full row identities before covariance weighting."""
    expected = (
        rows.labels,
        rows.kind,
        rows.bins,
        rows.angles,
    )
    for key, value in zip(ROW_KEYS, expected):
        if not np.array_equal(a1=arrays[key], a2=value):
            raise ValueError(f"Saved {key} differs from the baseline rows.")
    mean = arrays["mean"]
    if (mean.dtype != np.float64 or mean.shape != (rows.values.size,)
            or not np.all(a=np.isfinite(mean))):
        raise ValueError("Saved mean must be a finite float64 full-row vector.")
    if not np.array_equal(a1=arrays["observed_values"], a2=rows.values):
        raise ValueError("Saved measurement values differ from the baseline.")
    return mean


def candidate_configuration(config, path, output, name, lmax, mmax,
                            double_prepared_mu, exact_coupling,
                            driver, absolute_configuration):
    """Apply only declared angular changes and a new evaluate output path.

    The two flags independently double the inherited angular_boost and
    replace coupling interpolation by its exact finite analytic sum. Other
    accuracy settings and physical multipole interpolation axes stay fixed.
    """
    result = absolute_configuration(config=config, path=path)
    absolute_outputs = {}
    for key, value in result["outputs"].items():
        absolute_path = driver.resolve_path(configuration_path=path, value=value)
        absolute_outputs[key] = str(absolute_path)
    result["outputs"] = absolute_outputs
    result["outputs"]["evaluate"] = str(output / (name + ".npz"))
    aperture = result["mean"]["aperture_settings"]
    aperture["lmax"] = lmax
    aperture["mmax"] = mmax
    aperture["projection_settings"]["lmax"] = lmax
    if double_prepared_mu:
        result["accuracy"]["angular_boost"] *= 2.0
    if exact_coupling:
        aperture["transform_options"]["npsi"] = 0
    return result


def run(configuration, project, output, plan_note, execute=False):
    """Preflight one baseline and optionally execute its five mean checks.

    Existing results are reused only after checking candidate bytes, native
    and Python dependencies, input files, rows and finite means. A previously
    recorded result hash must also agree. Failed or interrupted CLI logs may
    be appended only within a verified study directory. No output is replaced.
    """
    project = project.resolve()
    output = output.resolve()
    plan_note = plan_note.resolve()
    if not (project / "scripts/run_3pcf.py").is_file():
        raise ValueError("--project must name the des_3pcf project directory.")
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project / "python"))
    import run_3pcf as driver
    from run_fisher_accuracy import absolute_configuration
    from des3pcf.accuracy import accuracy_settings
    from des3pcf.mean import aligned_fft_grid
    from des3pcf.provenance import file_digest, prediction_identity
    from des3pcf.provenance import verify_prediction_identity
    from scipy.linalg import solve_triangular
    from threadpoolctl import threadpool_limits

    # Importing a driver may set its own OMP environment. Each child below
    # receives the declared baseline count explicitly; the parent stays serial.
    os.environ["OMP_NUM_THREADS"] = "1"
    blas_limit = threadpool_limits(limits=1, user_api="blas")
    config, path, config_digest = driver.load_configuration(
        path=configuration, command="fisher",
    )
    validate_threads(threads=config["threads"])
    if config["fisher"].get("precision", {}).get("mode", "inverse") != "inverse":
        raise ValueError("This check requires the fixed supplied covariance inverse.")
    aperture = config["mean"]["aperture_settings"]
    options = aperture["transform_options"]
    if (config["mean"].get("aperture_estimator") != "finite_sas"
            or aperture["projection_mode"] != "prepared"
            or aperture["lmax"] != 20 or aperture["mmax"] != 20
            or aperture["projection_settings"]["lmax"] != 20
            or aperture["phi_bins"] != 63
            or aperture.get("angular_average", "point") != "point"
            or options.get("npsi", 200) != 200
            or not options.get("ordered_legs", True)):
        raise ValueError("This plan requires prepared L20/M20, 63-point ordered SAS.")
    accuracy = accuracy_settings(value=config["accuracy"])
    edges = driver.grid_values(settings=aperture["theta_edges_arcmin"])
    ell, indices = aligned_fft_grid(
        theta_lower=edges[:-1] * driver.ARCMIN_TO_RADIAN,
        dlnt=float(np.log(edges[1] / edges[0])),
        ell_min=aperture["ell_min"], ell_max=aperture["ell_max"],
        minimum_size=accuracy.count(
            baseline=aperture["minimum_size"], stage="fft_boost",
        ),
    )
    geometry = {
        "fft_count": int(ell.size),
        "selected_radial_count": int(indices.size),
        "selected_indices": indices.tolist(),
        "aligned_ell_bounds": [float(ell[0]), float(ell[-1])],
        "phi_count": aperture["phi_bins"],
        "angular_average": aperture.get("angular_average", "point"),
    }
    baseline = driver.resolve_path(
        configuration_path=path, value=config["outputs"]["fisher"],
    )
    baseline_sha = file_digest(path=baseline)
    arrays, metadata = load_arrays(path=baseline)
    if metadata["configuration_sha256"] != config_digest:
        raise ValueError("Baseline Fisher and current configuration bytes differ.")
    if metadata.get("configuration") != config:
        raise ValueError("Baseline retained configuration differs from the loaded file.")
    if metadata.get("command") != "fisher":
        raise ValueError("The baseline must be a completed Fisher archive.")
    rows = driver.rows_for_run(config=config, path=path)
    base_mean = check_mean(arrays=arrays, rows=rows)
    if base_mean.size != 480 or np.count_nonzero(a=rows.kind == "map3") != 80:
        raise ValueError("This study requires the declared 400 xi plus 80 Map3 rows.")
    covariance = driver.read_matrix(
        settings=config["covariance"], path=path, rows=rows, positive=True,
    )
    if not np.array_equal(a1=covariance, a2=arrays["fiducial_covariance"]):
        raise ValueError("Current covariance differs from the saved Fisher covariance.")
    third = project / "interface/cosmolike_des_3pcf_interface.so"
    core = config["mean"]["two_point"]
    second = None
    if core["mode"] == "core_tatt":
        core_directory = driver.resolve_path(
            configuration_path=path, value=core["directory"],
        )
        second = core_directory / (core["module"] + ".so")

    def snapshot():
        """Hash dependencies without constructing a model or recalculating CAMB."""
        return {
            "baseline_sha256": file_digest(path=baseline),
            "configuration_sha256": file_digest(path=path),
            "study_sha256": file_digest(path=Path(__file__)),
            "plan_sha256": file_digest(path=plan_note),
            "configuration_helper_sha256": file_digest(
                path=project / "scripts/run_fisher_accuracy.py",
            ),
            "prediction_identity": prediction_identity(
                config=config, configuration_path=path, project=project,
                three_point_path=third, two_point_path=second,
            ),
        }

    beginning = snapshot()
    if (beginning["configuration_sha256"] != config_digest
            or beginning["baseline_sha256"] != baseline_sha):
        raise ValueError("Baseline files changed while being loaded.")
    verify_prediction_identity(
        recorded=metadata.get("prediction_identity"),
        current=beginning["prediction_identity"], description="Angular-order baseline",
    )
    candidates = []
    expected_files = {
        "manifest.json",
        "study_plan.md",
        "summary.json",
    }
    for name, lmax, mmax, double_prepared_mu, exact_coupling in CASES:
        candidate = candidate_configuration(
            config=config, path=path, output=output, name=name, lmax=lmax,
            mmax=mmax, double_prepared_mu=double_prepared_mu,
            exact_coupling=exact_coupling, driver=driver,
            absolute_configuration=absolute_configuration,
        )
        content = json_bytes(record=candidate)
        candidate_path = output / (name + ".json")
        candidate_accuracy = accuracy_settings(value=candidate["accuracy"])
        resolved = candidate_accuracy.projection_settings(
            baseline=candidate["mean"]["aperture_settings"]["projection_settings"],
        )
        direct_nmu = candidate_accuracy.count(
            baseline=aperture["nmu"], stage="angular_boost", integration=True,
        )
        if direct_nmu <= lmax:
            raise ValueError("Direct-mode nmu must still pass the engine's nmu>L check.")
        candidate_options = candidate["mean"]["aperture_settings"]["transform_options"]
        if double_prepared_mu and exact_coupling:
            interpretation = (
                "Joint guard: doubled prepared mu sampling and exact analytic coupling."
            )
        elif double_prepared_mu:
            interpretation = (
                "Angular-only guard: doubled prepared mu sampling at fixed coupling."
            )
        elif exact_coupling:
            interpretation = (
                "Coupling-only guard: exact analytic coupling at fixed mu sampling."
            )
        else:
            interpretation = "Angular-order change with inherited sampling unchanged."
        record = {
            "name": name,
            "configuration": str(candidate_path),
            "configuration_sha256": hashlib.sha256(content).hexdigest(),
            "mean": str(output / (name + ".npz")),
            "lmax": lmax,
            "mmax": mmax,
            "double_prepared_mu": double_prepared_mu,
            "exact_coupling": exact_coupling,
            "prepared_mu_counts": [resolved["nmu_log"], resolved["nmu_linear"]],
            "resolved_projection_settings": resolved,
            "coupling_npsi": candidate_options["npsi"],
            "interpretation": interpretation,
        }
        candidates.append((
            record,
            candidate,
            content,
        ))
        for suffix in (
            ".json",
            ".npz",
            ".log",
            ".receipt.json",
        ):
            expected_files.add(name + suffix)
    candidate_records = []
    for record, candidate, content in candidates:
        candidate_records.append(record)
    manifest = {
        "format": "des3pcf-angular-order-study-v1",
        "baseline_configuration": str(path),
        "baseline_fisher": str(baseline),
        "identity": beginning,
        "inherited_accuracy": deepcopy(x=config["accuracy"]),
        "threads": config["threads"],
        "blas_threads": 1,
        "shared_geometry": geometry,
        "candidates": candidate_records,
        "interpretation": "Fiducial mean checks only; no derivative/FoM validation.",
    }
    if output.exists():
        if not output.is_dir():
            raise FileExistsError("Study output must be a directory.")
        existing_names = set()
        for existing_path in output.iterdir():
            existing_names.add(existing_path.name)
        extra = existing_names - expected_files
        if extra:
            raise FileExistsError(f"Unexpected files in study output: {sorted(extra)}")
        if existing_names and not (output / "manifest.json").is_file():
            raise FileExistsError("Existing files have no verified study manifest.")
    identical_file(path=output / "manifest.json", content=json_bytes(record=manifest))
    identical_file(path=output / "study_plan.md", content=plan_note.read_bytes())
    missing_result = False
    for record, _, content in candidates:
        candidate_path = Path(record["configuration"])
        identical_file(path=candidate_path, content=content)
        exists = Path(record["mean"]).exists()
        if exists and (missing_result or not candidate_path.is_file()):
            raise ValueError("Existing results need their configs and a complete prefix.")
        missing_result = missing_result or not exists
    verify_prediction_identity(recorded=beginning, current=snapshot(),
                               description="Angular-order preflight")
    if execute:
        output.mkdir(parents=True, exist_ok=True)
        identical_file(path=output / "manifest.json", content=json_bytes(record=manifest),
                       write=True)
        identical_file(path=output / "study_plan.md", content=plan_note.read_bytes(),
                       write=True)
        for record, _, content in candidates:
            identical_file(path=Path(record["configuration"]), content=content,
                           write=True)
    factor = np.linalg.cholesky(a=covariance)
    records = []
    previous = base_mean
    previous_name = "inherited_baseline"
    order_reference = None
    for record, candidate, content in candidates:
        mean_path = Path(record["mean"])
        receipt_path = output / (record["name"] + ".receipt.json")
        if receipt_path.exists() and not mean_path.exists():
            raise FileNotFoundError(
                "An existing result receipt has lost its mean archive."
            )
        if not mean_path.exists() and execute:
            verify_prediction_identity(recorded=beginning, current=snapshot(),
                                       description="Before angular-order candidate")
            identical_file(path=Path(record["configuration"]), content=content)
            environment = dict(os.environ)
            environment["OMP_NUM_THREADS"] = str(config["threads"])
            command = [
                sys.executable,
                str(project / "scripts/run_3pcf.py"),
                "evaluate",
                record["configuration"],
            ]
            with (output / (record["name"] + ".log")).open(mode="ab") as log:
                subprocess.run(
                    args=command,
                    cwd=project, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, check=True,
                )
        if not mean_path.exists():
            records.append({"name": record["name"], "status": "pending"})
            previous = None
            previous_name = record["name"]
            continue
        verify_prediction_identity(recorded=beginning, current=snapshot(),
                                   description="After/load angular-order candidate")
        identical_file(path=Path(record["configuration"]), content=content)
        values, result_metadata = load_arrays(path=mean_path)
        if (result_metadata.get("command") != "evaluate"
                or result_metadata["configuration_sha256"]
                != record["configuration_sha256"]
                or result_metadata["configuration"] != candidate):
            raise ValueError("Candidate result has different configuration/procedure.")
        # File identity deliberately omits numerical controls. The exact
        # candidate JSON hash and retained configuration above establish
        # L/M, angular sampling and coupling settings independently.
        verify_prediction_identity(
            recorded=result_metadata.get("prediction_identity"),
            current=beginning["prediction_identity"],
            description="Saved angular-order candidate",
        )
        mean = check_mean(arrays=values, rows=rows)
        xi = rows.kind != "map3"
        if not np.array_equal(
            a1=mean[xi].view(np.uint64), a2=base_mean[xi].view(np.uint64),
        ):
            raise ValueError(
                "An angular-order candidate changed supposedly fixed xi rows."
            )
        difference = mean - base_mean
        whitened = solve_triangular(a=factor, b=difference, lower=True)
        entry = {
            "name": record["name"],
            "status": "complete",
            "result_sha256": file_digest(path=mean_path),
            "delta_chi2_to_baseline": float(whitened @ whitened),
            "max_absolute_row_change": float(np.max(a=np.abs(difference))),
            "xi_bitwise_unchanged": True,
            "comparison": record["interpretation"],
        }
        if previous is not None:
            change = solve_triangular(a=factor, b=mean - previous, lower=True)
            entry["previous_case"] = previous_name
            entry["delta_chi2_to_previous"] = float(change @ change)
        if record["name"] == "orders_L50_M50":
            order_reference = mean
        elif order_reference is not None:
            # Adjacent angular-only and coupling-only cases change both
            # controls. This common reference isolates each intended test.
            reference_difference = mean - order_reference
            reference_change = solve_triangular(
                a=factor, b=reference_difference, lower=True,
            )
            entry["order_reference_case"] = "orders_L50_M50"
            entry["delta_chi2_to_L50_M50"] = float(reference_change @ reference_change)
        identical_file(path=receipt_path, content=json_bytes(record=entry), write=execute)
        records.append(entry)
        previous = mean
        previous_name = record["name"]
    verify_prediction_identity(recorded=beginning, current=snapshot(),
                               description="Completed angular-order inspection")
    for candidate_record, _, content in candidates:
        identical_file(path=Path(candidate_record["configuration"]), content=content)
    for result in records:
        if result["status"] == "complete":
            result_path = output / (result["name"] + ".npz")
            if file_digest(path=result_path) != result["result_sha256"]:
                raise ValueError("A saved candidate changed during the study.")
    complete = True
    for record in records:
        if record["status"] != "complete":
            complete = False
            break
    summary = {
        "manifest": manifest,
        "cases": records,
        "complete": complete,
        "metric": "Full fixed joint covariance, including xi-Map3 cross-covariance.",
    }
    if summary["complete"]:
        identical_file(path=output / "summary.json", content=json_bytes(record=summary),
                       write=execute)
    elif (output / "summary.json").exists():
        raise ValueError("A completed summary exists but one candidate is absent.")
    print(json.dumps(obj=summary, indent=2, allow_nan=False))
    del blas_limit
    return summary


def main():
    """Parse explicit paths; --execute is the only route to physical model calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--project", type=Path,
                        help="Override project discovery from this script location.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-note", type=Path, default=Path(__file__).with_name(
        "angular_order_validation.md",
    ))
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    run(
        configuration=arguments.configuration,
        project=project_directory(requested=arguments.project),
        output=arguments.output,
        plan_note=arguments.plan_note,
        execute=arguments.execute,
    )


if __name__ == "__main__":
    main()
