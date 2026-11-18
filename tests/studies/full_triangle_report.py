"""Compare saved default and full triangle-refined Fisher calculations.

The default invocation lists missing archives and writes nothing. --execute
performs array analysis only: no physical model, native prediction, CAMB
calculation, derivative evaluation or child process is invoked. Both NLA
and TATT full results must exist before the final JSON is written.

From the project directory:
  python tests/studies/full_triangle_report.py

After both full calculations have finished:
  python tests/studies/full_triangle_report.py --execute

The default output is runs/fisher_array_diagnostics/full_triangle.json.
An existing output is accepted only if its complete bytes match this report.
"""

import argparse
from copy import deepcopy
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
    os.environ["OMP_THREAD_LIMIT"] = "1"

import numpy as np


MODELS = ("nla", "tatt")
CASE = "triangle_1p5"
PAIR = ["w0", "wa"]
IA_NAMES = (
    "a1",
    "eta1",
    "a2",
    "eta2",
    "bta",
)


def project_directory(requested):
    """Locate the installed project or honor an explicit project argument."""
    if requested is not None:
        candidates = [Path(requested)]
    else:
        candidates = [
            Path(__file__).resolve().parent.parent.parent,
            Path.cwd(),
            Path.cwd() / "Cocoa/projects/des_3pcf",
        ]
    for candidate in candidates:
        if (candidate / "interface/cosmolike_des_3pcf_accuracy_wrappers.py").is_file():
            return candidate.resolve()
    raise ValueError("Pass --project with the des_3pcf project directory.")


def json_default(value):
    """Serialize returned scientific arrays without converting missing values to zero."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def ratio(numerator, denominator):
    """Keep unresolved or zero-denominator comparisons explicitly unavailable."""
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return float(numerator / denominator)


def check_matrix(saved, rebuilt, name):
    """Check an array-only Fisher reconstruction at float64 roundoff scale.

    This is an archive-consistency test, not a physical convergence tolerance.
    Both matrices pass the shared shape, finiteness and symmetry validator.
    """
    from des3pcf.forecast import symmetric_matrix

    first = symmetric_matrix(values=saved, name=name)
    second = symmetric_matrix(values=rebuilt, name=name + " reconstruction")
    if first.shape != second.shape:
        raise ValueError(f"Inconsistent matrix dimensions for {name}.")
    scale = max(float(np.max(a=np.abs(first))), float(np.max(a=np.abs(second))))
    error = float(np.max(a=np.abs(first - second)))
    tolerance = 128.0 * first.shape[0] * np.finfo(np.float64).eps * scale
    if error > tolerance:
        raise ValueError(
            f"Saved {name} does not reproduce from its archived derivatives."
        )
    return 0.0 if scale == 0.0 else error / scale


def information_summary(derivatives, arrays, metadata, saved_fisher=None, step=None):
    """Use shared Fisher, rank, fixed-DE and free-DE helpers for one derivative set.

    The fixed-DE summary fixes w0/wa at their unchanged reference values and
    marginalizes every other parameter. The free-DE summary marginalizes them
    too. Both retain the xi-Map3 cross-covariance and count each prior once.
    """
    from des3pcf.forecast import figure_of_merit
    from des3pcf.fisher_comparison import compare_information, marginalized_errors
    from cosmolike_des_3pcf_fisher_wrappers import fixed_dark_energy_summary
    from cosmolike_des_3pcf_fisher_wrappers import derived_amplitude_summary

    names = arrays["parameter_names"].tolist()
    comparison = compare_information(
        derivatives=derivatives, names=names, kind=arrays["row_kind"],
        covariance=arrays["fiducial_covariance"],
        priors=metadata["configuration"]["fisher"]["priors"],
    )
    reconstruction = None
    if saved_fisher is not None:
        reconstruction = check_matrix(
            saved=saved_fisher, rebuilt=comparison["fisher_joint"],
            name=f"full posterior Fisher ({step})",
        )
    if step == "h2":
        for key in (
            "fisher_prior",
            "fisher_joint",
            "fisher_two_point",
            "fisher_joint_likelihood",
            "fisher_two_point_likelihood",
        ):
            check_matrix(saved=arrays[key], rebuilt=comparison[key], name=key)
    errors, ranks = marginalized_errors(comparison=comparison, names=names)
    helper_input = dict(comparison)
    helper_input.update(errors)
    helper_input["parameter_names"] = arrays["parameter_names"]
    result = {
        "rank_status": ranks,
        "w0wa_fom": None,
        "saved_fisher_reconstruction_relative_max": reconstruction,
    }
    matrix = comparison["fisher_joint"] if saved_fisher is None else saved_fisher
    try:
        result["w0wa_fom"] = float(figure_of_merit(
            fisher=matrix, names=names, pair=PAIR,
        ))
    except ValueError as error:
        result["w0wa_rank_failure"] = str(error)
    try:
        fixed = fixed_dark_energy_summary(result=helper_input, metadata=metadata)
        ia_gains = {}
        for name in fixed["joint"]["sigma_ia"]:
            two_point_error = fixed["two_point"]["sigma_ia"][name]
            joint_error = fixed["joint"]["sigma_ia"][name]
            ia_gains[name] = ratio(numerator=two_point_error, denominator=joint_error)
        fixed["ia_gain_two_point_over_joint"] = ia_gains
        result["fixed_dark_energy"] = {"status": "resolved", "summary": fixed}
    except ValueError as error:
        result["fixed_dark_energy"] = {"status": "unresolved", "reason": str(error)}
    all_probes_resolved = True
    for status in ranks.values():
        if status != "resolved":
            all_probes_resolved = False
            break
    if all_probes_resolved:
        amplitude = derived_amplitude_summary(result=helper_input, metadata=metadata)
        amplitude["sigma_ia"] = {}
        for probe in ("two_point", "joint"):
            probe_errors = {}
            for name in IA_NAMES:
                if name in names:
                    parameter_index = names.index(name)
                    parameter_error = errors[f"sigma_{probe}"][parameter_index]
                    probe_errors[name] = float(parameter_error)
            amplitude["sigma_ia"][probe] = probe_errors
        amplitude_gains = {}
        for name in amplitude["sigma_ia"]["joint"]:
            two_point_error = amplitude["sigma_ia"]["two_point"][name]
            joint_error = amplitude["sigma_ia"]["joint"][name]
            amplitude_gains[name] = ratio(
                numerator=two_point_error, denominator=joint_error,
            )
        amplitude["ia_gain_two_point_over_joint"] = amplitude_gains
        result["free_dark_energy"] = {"status": "resolved", "summary": amplitude}
    else:
        result["free_dark_energy"] = {"status": "unresolved", "rank_status": ranks}
    return result


def derivative_changes(baseline, refined, covariance, names, steps, label):
    """Describe each column's change with the full covariance, in physical units."""
    from scipy.linalg import solve_triangular
    from des3pcf.forecast import symmetric_matrix

    shape = (covariance.shape[0], len(names))
    for values in (baseline, refined):
        if (values.dtype != np.float64 or values.shape != shape
                or not np.all(a=np.isfinite(values))):
            raise ValueError("Derivatives must be finite float64 full-row matrices.")
    matrix = symmetric_matrix(values=covariance, name="Joint covariance", positive=True)
    factor = np.linalg.cholesky(a=matrix)
    original = solve_triangular(a=factor, b=baseline, lower=True)
    change = solve_triangular(a=factor, b=refined - baseline, lower=True)
    if not np.all(a=np.isfinite(original)) or not np.all(a=np.isfinite(change)):
        raise ValueError("Whitened derivatives overflowed; do not assign a finite norm.")
    baseline_norms = np.sqrt(np.sum(a=original * original, axis=0))
    change_norms = np.sqrt(np.sum(a=change * change, axis=0))
    divisor = 1.0 if label == "h" else 2.0
    records = []
    for column, name in enumerate(names):
        records.append({
            "parameter": name,
            "default_derivative_C_norm": float(baseline_norms[column]),
            "derivative_difference_C_norm": float(change_norms[column]),
            "relative_derivative_C_norm_change": ratio(
                numerator=change_norms[column], denominator=baseline_norms[column],
            ),
            "absolute_parameter_step": float(steps[name] / divisor),
            "step_times_derivative_difference_C_norm": float(
                steps[name] / divisor * change_norms[column],
            ),
        })
    return records


def selected_stencil_check(report, noise_path, refined):
    """Compare screening means and derivative arithmetic with the full h2 archive.

    The noise archive's ddv field need not share the driver's accumulation
    order. Reconstruct from its retained five vectors in exactly the scan's
    documented order, then compare uint64 representations of the results.
    """
    columns = report["selected_columns"]
    names = refined["parameter_names"].tolist()
    with np.load(file=noise_path, allow_pickle=False) as archive:
        samples = archive["stencil_values"].copy()
        identity_text = str(archive["identity_json"].item())
        identity = json.loads(s=identity_text)
    if identity["absolute_steps"] != report["absolute_steps"]:
        raise ValueError("Screening and summary absolute steps differ.")
    expected_shape = (
        len(columns),
        5,
        refined["mean"].size,
    )
    if (samples.dtype != np.float64 or samples.shape != expected_shape
            or not np.all(a=np.isfinite(samples))):
        raise ValueError("Screening archive lacks finite complete selected stencils.")
    records = []
    for position, column in enumerate(columns):
        full = np.ascontiguousarray(a=refined["stencil_values_h2"][column])
        selected = np.ascontiguousarray(a=samples[position])
        if (full.dtype != np.float64 or full.shape != selected.shape
                or not np.all(a=np.isfinite(full))):
            raise ValueError("Full selected stencil has incompatible shape or values.")
        mismatch = full.view(np.uint64) != selected.view(np.uint64)
        step = report["absolute_steps"][column]
        derivative = (selected[0] - 8.0 * selected[1]
                      + 8.0 * selected[3] - selected[4]) / (12.0 * step)
        target = np.ascontiguousarray(a=refined["derivatives_h2"][:, column])
        records.append({
            "parameter": names[column],
            "stencils_bitwise_equal": bool(not np.any(a=mismatch)),
            "mismatched_stencil_values": int(np.count_nonzero(a=mismatch)),
            "maximum_absolute_stencil_difference": float(
                np.max(a=np.abs(full - selected)),
            ),
            "derivative_bitwise_equal": bool(np.array_equal(
                a1=derivative.view(np.uint64), a2=target.view(np.uint64),
            )),
            "center_bitwise_equal": bool(np.array_equal(
                a1=selected[2].view(np.uint64),
                a2=refined["mean"].view(np.uint64),
            )),
        })
    all_bitwise_equal = True
    for record in records:
        if (not record["stencils_bitwise_equal"]
                or not record["derivative_bitwise_equal"]
                or not record["center_bitwise_equal"]):
            all_bitwise_equal = False
            break
    return {
        "columns": records,
        "all_bitwise_equal": all_bitwise_equal,
        "interpretation": (
            "Bitwise agreement isolates full-versus-selected substitution differences "
            "from any change in these four evaluated stencils. It does not validate "
            "the omitted derivatives or the physical model."
        ),
    }


def compare_summaries(default, refined):
    """Report stability ratios for the paper-related summaries without changing priors."""
    result = {}
    for mode in ("fixed_dark_energy", "free_dark_energy"):
        default_record = default[mode]
        refined_record = refined[mode]
        if (default_record["status"] != "resolved"
                or refined_record["status"] != "resolved"):
            result[mode] = {"status": "unresolved"}
            continue
        default_summary = default_record["summary"]
        refined_summary = refined_record["summary"]
        record = {
            "status": "resolved",
            "triangle_over_default_joint_to_xi_omega_m_s8_fom_gain": ratio(
                numerator=refined_summary["joint_over_two_point_fom"],
                denominator=default_summary["joint_over_two_point_fom"],
            ),
            "probes": {},
        }
        for probe in ("two_point", "joint"):
            if mode == "fixed_dark_energy":
                default_ia = default_summary[probe]["sigma_ia"]
                refined_ia = refined_summary[probe]["sigma_ia"]
            else:
                default_ia = default_summary["sigma_ia"][probe]
                refined_ia = refined_summary["sigma_ia"][probe]
            ia_error_ratios = {}
            for name in default_ia:
                ia_error_ratios[name] = ratio(
                    numerator=refined_ia[name], denominator=default_ia[name],
                )
            record["probes"][probe] = {
                "triangle_over_default_sigma_s8": ratio(
                    numerator=refined_summary[probe]["sigma_s8"],
                    denominator=default_summary[probe]["sigma_s8"],
                ),
                "triangle_over_default_sigma_ia": ia_error_ratios,
            }
        result[mode] = record
    return result


def analyze_model(study):
    """Validate archive/config identity before comparing every saved derivative column."""
    import run_3pcf as driver
    from run_fisher_accuracy import absolute_configuration

    os.environ["OMP_NUM_THREADS"] = "1"
    from des3pcf.forecast import figure_of_merit
    from cosmolike_des_3pcf_fisher_wrappers import FisherWorkflow
    from cosmolike_des_3pcf_fisher_wrappers import finite_sign_pair_separation

    checks = study.load_full_checks()
    # The existing reader uses NaN only to mark unresolved ranks. Preserve
    # that meaning as JSON null together with its explicit rank-failure text.
    ratio_rows = []
    for row in checks["ratios"]:
        ratio_row = []
        for ratio_value in row:
            if np.isfinite(ratio_value):
                ratio_row.append(float(ratio_value))
            else:
                ratio_row.append(None)
        ratio_rows.append(ratio_row)
    checks["ratios"] = ratio_rows
    report, baseline, base_metadata = study.load()
    if report["pair"] != PAIR or CASE in checks["pending"]:
        raise ValueError("Require the completed w0-wa full triangle comparison.")
    case = None
    for record in report["cases"]:
        if record["name"] == CASE:
            case = record
            break
    if case is None:
        raise StopIteration
    config, path, _ = driver.load_configuration(
        path=study.configuration, command="fisher",
    )
    candidate, _, _ = driver.load_configuration(
        path=case["configuration"], command="fisher",
    )
    refined_workflow = FisherWorkflow(configuration=case["configuration"])
    refined, refined_metadata = refined_workflow.load_result(command="fisher")
    for settings, metadata in ((config, base_metadata), (candidate, refined_metadata)):
        if metadata["configuration"] != settings:
            raise ValueError(
                "An archive's retained physical configuration is inconsistent."
            )
        if settings["fisher"].get("precision", {}).get("mode", "inverse") != "inverse":
            raise ValueError(
                "This report requires the declared fixed covariance inverse."
            )
        if (settings["parameters"].get("w0") != -1.0
                or settings["parameters"].get("wa") != 0.0):
            raise ValueError("Fixed-DE comparison requires fiducial w0=-1, wa=0.")
    expected = absolute_configuration(config=config, path=path)
    expected["accuracy"]["triangle_boost"] = 1.5
    expected["outputs"] = deepcopy(x=candidate["outputs"])
    if expected != candidate:
        raise ValueError(
            "Triangle case changes settings beyond triangle_boost and outputs."
        )
    for accuracy_boost in config["accuracy"].values():
        if accuracy_boost != 1.0:
            raise ValueError(
                "This report's default archive must have unit accuracy boosts."
            )
    names = baseline["parameter_names"].tolist()
    if names != list(config["fisher"]["steps"]):
        raise ValueError("Parameter order differs from the configured derivative steps.")
    columns = report["selected_columns"]
    if len(columns) != 4 or len(set(columns)) != 4:
        raise ValueError("This comparison requires the recorded four distinct columns.")
    selected_names = []
    for column in columns:
        selected_names.append(names[column])
    if selected_names != report["selected_parameters"]:
        raise ValueError("This comparison requires the recorded four distinct columns.")
    h2_steps = []
    for name in names:
        h2_steps.append(config["fisher"]["steps"][name] / 2.0)
    if report["absolute_steps"] != h2_steps:
        raise ValueError("The selected-column scan must use the same absolute h/2 steps.")
    rows = baseline["row_kind"]
    if rows.size != 480 or np.count_nonzero(a=rows == "map3") != 80:
        raise ValueError("Expected the full 400 xi plus 80 Map3 demonstration vector.")
    norm = finite_sign_pair_separation(
        first=refined, second=baseline, covariance=baseline["fiducial_covariance"],
    )
    selected_check = selected_stencil_check(
        report=report, noise_path=study.output / (CASE + "_noise.npz"), refined=refined,
    )
    fixed_denominator = float(figure_of_merit(
        fisher=baseline["fisher_h2"], names=names, pair=PAIR,
    ))
    if not np.isclose(a=report["baseline_fom"], b=fixed_denominator,
                      rtol=1.0e-12, atol=0.0):
        raise ValueError(
            "Saved screening denominator differs from the full default h2 FoM."
        )
    unselected_parameters = []
    for column, name in enumerate(names):
        if column not in columns:
            unselected_parameters.append(name)
    result = {
        "selected_stencil_reproducibility": selected_check,
        "established_full_checks": checks,
        "physical_prediction_identity": report["prediction_identity"],
        "parameter_names": names,
        "selected_parameters": report["selected_parameters"],
        "unselected_parameters": unselected_parameters,
        "fiducial_distance": {
            "chi2_joint": norm["chi2_joint"],
            "chi2_two_point": norm["chi2_two_point"],
            "chi2_conditional_three_point": norm["chi2_conditional_three_point"],
        },
        "fixed_default_h2_w0wa_fom_denominator": fixed_denominator,
        "steps": {},
        "selected_screen_h2_fom_ratio": case["selected_derivative_replacement_fom_ratio"],
        "interpretation": (
            "Full, selected-only and complement-only substitutions are coherent "
            "matrix comparisons, not additive error contributions. The mixed-grid "
            "substitutions are diagnostics, not completed refined forecasts."
        ),
    }
    for label in ("h", "h2"):
        original = baseline[f"derivatives_{label}"]
        changed = refined[f"derivatives_{label}"]
        changes = derivative_changes(
            baseline=original, refined=changed,
            covariance=baseline["fiducial_covariance"],
            names=names, steps=config["fisher"]["steps"], label=label,
        )
        selected_only = original.copy()
        selected_only[:, columns] = changed[:, columns]
        complement_only = changed.copy()
        complement_only[:, columns] = original[:, columns]
        default_summary = information_summary(
            derivatives=original, arrays=baseline, metadata=base_metadata,
            saved_fisher=baseline[f"fisher_{label}"], step=label,
        )
        triangle_summary = information_summary(
            derivatives=changed, arrays=refined, metadata=refined_metadata,
            saved_fisher=refined[f"fisher_{label}"], step=label,
        )
        selected_summary = information_summary(
            derivatives=selected_only, arrays=baseline, metadata=base_metadata,
        )
        complement_summary = information_summary(
            derivatives=complement_only, arrays=baseline, metadata=base_metadata,
        )
        descriptions = {
            "default": default_summary,
            "full_triangle": triangle_summary,
            "selected_four_only": selected_summary,
            "unselected_complement_only": complement_summary,
        }
        default_fom = descriptions["default"]["w0wa_fom"]
        full_fom = descriptions["full_triangle"]["w0wa_fom"]
        for description in descriptions.values():
            description["ratio_to_fixed_default_h2_w0wa_fom"] = ratio(
                numerator=description["w0wa_fom"], denominator=fixed_denominator,
            )
            description["ratio_to_same_step_default_w0wa_fom"] = ratio(
                numerator=description["w0wa_fom"], denominator=default_fom,
            )
        chosen = descriptions["selected_four_only"]["w0wa_fom"]
        complement = descriptions["unselected_complement_only"]["w0wa_fom"]
        difference = None
        if full_fom is not None and chosen is not None:
            difference = full_fom - chosen
        nonadditivity = None
        merits = (
            full_fom,
            chosen,
            complement,
            default_fom,
        )
        all_foms_available = True
        for merit in merits:
            if merit is None:
                all_foms_available = False
                break
        if all_foms_available:
            nonadditivity = full_fom - chosen - complement + default_fom
        xi_rows = rows != "map3"
        original_xi = np.ascontiguousarray(a=original[xi_rows])
        changed_xi = np.ascontiguousarray(a=changed[xi_rows])
        xi_bitwise_unchanged = np.array_equal(
            a1=original_xi.view(np.uint64), a2=changed_xi.view(np.uint64),
        )
        result["steps"][label] = {
            "information": descriptions,
            "derivative_columns": changes,
            "xi_derivatives_bitwise_unchanged": bool(xi_bitwise_unchanged),
            "full_over_selected_four_w0wa_fom": ratio(
                numerator=full_fom, denominator=chosen,
            ),
            "full_minus_selected_four_w0wa_fom": difference,
            "nonadditive_w0wa_fom_remainder": nonadditivity,
            "paper_related_stability": compare_summaries(
                default=descriptions["default"], refined=descriptions["full_triangle"],
            ),
        }
    reproduced = result["steps"]["h2"]["information"]["selected_four_only"]
    scan_ratio = result["selected_screen_h2_fom_ratio"]
    reproduced_ratio = reproduced["ratio_to_fixed_default_h2_w0wa_fom"]
    screen_difference = None
    if scan_ratio is not None and reproduced_ratio is not None:
        screen_difference = reproduced_ratio - scan_ratio
    result["full_selected_h2_minus_screen_ratio"] = screen_difference
    return result


def run(project, destination, execute=False):
    """Report pending archives, or save a fully verified array-only comparison."""
    project = project.resolve()
    destination = destination.resolve()
    sys.path.insert(0, str(project / "interface"))
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project / "python"))
    import run_3pcf as driver
    from cosmolike_des_3pcf_accuracy_wrappers import AccuracyStudy
    from des3pcf.provenance import file_digest
    from threadpoolctl import threadpool_limits
    import scipy

    os.environ["OMP_NUM_THREADS"] = "1"
    blas_limit = threadpool_limits(limits=1, user_api="blas")
    studies = {}
    inputs = set()
    status = {}
    for model in MODELS:
        configuration = project / f"configs/des_y3_{model}_demo_fisher.yaml"
        scan = project / f"runs/{model}_accuracy"
        plan = project / "configs/des_y3_fisher_accuracy.yaml"
        config, path, _ = driver.load_configuration(path=configuration, command="fisher")
        baseline = driver.resolve_path(
            configuration_path=path, value=config["outputs"]["fisher"],
        )
        candidate_path = scan / (CASE + ".json")
        required = [
            configuration,
            baseline,
            plan,
            scan / "summary.json",
            candidate_path,
            scan / (CASE + "_noise.npz"),
        ]
        if candidate_path.is_file():
            candidate, origin, _ = driver.load_configuration(
                path=candidate_path, command="fisher",
            )
            full_result = driver.resolve_path(
                configuration_path=origin, value=candidate["outputs"]["fisher"],
            )
            required.append(full_result)
        missing = []
        for required_path in required:
            if not required_path.is_file():
                missing.append(str(required_path))
        if missing:
            model_status = "pending"
        else:
            model_status = "ready"
        status[model] = {"status": model_status, "missing": missing}
        inputs.update(required)
        studies[model] = AccuracyStudy(
            configuration=configuration, output=scan, plan=plan,
        )
        if (scan / "summary.json").is_file():
            summary_path = scan / "summary.json"
            summary_text = summary_path.read_text(encoding="utf-8")
            report = json.loads(s=summary_text)
            for case in report["cases"]:
                case_path = Path(case["configuration"])
                noise = scan / (case["name"] + "_noise.npz")
                inputs.update((case_path, noise))
                for dependency in (case_path, noise):
                    if not dependency.is_file() and str(dependency) not in missing:
                        missing.append(str(dependency))
                        status[model]["status"] = "pending"
                if case_path.is_file():
                    other, other_path, _ = driver.load_configuration(
                        path=case_path, command="fisher",
                    )
                    full = driver.resolve_path(
                        configuration_path=other_path, value=other["outputs"]["fisher"],
                    )
                    inputs.add(full)
    preflight = {
        "models": status,
        "array_analysis_only": True,
        "destination": str(destination),
        "execute_requested": execute,
    }
    pending_model = False
    if execute:
        for status_record in status.values():
            if status_record["status"] == "pending":
                pending_model = True
                break
    if not execute or pending_model:
        print(json.dumps(obj=preflight, indent=2, allow_nan=False))
        return preflight
    code_directory = project / "python/des3pcf"
    code = list(code_directory.glob("*.py"))
    code.extend([
        Path(__file__).resolve(),
        project / "scripts/run_3pcf.py",
        project / "scripts/run_fisher_accuracy.py",
        project / "interface/cosmolike_des_3pcf_accuracy_wrappers.py",
        project / "interface/cosmolike_des_3pcf_fisher_wrappers.py",
    ])

    def snapshot():
        """Hash analysis inputs/code; do not attest to a current physical model."""
        input_hashes = {}
        for input_path in sorted(inputs):
            if input_path.is_file():
                digest = file_digest(path=input_path)
            else:
                digest = None
            input_hashes[str(input_path)] = digest
        code_hashes = {}
        for code_path in sorted(code):
            code_hashes[str(code_path)] = file_digest(path=code_path)
        return {
            "input_sha256": input_hashes,
            "analysis_code_sha256": code_hashes,
        }

    beginning = snapshot()
    result = {
        "format": "des3pcf-full-triangle-array-report-v1",
        "provenance": beginning,
        "analysis_libraries": {"numpy": np.__version__, "scipy": scipy.__version__},
        "scope": (
            "Historical saved-array comparison. Archived physical identities must "
            "agree between default, screening and full candidates. Current analysis "
            "code is fingerprinted separately; no physical model is evaluated."
        ),
        "models": {},
    }
    for model, study in studies.items():
        result["models"][model] = analyze_model(study=study)
    if snapshot() != beginning:
        raise ValueError(
            "Input archives, configurations or analysis code changed during reading."
        )
    report_text = json.dumps(
        obj=result, indent=2, allow_nan=False, default=json_default,
    )
    report_text += "\n"
    content = report_text.encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(
                "Different report exists; choose a new --output filename."
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open(mode="xb") as stream:
            stream.write(content)
    saved_report = {
        "saved": str(destination),
        "models": list(result["models"]),
        "array_analysis_only": True,
    }
    print(json.dumps(obj=saved_report, indent=2))
    del blas_limit
    return result


def main():
    """Require --execute for array analysis; never request a physical calculation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    project = project_directory(requested=arguments.project)
    destination = arguments.output or (
        project / "runs/fisher_array_diagnostics/full_triangle.json"
    )
    run(project=project, destination=destination, execute=arguments.execute)


if __name__ == "__main__":
    main()
