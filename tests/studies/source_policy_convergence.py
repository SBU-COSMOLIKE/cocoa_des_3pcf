"""Compare explicit source reconstructions without running CAMB.

This study uses the explicit measured-edge source policy. It reads
the existing demonstration power/background archive, replaces only its
source samples, and evaluates the full 400 xi plus 80 Map3 rows. The joint
480 by 480 covariance remains fixed. No scale selection or compression is
performed. Both the linear-midpoint and measured-edge PCHIP choices are
source models; changing between them is not a numerical accuracy boost.

Run from the project directory with Cocoa active:
  python tests/studies/source_policy_convergence.py --project . \
    --output /absolute/new_source_study --models nla tatt --threads 4

Use --cases to select named cases and --resume to continue an existing
study with exactly matching files and settings. Nine distinct dz1 values
supply all three five-point stencils at h=.001, .0005 and .00025. Each mean
and its source/kernel diagnostics is saved immediately. The script never
imports or calls a CAMB calculation: PhysicalProvider uses fixed_tables.

Fourth differences are deterministic residual diagnostics here. A hard
source-support edge can create a physical cusp in shifted II/III overlap
integrals. Smooth fourth derivatives and correlated numerical errors can
also contribute. A small fourth difference does not certify a derivative,
and a large one must not automatically be called independent random noise.
"""


import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

# The caller allocates at most four OpenMP workers. Every BLAS library
# remains single-threaded, including the covariance diagnostic solves.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "4"

import numpy as np
import scipy
from scipy.linalg import solve_triangular


# Named single-axis cases separate source sampling from kernel and LOS
# sampling. The combined cases test whether those errors cancel when each
# axis is refined alone. All retain the same angular interpolation controls.
CASES = {
    "linear1201": ("linear_midpoint", "zero_outside_native", 1201, 201, 1.0),
    "pchip1201": ("pchip_midpoint", "zero_at_measured_edges", 1201, 201, 1.0),
    "pchip2401": ("pchip_midpoint", "zero_at_measured_edges", 2401, 201, 1.0),
    "pchip4801": ("pchip_midpoint", "zero_at_measured_edges", 4801, 201, 1.0),
    "pchip1201_kernel401": (
        "pchip_midpoint", "zero_at_measured_edges", 1201, 401, 1.0,
    ),
    "pchip1201_chi2": (
        "pchip_midpoint", "zero_at_measured_edges", 1201, 201, 2.0,
    ),
    "pchip4801_kernel401_chi2": (
        "pchip_midpoint", "zero_at_measured_edges", 4801, 401, 2.0,
    ),
    "pchip9601_kernel401_chi2": (
        "pchip_midpoint", "zero_at_measured_edges", 9601, 401, 2.0,
    ),
}
STEPS = np.asarray([0.001, 0.0005, 0.00025], dtype=np.float64)
OFFSETS = np.asarray(
    [0.0, -0.002, 0.002, -0.001, 0.001, -0.0005, 0.0005,
     -0.00025, 0.00025], dtype=np.float64,
)


import cosmolike_des_3pcf_interface as ci


def digest(path):
    """Hash one exact dependency, reading bounded chunks rather than an NPZ copy."""
    result = hashlib.sha256()
    with Path(path).open(mode="rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            result.update(block)
    return result.hexdigest()


def save_json(path, record):
    """Atomically replace this study's progress record; never write input files."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
    temporary.write_text(content + "\n", encoding="utf-8")
    temporary.replace(target=path)


def save_arrays(path, arrays, metadata):
    """Atomically save owned numerical outputs and a JSON provenance record."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    output = dict(arrays)
    output["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, allow_nan=False),
    )
    with temporary.open(mode="wb") as stream:
        np.savez_compressed(stream, **output)
    temporary.replace(target=path)


def absolute_config(driver, config, origin):
    """Resolve every physical file before saving a case in another directory."""
    result = deepcopy(config)
    locations = (
        ("source", "path"),
        ("physics", "tables"),
        ("rows", "two_point"),
        ("rows", "aperture"),
        ("covariance", "path"),
    )
    for section, name in locations:
        result[section][name] = str(driver.resolve_path(
            configuration_path=origin, value=result[section][name],
        ))
    core = result["mean"]["two_point"]
    core["directory"] = str(driver.resolve_path(
        configuration_path=origin, value=core["directory"],
    ))
    return result


def source_archive(config, case, directory, original):
    """Replace source arrays while retaining every saved power/background value.

    Arguments:
      config = original configuration with absolute paths.
      case = name in CASES; directory = this model/case output directory.
      original = existing physical-table NPZ, never modified.
    Returns:
      path of the independent source archive. Supplied native FITS edges
      are retained even for the linear case so provenance remains complete.
    """
    from des3pcf.physical_provider import load_tables, save_tables
    from cobaya.likelihoods.des_3pcf.survey import reconstructed_sources
    from cobaya.likelihoods.des_3pcf.survey import read_source_distributions

    method, padding, count, _, _ = CASES[case]
    tables, metadata = load_tables(path=original)
    native_z, native, low, high = read_source_distributions(
        path=config["source"]["path"], nbin=config["source"]["nbin"],
        extension=config["source"]["extension"], with_edges=True,
    )
    nodes = np.linspace(start=0.0, stop=tables["z"][-1], num=count)
    sources, preparation = reconstructed_sources(
        native_z=native_z, native_sources=native, source_z=nodes,
        shifts=np.zeros(shape=native.shape[0]), reconstruction=method,
        padding=padding, native_low=low, native_high=high,
    )
    tables["source_z"] = nodes
    tables["sources"] = sources
    tables["source_native_z"] = native_z
    tables["source_native"] = native
    tables["source_native_low"] = low
    tables["source_native_high"] = high
    metadata.update(preparation)
    metadata["source_interpolation"] = method
    metadata["source_fits"] = config["source"]["path"]
    metadata["source_fits_sha256"] = digest(path=config["source"]["path"])
    metadata["source_integrals_before_normalization"] = (
        preparation["source_shift_normalizations"]
    )
    metadata["source_reconstruction_library"] = "NumPy linear interpolation"
    if method == "pchip_midpoint":
        metadata["source_reconstruction_library"] = "scipy.interpolate.PchipInterpolator"
    metadata["source_reconstruction_library_version"] = (
        scipy.__version__ if method == "pchip_midpoint" else np.__version__
    )
    metadata["source_policy_study"] = {
        "parent_power_archive": str(original),
        "parent_power_archive_sha256": digest(path=original),
        "no_camb": True,
        "native_definition": "unchanged samples and edges of the processed demo FITS",
    }
    destination = directory / "sources_and_physics.npz"
    if destination.exists():
        saved, record = load_tables(path=destination)
        if record != metadata or set(saved) != set(tables):
            raise ValueError(
                "Source archive metadata changed; use a new study directory."
            )
        for name in tables:
            if not np.array_equal(a1=saved[name], a2=tables[name]):
                raise ValueError(f"Source archive {name} changed; use a new directory.")
    else:
        save_tables(path=destination, tables=tables, metadata=metadata)
    return destination


def weighted_square(values, cholesky):
    """Compute v^T C^-1 v with the full joint covariance, including cross blocks.

    A matrix input has one complete 480-row vector per row. Whitening by
    its lower Cholesky factor avoids constructing a numerical inverse.
    This is not the norm obtained from the marginal Map3 covariance alone.
    """
    whitened = solve_triangular(a=cholesky, b=np.asarray(values).T, lower=True)
    return np.sum(whitened * whitened, axis=0)


def stencil_diagnostics(values):
    """Reuse nine saved values for three central and one-sided derivatives.

    Arguments:
      values = [9,...] in OFFSETS order; trailing axes retain their meaning.
    Returns:
      derivative4, fourth_difference, right_slope and left_slope at STEPS.
      The fourth-order first derivative uses [1,-8,0,8,-1]/(12h). The fourth
      difference is [1,-4,6,-4,1], with no interpretation as random noise.
    """
    lookup = {}
    for index, offset in enumerate(OFFSETS):
        lookup[float(offset).hex()] = values[index]
    center = lookup[float(0.0).hex()]
    outputs = {
        "derivative4": [],
        "fourth_difference": [],
        "right_slope": [],
        "left_slope": [],
    }
    for step in STEPS:
        left2 = lookup[float(-2.0 * step).hex()]
        left = lookup[float(-step).hex()]
        right = lookup[float(step).hex()]
        right2 = lookup[float(2.0 * step).hex()]
        outputs["derivative4"].append(
            (left2 - 8.0 * left + 8.0 * right - right2) / (12.0 * step),
        )
        outputs["fourth_difference"].append(
            left2 - 4.0 * left + 6.0 * center - 4.0 * right + right2,
        )
        outputs["right_slope"].append((right - center) / step)
        outputs["left_slope"].append((center - left) / step)
    for name in outputs:
        outputs[name] = np.asarray(outputs[name])
    outputs["one_sided_difference"] = (
        outputs["right_slope"] - outputs["left_slope"]
    )
    outputs["derivative_change"] = np.diff(outputs["derivative4"], axis=0)
    return outputs


def run_case(driver, project, model_name, case, original_config, origin,
             directory, threads):
    """Evaluate a single model/policy in sequence, saving each point immediately.

    All exact physical arrays and numerical controls belong to the case
    record. A resumed point is accepted only if that record still matches.
    File identities are checked again after each mean to avoid associating
    a changed binary or input with the initial dependency record.
    """

    from des3pcf.provenance import prediction_identity

    directory.mkdir(parents=True, exist_ok=True)
    config = absolute_config(driver=driver, config=original_config, origin=origin)
    original_tables = Path(config["physics"]["tables"])
    config["physics"]["tables"] = str(source_archive(
        config=config, case=case, directory=directory, original=original_tables,
    ))
    method, padding, source_count, kernel_count, chi_boost = CASES[case]
    config["physics"]["cosmology_mode"] = "fixed_tables"
    config["physics"]["source_interpolation"] = method
    config["physics"]["source_padding"] = padding
    config["physics"]["source_nz"] = source_count
    config["physics"]["kernel_nz"] = kernel_count
    config["threads"] = threads
    config["accuracy"] = ci.init_3pcf_accuracy(value={"chi_boost": chi_boost})
    projection = config["mean"]["aperture_settings"]["projection_settings"]
    projection.update({
        "signed_baseline": 1,
        "residual_scaling": 1,
        "shape_stretch": 10.1062201,
        "nshape": 160,
        "psi_scale": 0.02,
    })
    config["outputs"] = {
        "evaluate": str(directory / "center.npz"),
        "fisher": str(directory / "not_a_full_fisher.npz"),
    }
    config_path = directory / "configuration.json"
    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config != config:
            raise ValueError("Case configuration changed; choose a new output directory.")
    else:
        save_json(path=config_path, record=config)
    # Reuse the production schema. This also rejects an unapplied edge
    # policy patch before importing a compiled extension or preparing FFTs.
    config, config_path, config_hash = driver.load_configuration(
        path=config_path, command="evaluate",
    )
    if config["parameters"].get("dz1", 0.0) != 0.0:
        raise ValueError(
            "This support study requires the declared dz1 center to be zero."
        )
    rows = driver.rows_for_run(config=config, path=config_path)
    if rows.values.size != 480 or np.count_nonzero(rows.kind == "map3") != 80:
        raise ValueError("This study requires all 400 xi and 80 Map3 rows.")
    covariance = driver.read_matrix(
        settings=config["covariance"], path=config_path, rows=rows, positive=True,
    )
    cholesky = np.linalg.cholesky(covariance)
    compiled = driver.compiled_extension(config=config)
    second = config["mean"]["two_point"]
    two_point_path = Path(second["directory"]) / (second["module"] + ".so")

    def current_identity():
        """Fingerprint physical inputs and loaded implementation paths, without CAMB."""
        identity = prediction_identity(
            config=config, configuration_path=config_path, project=project,
            three_point_path=compiled.__file__, two_point_path=two_point_path,
        )
        identity["study_script_sha256"] = digest(path=__file__)
        identity["parent_configuration_sha256"] = digest(path=origin)
        identity["parent_tables_sha256"] = digest(path=original_tables)
        return identity

    identity = current_identity()
    manifest = {
        "format": "des3pcf-source-policy-study-v1",
        "model": model_name,
        "case": case,
        "configuration": config,
        "configuration_sha256": config_hash,
        "dependencies": identity,
        "dz1_offsets": OFFSETS.tolist(),
        "derivative_steps": STEPS.tolist(),
        "no_camb": True,
        "covariance_use": (
            "fixed full joint inverse metric, including xi/Map3 cross blocks"
        ),
        "interpretation": (
            "Deterministic source-support and resolution study; fourth differences "
            "are not assumed to be independent random jitter. No full Fisher rerun."
        ),
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError("Case dependencies changed; use a new output directory.")
    else:
        save_json(path=manifest_path, record=manifest)
    manifest_hash = digest(path=manifest_path)
    provider, _ = driver.provider_for_run(compiled=compiled, config=config,
                                          path=config_path)
    prediction = driver.prediction_for_run(
        compiled=compiled, config=config, path=config_path, rows=rows,
        provider=provider,
    )
    controls = ci.init_3pcf_accuracy(value=config["accuracy"])
    resolved = {
        "source_nz": ci.accuracy_count(accuracy=controls,
            baseline=source_count, stage="source_boost", endpoints=True,
        ),
        "model": ci.accuracy_model_settings(accuracy=controls,
            kernel_nz=kernel_count,
            table_boost=config["physics"].get("table_boost", 8),
            fit_nz=config["physics"].get("fit_nz", 33),
        ),
        "projection": dict(prediction.map_engine.projection_settings),
        "fft_ell_count": int(prediction.map_engine.ell.size),
        "fft_ell_bounds": prediction.map_engine.ell[[0, -1]].tolist(),
        "xi_ell_count": int(prediction.xi_ell.size),
    }
    save_json(path=directory / "resolved_numerics.json", record=resolved)
    points = []
    for index, offset in enumerate(OFFSETS):
        destination = directory / f"point_{index:02d}.npz"
        point_record = {
            "manifest_sha256": manifest_hash,
            "dz1": float(offset),
            "offset_index": index,
        }
        if destination.exists():
            with np.load(file=destination, allow_pickle=False) as archive:
                saved = json.loads(str(archive["metadata_json"].item()))
                if saved != point_record:
                    raise ValueError("Saved point identity changed; use a new directory.")
                arrays = {}
                for name in archive.files:
                    if name != "metadata_json":
                        arrays[name] = archive[name].copy()
        else:
            parameters = dict(config["parameters"])
            parameters["dz1"] = float(offset)
            mean = prediction(parameters=parameters)
            physical, tables, source_record = provider.get(parameters=parameters)
            radial = physical.radial()
            nodes = tables["source_z"]
            source = tables["sources"]
            area = np.asarray(source_record["source_shift_normalizations"])
            moment = np.trapz(y=source * nodes[None, :], x=nodes, axis=1)
            arrays = {
                "mean": mean,
                "source_z": nodes,
                "sources": source,
                "source_raw_area": area,
                "source_normalized_area": np.trapz(y=source, x=nodes, axis=1),
                "source_first_moment": moment,
                "source_raw_first_moment": moment * area,
                "shape_kernel_minimum_at_los": np.min(radial["shape"], axis=1),
                "shape_kernel_negative_los_count": np.sum(radial["shape"] < 0, axis=1),
                "row_labels": rows.labels,
            }
            for name, values in radial.items():
                arrays["radial_" + name] = values
            if current_identity() != identity:
                raise ValueError("A physical dependency changed during evaluation.")
            save_arrays(path=destination, arrays=arrays, metadata=point_record)
        if (arrays["mean"].shape != (480,)
                or not np.all(np.isfinite(arrays["mean"]))
                or not np.array_equal(a1=arrays["row_labels"], a2=rows.labels)):
            raise ValueError(
                "Saved point is not a finite vector in the required row order."
            )
        points.append(arrays)
        save_json(path=directory / "progress.json", record={
            "completed_points": index + 1,
            "required_points": OFFSETS.size,
            "model": model_name,
            "case": case,
        })
        print(
            f"{model_name}/{case}: saved point {index + 1}/9 dz1={offset:g}",
            flush=True,
        )

    means = np.stack(arrays=[point["mean"] for point in points])
    diagnostics = stencil_diagnostics(values=means)
    arrays = {
        "dz1_offsets": OFFSETS,
        "steps": STEPS,
        "means": means,
        "fixed_joint_covariance": covariance,
        "row_labels": rows.labels,
        "row_kind": rows.kind,
        "row_bins": rows.bins,
        "row_angles_radians": rows.angles,
    }
    arrays.update(diagnostics)
    for name in ("derivative4", "fourth_difference", "one_sided_difference",
                 "derivative_change"):
        arrays[name + "_joint_norm2"] = weighted_square(
            values=diagnostics[name], cholesky=cholesky,
        )
    arrays["displacement_joint_norm2"] = weighted_square(
        values=means - means[0], cholesky=cholesky,
    )
    source_metrics = ("source_raw_area", "source_normalized_area",
                      "source_first_moment", "source_raw_first_moment")
    for name in source_metrics:
        values = np.stack(arrays=[point[name] for point in points])
        arrays[name] = values
        source_diagnostics = stencil_diagnostics(values=values)
        for diagnostic, values in source_diagnostics.items():
            arrays[name + "_" + diagnostic] = values
    for name in ("shape_kernel_minimum_at_los", "shape_kernel_negative_los_count"):
        arrays[name] = np.stack(arrays=[point[name] for point in points])
    if current_identity() != identity:
        raise ValueError("Dependencies changed before completing the case diagnostics.")
    save_arrays(path=directory / "diagnostics.npz", arrays=arrays, metadata=manifest)
    summary = {
        "model": model_name,
        "case": case,
        "completed_points": len(points),
        "fourth_difference_joint_norm2": arrays["fourth_difference_joint_norm2"].tolist(),
        "derivative_joint_norm2": arrays["derivative4_joint_norm2"].tolist(),
        "derivative_change_joint_norm2": arrays["derivative_change_joint_norm2"].tolist(),
        "one_sided_difference_joint_norm2": arrays[
            "one_sided_difference_joint_norm2"
        ].tolist(),
        "shape_minima_at_los": arrays["shape_kernel_minimum_at_los"].tolist(),
        "shape_minimum_scope": (
            "Sampled only at the actual LOS nodes; not a continuous positivity proof."
        ),
        "source_model_scope": (
            "Linear native support and measured-edge PCHIP are distinct assumptions. "
            "PCHIP preserves selected support, not histogram cell probabilities."
        ),
    }
    save_json(path=directory / "summary.json", record=summary)
    return summary


def compare_cases(output, models):
    """Rescore completed policy/refinement differences with no new model calls.

    The pchip1201 case is only a named comparison baseline, not a truth
    model. Save the signed changes themselves as well as their joint
    covariance norms. Incomplete cases remain explicitly absent.
    """
    for model in models:
        reference_path = output / model / "pchip1201" / "diagnostics.npz"
        if not reference_path.exists():
            continue
        with np.load(file=reference_path, allow_pickle=False) as archive:
            reference_mean = archive["means"].copy()
            reference_derivative = archive["derivative4"].copy()
            covariance = archive["fixed_joint_covariance"].copy()
            labels = archive["row_labels"].copy()
            reference_record = json.loads(str(archive["metadata_json"].item()))
        cholesky = np.linalg.cholesky(covariance)
        report = {
            "reference": "pchip1201",
            "reference_is_truth": False,
            "metric": "full joint covariance inverse",
            "comparisons": [],
        }
        for case in CASES:
            path = output / model / case / "diagnostics.npz"
            if case == "pchip1201" or not path.exists():
                continue
            with np.load(file=path, allow_pickle=False) as archive:
                case_record = json.loads(str(archive["metadata_json"].item()))
                for name in ("implementation", "study_script_sha256",
                             "parent_configuration_sha256", "parent_tables_sha256"):
                    if (case_record["dependencies"][name]
                            != reference_record["dependencies"][name]):
                        raise ValueError(
                            "Cross-case implementations or parents differ; "
                            "start a new study."
                        )
                if (not np.array_equal(a1=archive["row_labels"], a2=labels)
                        or not np.array_equal(
                            a1=archive["fixed_joint_covariance"], a2=covariance,
                        )):
                    raise ValueError(
                        "Cross-case rows/covariance differ; stop comparison."
                    )
                mean_difference = archive["means"] - reference_mean
                derivative_difference = archive["derivative4"] - reference_derivative
            arrays = {
                "dz1_offsets": OFFSETS,
                "steps": STEPS,
                "mean_difference": mean_difference,
                "derivative_difference": derivative_difference,
                "mean_difference_joint_norm2": weighted_square(
                    values=mean_difference, cholesky=cholesky,
                ),
                "derivative_difference_joint_norm2": weighted_square(
                    values=derivative_difference, cholesky=cholesky,
                ),
            }
            comparison = {
                "case": case,
                "reference": "pchip1201",
                "difference": "case minus reference",
                "center_mean_joint_norm2": float(
                    arrays["mean_difference_joint_norm2"][0],
                ),
                "derivative_joint_norm2": arrays[
                    "derivative_difference_joint_norm2"
                ].tolist(),
            }
            save_arrays(path=output / model / f"{case}_minus_pchip1201.npz",
                        arrays=arrays, metadata=comparison)
            report["comparisons"].append(comparison)
        save_json(path=output / model / "case_comparisons.json", record=report)


def main():
    """Select explicit subsets and execute models/cases sequentially, never in a pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("nla", "tatt"),
                        default=["nla", "tatt"])
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES),
                        default=list(CASES))
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory, or explicit --resume.")
    if not output.exists() and args.resume:
        raise FileNotFoundError("--resume needs an existing study directory.")
    if (len(set(args.models)) != len(args.models)
            or len(set(args.cases)) != len(args.cases)):
        raise ValueError("Select each model and case only once.")
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project.parents[1] / "external_modules/code/cosmolike_core"))
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project.parents[1] / "external_modules/code/cosmolike_core"))
    import run_3pcf as driver

    configs = {}
    covariance_path = None
    # Check all requested baseline files before creating the output root.
    # Both models must use the same fixed covariance bytes in this study.
    for model in args.models:
        origin = project / "configs" / f"des_y3_{model}_demo_fisher.yaml"
        config, origin, _ = driver.load_configuration(path=origin, command="evaluate")
        if config["covariance"]["mode"] != "file":
            raise ValueError("Source study needs the existing fixed joint covariance.")
        current = driver.resolve_path(origin, config["covariance"]["path"])
        if covariance_path is not None and digest(current) != digest(covariance_path):
            raise ValueError("NLA and TATT must share identical fixed covariance bytes.")
        covariance_path = current
        configs[model] = (config, origin)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for model in args.models:
        config, origin = configs[model]
        for case in args.cases:
            result = run_case(
                driver=driver, project=project, model_name=model, case=case,
                original_config=config, origin=origin,
                directory=output / model / case, threads=args.threads,
            )
            results.append(result)
            compare_cases(output=output, models=[model])
    save_json(path=output / "last_invocation.json", record={
        "models": args.models,
        "cases": args.cases,
        "results": results,
        "no_camb": True,
        "threads": args.threads,
        "blas_threads": 1,
    })


if __name__ == "__main__":
    main()
