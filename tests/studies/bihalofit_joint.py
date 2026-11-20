"""Compare internal and supplied BiHalofit on the declared joint DES rows.

The configuration selects the measurement, all row labels, its fixed joint
covariance and the projection settings. Every input file is copied and hashed
before the comparison; the explicit mask keeps every selected row. The study
reports both the norm of the model difference and the change in chi-squared
against the supplied data. The latter also depends on the residual of the
reference model, and therefore need not equal the norm of the difference.

CAMB supplies only the missing background density fraction in this study.
Power, distances, growth and source curves are read from the saved physical
tables. Agreement of CAMB H(z) with those saved tables is checked before the
two are combined. No power-spectrum or transfer calculation is requested.
The final pair of comparisons holds the bispectrum sampling fixed while
replacing only CAMB's matter fraction with the internal radiation-free
prescription. This separates that background choice from table refinement.

Run from the project directory with an explicitly selected compiled library:
  python tests/studies/bihalofit_joint.py --extension /path/to/interface \
      --output /path/to/results --threads 6
"""

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


# Matrix factorizations use one BLAS worker. Only the C projection may use
# the requested OpenMP team, whose maximum is the six-worker run limit.
for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"


def write_json(path, record):
    """Write a finite JSON record with every result still available on failure."""
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def background_fraction(tables, parameters, np):
    """Read CAMB's total matter fraction on the saved background grid.

    Arguments:
      tables = physical tables with z and dz/dchi in h/Mpc.
      parameters = the cosmology declared by their producer.
      np = the already imported NumPy module.
    Returns:
      density fractions and the largest relative H(z) consistency residual.
    Raises:
      ValueError if CAMB and the stored expansion rates differ by over 1e-8.
    """
    import camb

    h = parameters["h"]
    settings = camb.CAMBparams()
    settings.set_cosmology(
        H0=100.0 * h,
        ombh2=parameters["omega_b"] * h**2,
        omch2=(parameters["omega_m"] - parameters["omega_b"]) * h**2,
        mnu=parameters.get("mnu", 0.0),
        omk=0.0,
        num_massive_neutrinos=parameters.get("num_massive_neutrinos", 1),
        nnu=parameters.get("nnu", 3.046),
    )
    settings.omch2 -= settings.omnuh2
    settings.set_dark_energy(
        w=parameters["w0"], wa=parameters["wa"], dark_energy_model="ppf",
    )
    results = camb.get_background(params=settings)
    redshift = tables["z"]
    stored_hubble = tables["dz_dchi"] * (299792.458 * h)
    supplied_hubble = results.hubble_parameter(z=redshift)
    residual = float(np.max(np.abs(supplied_hubble / stored_hubble - 1.0)))
    if residual > 1.0e-8:
        raise ValueError(
            f"Saved/CAMB H(z) differ by {residual}; regenerate physical tables "
            "for the declared cosmology before combining their backgrounds."
        )
    density = results.get_Omega(var="cdm", z=redshift)
    density += results.get_Omega(var="baryon", z=redshift)
    density += results.get_Omega(var="nu", z=redshift)
    return np.asarray(density), residual


def table_settings(counts, maximum_z):
    """Declare fixed bounds and independently refinable uniform coordinates.

    The scale bounds cover the configured 0.001..300000 multipole table
    and its line-of-sight distances. The shape range includes the small
    side ratios used when that table prepares squeezed triangles. Queries
    beyond these bounds must fail; no clamping is requested.
    """
    return {
        "z": {"lower": 0.0, "upper": maximum_z, "count": counts[0]},
        "logk": {"lower": -18.0, "upper": 12.0, "count": counts[1]},
        "logu": {"lower": -24.0, "upper": 0.0, "count": counts[2]},
        "v": {"lower": 0.0, "upper": 1.0, "count": counts[3]},
        "boost": 1,
    }


def radiation_free_fraction(tables, np):
    """Reproduce the internal fit's background as a comparison input.

    Matter scales as a^-3 and dark energy follows w(a)=w0+wa*(1-a).
    The internal fit omits radiation in this fraction. Sampling that
    prescription on the same grid as the CAMB fraction allows a comparison
    that changes only the background supplied to the Python equation.
    Power, growth, distances and source curves remain the saved inputs.
    This array is a diagnostic; it is not the supplied route's default.
    """
    scale = 1.0 / (1.0 + tables["z"])
    matter = tables["omega_m"] / scale**3
    dark = tables["omega_de"] * scale**(
        -3.0 * (1.0 + tables["w0"] + tables["wa"])
    ) * np.exp(-3.0 * tables["wa"] * (1.0 - scale))
    curvature = (1.0 - tables["omega_m"] - tables["omega_de"]) / scale**2
    return matter / (matter + dark + curvature)


def run(options):
    """Pin inputs, evaluate each route, and retain both likelihood diagnostics.

    Arguments:
      options = parsed configuration, extension, output and thread selections.
    Returns:
      nothing; writes copied inputs, vectors and report.json under output.
    Raises:
      ValueError for inconsistent physical inputs or nonpositive covariance.
      FileExistsError if the output directory already exists.
    """
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project.parents[1] / "external_modules/code/cosmolike_core"))
    import run_3pcf as driver
    import numpy as np
    from des3pcf.physical_provider import load_tables, save_tables
    from des3pcf.provenance import prediction_identity, verify_prediction_identity

    sys.path.insert(0, str(options.extension.resolve()))
    import cosmolike_des_3pcf_interface as compiled

    config, config_origin, _ = driver.load_configuration(
        path=options.config, command="evaluate",
    )
    options.output = options.output.resolve()
    options.output.mkdir(parents=True, exist_ok=False)
    inputs = options.output / "inputs"
    inputs.mkdir()
    file_hashes = {}
    # Copy the files whose contents define data, sources and the metric.
    # The input record names both original and copied paths and their hash.
    locations = (
        ("physics", "tables"),
        ("source", "path"),
        ("rows", "two_point"),
        ("rows", "aperture"),
        ("covariance", "path"),
    )
    for group, name in locations:
        value = config[group].get(name)
        if value is None:
            continue
        original = driver.resolve_path(configuration_path=config_origin, value=value)
        copied = inputs / (group + "_" + name + original.suffix)
        shutil.copyfile(src=original, dst=copied)
        file_hashes[group + "." + name] = {
            "original": str(original),
            "copied": str(copied),
            "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
        }
        config[group][name] = str(copied)
    second = config["mean"]["two_point"]
    second["directory"] = str(driver.resolve_path(
        configuration_path=config_origin, value=second["directory"],
    ))
    tables, metadata = load_tables(path=config["physics"]["tables"])
    density, expansion_error = background_fraction(
        tables=tables, parameters=config["parameters"], np=np,
    )
    tables["omega_m_z"] = density
    metadata["omega_m_z_definition"] = (
        "CAMB Omega_cdm(z)+Omega_baryon(z)+Omega_nu_massive(z)"
    )
    physical_path = inputs / "physical_with_background.npz"
    save_tables(path=physical_path, tables=tables, metadata=metadata)
    comparison_tables = dict(tables)
    comparison_tables["omega_m_z"] = radiation_free_fraction(tables=tables, np=np)
    comparison_metadata = dict(metadata)
    comparison_metadata["omega_m_z_definition"] = (
        "radiation-free internal w0-wa fraction sampled on the same background grid"
    )
    comparison_path = inputs / "physical_radiation_free_comparison.npz"
    save_tables(
        path=comparison_path, tables=comparison_tables, metadata=comparison_metadata,
    )
    config["physics"]["tables"] = str(physical_path)
    config["physics"]["cosmology_mode"] = "fixed_tables"
    config["threads"] = options.threads
    config_path = options.output / "resolved_configuration.json"
    write_json(path=config_path, record=config)
    rows = driver.rows_for_run(config=config, path=config_path)
    covariance = driver.read_matrix(
        settings=config["covariance"], path=config_path, rows=rows, positive=True,
    )
    factor = np.linalg.cholesky(a=covariance)
    selected = np.ones(shape=rows.values.size, dtype=np.int32)
    metric_path = inputs / "metric_and_data.npz"
    np.savez(file=metric_path, data=rows.values, covariance=covariance,
             mask=selected, row_labels=rows.labels)
    native_two = Path(second["directory"]) / (second["module"] + ".so")
    identity = prediction_identity(
        config=config,
        configuration_path=config_path,
        project=project,
        three_point_path=compiled.__file__,
        two_point_path=native_two,
    )
    compiled.set_omp_threads(nthreads=options.threads)
    report = {
        "scope": "declared joint DES Y3 example; unchanged estimator assumptions",
        "background_definition": metadata["omega_m_z_definition"],
        "maximum_relative_hubble_residual": expansion_error,
        "lensing_omega_m0": float(tables["omega_m"]),
        "provider_omega_m0": float(density[0]),
        "relative_omega_m0_difference": float(density[0] / tables["omega_m"] - 1.0),
        "maximum_relative_background_difference": float(np.max(np.abs(
            density / comparison_tables["omega_m_z"] - 1.0,
        ))),
        "original_files": file_hashes,
        "metric_and_data_sha256": hashlib.sha256(metric_path.read_bytes()).hexdigest(),
        "resolved_configuration_sha256": hashlib.sha256(
            config_path.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction_identity": identity,
        "threads": options.threads,
        "row_count": rows.values.size,
        "cases": [],
    }
    cases = [("internal", None)]
    cases.extend([
        ("coarse", (9, 65, 33, 9)),
        ("medium", (17, 97, 49, 17)),
        ("fine", (33, 129, 65, 25)),
        ("refined", (49, 193, 97, 33)),
        ("refined_radiation_free", (49, 193, 97, 33)),
    ])
    reference = None
    previous = None
    reference_chi2 = None
    for name, counts in cases:
        candidate = deepcopy(config)
        candidate["physics"]["bispectrum_source"] = "internal"
        if counts is not None:
            candidate["physics"]["bispectrum_source"] = "supplied"
            candidate["physics"]["bispectrum_table"] = table_settings(
                counts=counts, maximum_z=float(tables["z"][-1]),
            )
        if name == "refined_radiation_free":
            candidate["physics"]["tables"] = str(comparison_path)
        candidate_identity = prediction_identity(
            config=candidate, configuration_path=config_path, project=project,
            three_point_path=compiled.__file__, two_point_path=native_two,
        )
        provider, _ = driver.provider_for_run(
            compiled=compiled, config=candidate, path=config_path,
        )
        prediction = driver.prediction_for_run(
            compiled=compiled, config=candidate, path=config_path,
            rows=rows, provider=provider,
        )
        mean = prediction(parameters=candidate["parameters"])
        candidate_completed = prediction_identity(
            config=candidate, configuration_path=config_path, project=project,
            three_point_path=compiled.__file__, two_point_path=native_two,
        )
        verify_prediction_identity(
            recorded=candidate_identity, current=candidate_completed,
            description="Joint bispectrum case " + name,
        )
        residual = np.linalg.solve(a=factor, b=mean - rows.values)
        chi2 = float(residual @ residual)
        if reference is None:
            reference = mean.copy()
            reference_chi2 = chi2
        difference = np.linalg.solve(a=factor, b=mean - reference)
        result = {
            "name": name,
            "coarse_counts": counts,
            "prediction_identity": candidate_identity,
            "chi2": chi2,
            "delta_chi2_data": chi2 - reference_chi2,
            "squared_difference_norm": float(difference @ difference),
            "two_point_bitwise_equal": bool(np.array_equal(
                mean[rows.kind != "map3"], reference[rows.kind != "map3"],
            )),
        }
        if previous is not None:
            step = np.linalg.solve(a=factor, b=mean - previous)
            result["squared_difference_previous"] = float(step @ step)
        if provider._active_baseline is not None:
            result["baseline"] = provider._active_baseline["metadata"]
        mean_path = options.output / (name + "_mean.npz")
        np.savez(file=mean_path, mean=mean)
        result["mean_npz_sha256"] = hashlib.sha256(mean_path.read_bytes()).hexdigest()
        report["cases"].append(result)
        write_json(path=options.output / "report.json", record=report)
        print(name, "delta chi2", result["delta_chi2_data"],
              "difference norm squared", result["squared_difference_norm"], flush=True)
        previous = mean.copy()
        del prediction, provider
        gc.collect()
    completed = prediction_identity(
        config=config,
        configuration_path=config_path,
        project=project,
        three_point_path=compiled.__file__,
        two_point_path=native_two,
    )
    verify_prediction_identity(
        recorded=identity, current=completed, description="Joint bispectrum comparison",
    )
    report["input_identity_unchanged"] = True
    write_json(path=options.output / "report.json", record=report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/des_y3_nla_demo_fisher.yaml",
    ))
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6, choices=range(1, 7))
    run(options=parser.parse_args())
