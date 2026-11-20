"""Compare prepared interpolation stages in all 80 finite-SAS aperture rows.

The physical tables, source curves, angular support, FFT and covariance are
fixed. Only projection and multipole interpolation change. This separates
their errors from CAMB, source reconstruction and estimator assumptions.
Both the marginal aperture norm and the full joint norm with zero xi error
are recorded. They differ because the joint measurement has cross covariance.
Neither turns this single-point check into a derivative validation.

Run from the project directory with --output pointing to a new directory.
Every completed case is saved before the next physical calculation starts.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

for variable in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OMP_THREAD_LIMIT"] = "4"

import numpy as np
from scipy.linalg import solve_triangular


def main():
    """Save NLA/TATT means and differences from the finest declared case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    destination = options.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    project = Path.cwd()
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project.parents[1] / "external_modules/code/cosmolike_core"))
    import run_3pcf as driver

    report = {"scope": "fixed-source aperture-mean interpolation", "models": {}}
    for model_name in ("nla", "tatt"):
        config, path, digest = driver.load_configuration(
            path=project / f"configs/des_y3_{model_name}_demo_fisher.yaml",
            command="evaluate",
        )
        config["physics"]["cosmology_mode"] = "fixed_tables"
        config["threads"] = 4
        compiled = driver.compiled_extension(config=config)
        provider, _ = driver.provider_for_run(
            compiled=compiled, config=config, path=path,
        )
        full_rows = driver.rows_for_run(config=config, path=path)
        selected = np.flatnonzero(full_rows.kind == "map3")
        rows = full_rows.select(indices=selected)
        covariance = driver.read_matrix(
            settings=config["covariance"], path=path,
            rows=full_rows, positive=True,
        )
        joint_lower = np.linalg.cholesky(covariance)
        covariance = covariance[np.ix_(selected, selected)]
        lower = np.linalg.cholesky(covariance)
        settings = config["mean"]["aperture_settings"]["projection_settings"]
        minimum = min(1.0e-6, 0.25 * settings["ell_min"] / settings["ell_max"])
        base = {
            "signed_baseline": 1,
            "residual_scaling": 1,
            "shape_stretch": 0.5 * np.log(0.5 / minimum),
            "nshape": 160,
            "psi_scale": 0.02,
        }
        finer_projection = {
            "nr": 63, "nu": 47, "nv": 47, "np": 63, "nshape": 319,
        }
        finer_multipoles = {
            "nell": 127, "npsi": 159, "multipole_boost": 4,
        }
        cases = (
            ("corrected", {}),
            ("multipoles", finer_multipoles),
            ("projection", finer_projection),
            ("both", dict(finer_projection, **finer_multipoles)),
        )
        values = {}
        records = []
        for name, changes in cases:
            case = deepcopy(config)
            controls = case["mean"]["aperture_settings"]["projection_settings"]
            controls.update(base)
            controls.update(changes)
            prediction = driver.prediction_for_run(
                compiled=compiled, config=case, path=path,
                rows=rows, provider=provider,
            )
            mean = prediction(parameters=case["parameters"])
            values[name] = mean
            record = {
                "model": model_name,
                "case": name,
                "projection_settings": controls,
                "configuration_sha256": digest,
                "execution_overrides": {"cosmology_mode": "fixed_tables"},
                "saved_covariance_scope": "marginal_map3_block",
                "comparison_reference": "both: finite refined calculation",
                "likelihood_correction": "none",
            }
            np.savez_compressed(
                file=destination / f"{model_name}_{name}.npz",
                mean=mean, row_labels=rows.labels, covariance=covariance,
                metadata_json=np.asarray(json.dumps(record, sort_keys=True)),
            )
            records.append(record)
            print(json.dumps({"model": model_name, "completed": name}), flush=True)
            del prediction
        for record in records:
            difference = values[record["case"]] - values["both"]
            whitened = solve_triangular(a=lower, b=difference, lower=True)
            record["delta_chi2_to_both"] = float(whitened @ whitened)
            joint_difference = np.zeros(full_rows.values.size)
            joint_difference[selected] = difference
            joint_whitened = solve_triangular(
                a=joint_lower, b=joint_difference, lower=True,
            )
            record["joint_delta_chi2_to_both_zero_xi_error"] = float(
                joint_whitened @ joint_whitened
            )
            record["maximum_diagonal_sigma_error"] = float(np.max(
                np.abs(difference) / np.sqrt(np.diag(covariance))
            ))
        report["models"][model_name] = records
        (destination / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"model": model_name, "comparisons": records}), flush=True)


if __name__ == "__main__":
    main()
