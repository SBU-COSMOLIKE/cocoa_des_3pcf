"""Check the xi integration before using it in joint Fisher derivatives.

The ell integral contains oscillating Bessel functions. A small logarithmic
grid can alias those oscillations even when the angular power spectrum is
smooth. This study varies the Fourier and angular-bin quadratures separately
at one physical point. Every discrepancy is a vector-to-vector quadratic
distance using the same two-point covariance, never a change of fit to data.

Run from the project directory with its existing compiled modules:
  python tests/studies/xi_refinement.py configs/des_y3_tatt_demo_fisher.yaml

The declared physical table must already exist. The study writes arrays and
resolved settings below chains/xi_refinement; it does not change a configuration
or select an accuracy threshold on behalf of a forecast.
"""

import argparse
from copy import deepcopy
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
from scipy.linalg import solve_triangular

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))
import run_3pcf as driver


def run(configuration, output):
    """Save xi vectors for fixed cosmology and an explicit quadrature ladder.

    Arguments:
      configuration = joint YAML with physical tables and covariance present.
      output = destination directory for vectors.npz and summary.json.
    Returns:
      list of settings and quadratic residuals relative to the finest case.
    Side effects:
      Runs C serially across cases, with at most four workers per case.
    """
    config, path, digest = driver.load_configuration(
        path=configuration, command="evaluate",
    )
    output = Path(output)
    destination = output / "vectors.npz"
    if destination.exists():
        raise FileExistsError(f"{destination} exists; choose a distinct study output.")
    all_rows = driver.rows_for_run(config=config, path=path)
    selected = np.flatnonzero(all_rows.kind != "map3")
    rows = all_rows.select(indices=selected)
    matrix = driver.read_matrix(
        settings=config["covariance"], path=path, rows=all_rows, positive=True,
    )
    covariance = matrix[np.ix_(selected, selected)]
    factor = np.linalg.cholesky(covariance)
    compiled = driver.compiled_extension(config=config)
    provider, provenance = driver.provider_for_run(
        compiled=compiled, config=config, path=path,
    )
    parameters = dict(config["ia"])
    parameters.update(config["parameters"])
    # The first ladder fixes angular averaging. The second changes only
    # that averaging at a fine Fourier grid; the final cases extend support.
    cases = []
    for count in (256, 512, 1024, 2048, 4096, 8192, 16384):
        cases.append((count, 32, 40000.0))
    for count in (64, 128):
        cases.append((16384, count, 40000.0))
    cases.append((32768, 128, 80000.0))
    cases.append((32768, 256, 80000.0))
    vectors = []
    records = []
    for ell_count, theta_count, ell_upper in cases:
        settings = deepcopy(config)
        settings["mean"]["xi_settings"]["ell"]["count"] = ell_count
        settings["mean"]["xi_settings"]["ell"]["upper"] = ell_upper
        settings["mean"]["xi_settings"]["theta_count"] = theta_count
        prediction = driver.prediction_for_run(
            compiled=compiled, config=settings, path=path, rows=rows,
            provider=provider,
        )
        vector = prediction(parameters=parameters)
        vectors.append(vector)
        records.append({
            "ell_count": ell_count,
            "theta_count": theta_count,
            "ell_upper": ell_upper,
        })
        print(json.dumps(records[-1]), flush=True)
    reference = vectors[-1]
    for index, vector in enumerate(vectors):
        residual = vector - reference
        whitened = solve_triangular(a=factor, b=residual, lower=True)
        records[index]["quadratic_distance_to_finest"] = float(whitened @ whitened)
        records[index]["max_residual_over_row_sigma"] = float(
            np.max(np.abs(residual) / np.sqrt(np.diag(covariance)))
        )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file=destination, vectors=np.asarray(vectors), covariance=covariance,
        row_labels=rows.labels,
    )
    report = {
        "configuration_sha256": digest,
        "configuration": config,
        "physical_provenance": provenance,
        "cases": records,
        "interpretation": "Fiducial quadrature check; no derivative or survey-model convergence claim.",
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return records


def main():
    """Read the declared configuration and run the bounded serial ladder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT / "chains/xi_refinement")
    arguments = parser.parse_args()
    report = run(configuration=arguments.configuration, output=arguments.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
