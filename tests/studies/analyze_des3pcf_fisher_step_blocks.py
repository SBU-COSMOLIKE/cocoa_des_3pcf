"""Inspect historical parameter-step changes separately in xi and Map3 rows.

Each substitution keeps the full covariance and all priors. Block effects
are therefore not independent or additive. This script evaluates no model.
Place it beside analyze_des3pcf_fisher_amplification.py under tests/studies.
"""

import json
import os
from pathlib import Path

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
from threadpoolctl import threadpool_limits

from analyze_des3pcf_fisher_amplification import (
    augmented,
    digest,
    marginal,
    paths_from_arguments,
    read_arrays,
)


def inspect_blocks(arrays):
    """Substitute saved coarse-step entries into the fine-step derivative matrix.

    Arguments:
      arrays = one complete historical Fisher archive, including priors and C.
    Returns:
      Relative target FoMs for Omega_m, Hubble h and both columns together,
      replacing either xi entries, Map3 entries or the entire columns.
    """
    names = arrays["parameter_names"].tolist()
    target = [names.index("w0"), names.index("wa")]
    fine = arrays["derivatives_h2"]
    coarse = arrays["derivatives_h"]

    def merit(jacobian):
        """Use nuisance QR to avoid forming or inverting the Fisher Gram matrix."""
        response = augmented(
            jacobian=jacobian,
            covariance=arrays["fiducial_covariance"],
            prior=arrays["fisher_prior"],
        )
        effective, _ = marginal(array=response, target=target)
        return np.sqrt(np.linalg.det(a=effective))

    reference = merit(jacobian=fine)
    groups = [
        ["omega_m"],
        ["h"],
        ["omega_m", "h"],
    ]
    blocks = [
        ("xi", arrays["two_point_indices"]),
        ("map3", arrays["three_point_indices"]),
        ("all", np.arange(start=0, stop=fine.shape[0])),
    ]
    result = {}
    for members in groups:
        columns = [names.index(name) for name in members]
        group = {}
        for label, rows in blocks:
            derivative = fine.copy()
            derivative[np.ix_(rows, columns)] = coarse[np.ix_(rows, columns)]
            group[label] = float(merit(jacobian=derivative)/reference)
        result["_".join(members)] = group
    return result


def main():
    """Read the two historical bases and save hashes with the substitution ratios."""
    project, output = paths_from_arguments(description=__doc__)
    limit = threadpool_limits(limits=1, user_api="blas")
    report = {}
    for model in ("nla", "tatt"):
        path = project/"runs"/f"y3_{model}_measured_edge_demo_fisher.npz"
        arrays, input_record = read_arrays(path=path)
        result = inspect_blocks(arrays=arrays)
        result["input"] = input_record
        report[model] = result
    report["inspection"] = {
        "interpretation": (
            "Historical array substitutions, not newly evaluated forecasts."
        ),
        "script_sha256": digest(path=Path(__file__)),
        "helper_sha256": digest(
            path=Path(__file__).with_name("analyze_des3pcf_fisher_amplification.py"),
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output/"step_blocks.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(destination)


if __name__ == "__main__":
    main()
