"""Exercise the installed Nautilus checkpoint API on an analytic likelihood.

This bounded check evaluates no cosmology. Its two parameters have a uniform
prior on [-5,5] and a normalized unit Gaussian likelihood. The analytic
posterior and evidence are therefore known, including the finite prior
boundary. The first run stops early, and a new sampler resumes its HDF5
checkpoint. This checks the sampling library used by the project examples;
the separate inference tests check target identity and joint covariance use.

Run from the project directory with the activated Cocoa Python. The explicit
output directory must be new; the study never overwrites an earlier record.
"""

import argparse
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from nautilus import Prior, Sampler
from scipy.special import erf


def gaussian_loglike(point):
    """Return the normalized two-dimensional Gaussian log density."""
    values = np.asarray(point, dtype=np.float64)
    return -0.5 * float(values @ values) - np.log(2.0 * np.pi)


def run(output):
    """Stop and resume the same target, then compare with analytic moments."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = str(output / "checkpoint.hdf5")
    prior = Prior()
    prior.add_parameter(key="first", dist=(-5.0, 5.0))
    prior.add_parameter(key="second", dist=(-5.0, 5.0))
    settings = {
        "prior": prior, "likelihood": gaussian_loglike, "pass_dict": False,
        "n_live": 100, "n_networks": 1, "n_batch": 50, "seed": 23719,
        "filepath": checkpoint,
    }
    initial = Sampler(**settings, resume=False)
    initial_converged = initial.run(
        n_eff=1000, n_like_max=200, discard_exploration=True, verbose=False,
    )
    first_count = int(initial.n_like)
    assert not initial_converged, "The intended interrupted run already converged."
    resumed = Sampler(**settings, resume=True)
    assert resumed.n_like == first_count
    converged = resumed.run(
        n_eff=1000, n_like_max=10000, discard_exploration=True, verbose=False,
    )
    points, log_weights, log_likes = resumed.posterior()
    weights = np.exp(log_weights)
    mean = np.sum(weights[:, None] * points, axis=0)
    centered = points - mean
    covariance = (centered * weights[:, None]).T @ centered
    retained_probability = erf(5.0 / np.sqrt(2.0))
    expected_log_evidence = 2.0 * np.log(retained_probability / 10.0)
    boundary_density = np.exp(-0.5 * 5.0**2) / np.sqrt(2.0 * np.pi)
    expected_variance = 1.0 - 10.0 * boundary_density / retained_probability
    record = {
        "scope": "Analytic Gaussian sampler/resume check; no DES posterior",
        "initial_converged": bool(initial_converged),
        "initial_evaluations": first_count,
        "converged": bool(converged),
        "total_evaluations": int(resumed.n_like),
        "effective_samples": float(resumed.n_eff),
        "mean": mean.tolist(), "covariance": covariance.tolist(),
        "expected_variance": float(expected_variance),
        "log_evidence": float(resumed.log_z),
        "expected_log_evidence": float(expected_log_evidence),
    }
    # These tolerances are fixed before running the seeded experiment. They
    # test units, weights and resume behavior, not exact Monte Carlo equality.
    assert converged
    np.testing.assert_allclose(actual=np.sum(weights), desired=1.0, atol=1e-12)
    assert np.max(np.abs(mean)) < 0.15
    assert np.max(np.abs(covariance - expected_variance * np.eye(2))) < 0.2
    assert abs(resumed.log_z - expected_log_evidence) < 0.3
    expected_likes = -0.5 * np.sum(points**2, axis=1) - np.log(2.0 * np.pi)
    np.testing.assert_allclose(actual=log_likes, desired=expected_likes)
    (output / "summary.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    run(output=arguments.output)
