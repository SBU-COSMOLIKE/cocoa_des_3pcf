"""Check that the joint dataset keeps no state from an earlier dataset.

ci.init_joint_data replaces the whole joint state: data, covariance,
selected rows, compression, the Cholesky factor and the finite-simulation
settings. Dataset A, then a dataset B that differs in every one of these
(size, selected rows, compression shape, mock count), then A again must
give exactly what a fresh interpreter gives for A alone. A refused
dataset must leave no dataset at all, not the previous one.
"""

import json
import subprocess
import sys

import numpy as np
import pytest


# Each dataset is built from a seed, so the child interpreter rebuilds the
# same arrays. A has six rows, keeps five and compresses them to three with
# 400 mocks; B has nine rows, keeps four, no compression, Gaussian.
DATASETS = """
import numpy as np
def dataset(name):
    size, kept, modes, mocks, seed = {"A": (6, [5, 0, 2, 3, 1], 3, 400, 1),
                                      "B": (9, [8, 1, 4, 6], 0, 0, 2)}[name]
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((size, size))
    covariance = factor @ factor.T + size * np.eye(size)
    covariance = 0.5 * (covariance + covariance.T)
    compression = (rng.standard_normal((modes, len(kept))) if modes
                   else np.empty((0, 0)))
    return dict(data=rng.standard_normal(size), covariance=covariance,
                indices=np.array(kept, dtype=np.int32), compression=compression,
                mock_count=mocks, parameter_count=2 if mocks else 0)
def summary(compiled, name):
    prediction = np.random.default_rng(7).standard_normal(dataset(name)["data"].size)
    return dict(chi2=compiled.compute_joint_chi2(prediction=prediction).hex(),
                loglike=compiled.compute_joint_loglike(prediction=prediction).hex(),
                exponent=float(compiled.get_joint_exponent()).hex(),
                data=[value.hex() for value in compiled.get_joint_data()],
                covariance=[value.hex() for value in compiled.get_joint_covariance().ravel()])
"""

FRESH = DATASETS + """
import json
import cosmolike_des_3pcf_interface as compiled
compiled.init_joint_data(**dataset("A"))
print(json.dumps(summary(compiled, "A")))
"""


def test_dataset_sequence_matches_a_fresh_interpreter(compiled):
    """A, B, A in one process equals A in a fresh interpreter, bit for bit."""
    namespace = {}
    exec(DATASETS, namespace)
    dataset, summary = namespace["dataset"], namespace["summary"]

    # Step 1: A, then B (checked against its own expected size), then A.
    compiled.init_joint_data(**dataset("A"))
    first = summary(compiled, "A")
    assert float.fromhex(first["exponent"]) > 0.0
    compiled.init_joint_data(**dataset("B"))
    assert compiled.get_joint_data().size == 4
    # B is Gaussian: no finite-simulation exponent survives from A.
    assert compiled.get_joint_exponent() == 0.0
    compiled.init_joint_data(**dataset("A"))
    again = summary(compiled, "A")
    assert again == first

    # Step 2: a fresh interpreter that only ever saw A.
    result = subprocess.run(args=[sys.executable, "-B", "-c", FRESH],
                            capture_output=True, text=True, check=True, timeout=120)
    fresh = json.loads(result.stdout.strip().splitlines()[-1])
    assert fresh == first


def test_refused_dataset_leaves_no_dataset(compiled):
    """After a refused init_joint_data, nothing can be scored.

    A caller that catches the refusal must not go on scoring predictions
    against the dataset installed before it.
    """
    namespace = {}
    exec(DATASETS, namespace)
    dataset = namespace["dataset"]

    # Step 1: a valid dataset that scores.
    compiled.init_joint_data(**dataset("A"))
    prediction = np.zeros(dataset("A")["data"].size)
    assert np.isfinite(compiled.compute_joint_chi2(prediction=prediction))

    # Step 2: a refused dataset: one entry between two selected rows
    # breaks the symmetry of the selected covariance.
    refused = dataset("B")
    first, second = refused["indices"][:2]
    refused["covariance"] = refused["covariance"].copy()
    refused["covariance"][first, second] += 0.5
    with pytest.raises(ValueError, match="not symmetric"):
        compiled.init_joint_data(**refused)

    # Step 3: neither the old nor the refused dataset is active.
    with pytest.raises(ValueError, match="Initialize joint data"):
        compiled.compute_joint_chi2(prediction=prediction)
    with pytest.raises(ValueError):
        compiled.get_joint_exponent()
