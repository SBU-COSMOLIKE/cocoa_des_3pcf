"""Check the input checks of the joint likelihood at its real entry points.

Two inputs could produce an apparently successful likelihood although they
are invalid.

1. A covariance whose two triangles disagree. The Cholesky factorization
   of ci.init_joint_data reads one triangle, so it would silently use one
   of two different symmetric matrices. The symmetry test must work in
   correlation units, (C_ij - C_ji)/sqrt(C_ii C_jj): Map3 entries are many
   orders of magnitude below xi entries, and a tolerance in the matrix's
   own units accepts an asymmetric small block.

2. A compression trained on the same simulations as the finite-simulation
   covariance. The Percival likelihood treats the compression T as fixed;
   a T trained on the mocks is a random function of them. The production
   likelihood must refuse a compression file that records such training.
"""

import json
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]

# The two-entry asymmetric covariance of the red-team probe: its triangles
# hold 1.5 and 0.1, and its two symmetric completions give different chi2.
ASYMMETRIC = np.array([[2.0, 1.5], [0.1, 2.0]])
SYMMETRIC = np.array([[2.0, 0.8], [0.8, 2.0]])


def init(compiled, data, covariance, compression=None):
    """Install one small joint dataset with every row selected."""
    compiled.init_joint_data(
        data=np.ascontiguousarray(data, dtype=np.float64),
        covariance=np.ascontiguousarray(covariance, dtype=np.float64),
        indices=np.arange(len(data), dtype=np.int32),
        compression=(np.empty((0, 0)) if compression is None
                     else np.ascontiguousarray(compression, dtype=np.float64)),
    )


@pytest.mark.parametrize("scale", [1.0, 1.0e-20])
@pytest.mark.parametrize("transpose", [False, True])
def test_asymmetric_covariance_is_refused_at_every_scale(compiled, scale, transpose):
    """Both orientations are refused, whatever the overall units."""
    matrix = scale * (ASYMMETRIC.T if transpose else ASYMMETRIC)
    with pytest.raises(ValueError, match="not symmetric"):
        init(compiled, data=np.zeros(2), covariance=matrix)


def test_asymmetric_small_block_is_refused(compiled):
    """A small-amplitude block next to a unit block is judged on its own scale."""
    matrix = np.eye(3)
    matrix[1:, 1:] = 1.0e-20 * ASYMMETRIC
    with pytest.raises(ValueError, match="not symmetric"):
        init(compiled, data=np.zeros(3), covariance=matrix)


def test_nonpositive_diagonal_is_refused(compiled):
    """A zero variance leaves the correlation scale undefined."""
    matrix = np.diag([1.0, 0.0])
    with pytest.raises(ValueError, match="positive diagonal"):
        init(compiled, data=np.zeros(2), covariance=matrix)


def test_chi2_is_unchanged_by_the_units_of_each_row(compiled):
    """Rescaling each row with its covariance leaves chi2 unchanged.

    The rows get scales from 1 to 1e-20, as xi and Map3 differ in size;
    every scaled matrix is symmetric and must be accepted.
    """
    # Step 1: a random positive definite 4 x 4 covariance and a residual.
    rng = np.random.default_rng(seed=4)
    factor = rng.standard_normal((4, 4))
    covariance = factor @ factor.T + 4.0 * np.eye(4)
    covariance = 0.5 * (covariance + covariance.T)
    data = rng.standard_normal(4)
    prediction = rng.standard_normal(4)
    init(compiled, data=data, covariance=covariance)
    reference = compiled.compute_joint_chi2(prediction=prediction)

    # Step 2: the same quantities in other units, row by row.
    scales = np.array([1.0, 1.0e-6, 1.0e-12, 1.0e-20])
    scaled = scales[:, None] * covariance * scales[None, :]
    init(compiled, data=scales * data, covariance=scaled)
    value = compiled.compute_joint_chi2(prediction=scales * prediction)
    assert value == pytest.approx(reference, rel=1.0e-12)


def test_compressed_covariance_is_checked_and_used(compiled):
    """A compression with small rows works; a degenerate one is refused."""
    # Step 1: a symmetric covariance with one small row, and a compression.
    covariance = np.diag([1.0, 1.0, 1.0e-20])
    covariance[:2, :2] = SYMMETRIC
    data = np.array([0.3, -0.2, 2.0e-10])
    transform = np.array([[1.0, 0.5, 0.0], [0.0, 0.0, 1.0e10]])
    init(compiled, data=data, covariance=covariance, compression=transform)

    # Step 2: chi2 of the compressed vector, computed directly.
    prediction = np.zeros(3)
    residual = transform @ (data - prediction)
    expected = residual @ np.linalg.solve(transform @ covariance @ transform.T, residual)
    assert compiled.compute_joint_chi2(prediction=prediction) == pytest.approx(expected, rel=1.0e-12)

    # Step 3: a compression row of zeros leaves a zero variance.
    degenerate = np.array([[1.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="positive diagonal"):
        init(compiled, data=data, covariance=covariance, compression=degenerate)


def dataset_copy(directory, compression_file, mock_count):
    """Write case 5's dataset with absolute input paths into directory.

    Arguments:
      directory = pytest tmp_path.
      compression_file = absolute path or "" for no compression.
      mock_count = finite-simulation count; 0 selects the Gaussian likelihood.
    Returns:
      the file name of the written dataset.
    """
    absolute = ("data_file", "cov_file", "nz_source_file")
    lines = []
    for line in (PROJECT / "data/des_y3_joint5_sim.dataset").read_text().splitlines():
        key = line.split("=")[0].strip()
        if key in absolute:
            line = f"{key} = {PROJECT / 'data' / line.split('=')[1].strip()}"
        elif key == "compression_file":
            line = f"compression_file = {compression_file}"
        elif key == "mock_count":
            line = f"mock_count = {mock_count}"
        lines.append(line)
    name = f"case5_{mock_count}_{Path(compression_file).stem or 'none'}.dataset"
    (directory / name).write_text("\n".join(lines) + "\n")
    return name


def test_production_likelihood_refuses_mock_trained_compression(tmp_path):
    """The real likelihood class enforces the recorded training independence."""
    from cobaya.likelihoods.des_3pcf.joint_shear import joint_shear

    # Step 1: the selected row labels, from the uncompressed dataset (with
    # the Gaussian likelihood: 480 rows and 796 mocks admit no normalized
    # finite-simulation density without compression).
    plain = joint_shear(info={"path": str(tmp_path),
                              "data_file": dataset_copy(tmp_path, "", 0)})
    labels = np.array(plain.rows.labels, copy=True)
    assert labels.size == 480

    # Step 2: two archives that differ only in the recorded training.
    archives = {}
    for independent in (True, False):
        path = tmp_path / f"compression_{str(independent).lower()}.npz"
        np.savez(path, matrix=np.eye(480)[:80], input_labels=labels,
                 output_labels=np.array([f"mode_{i}" for i in range(80)]),
                 metadata=np.asarray(json.dumps(
                     {"covariance_training_independent": independent})))
        archives[independent] = path

    # Step 3: an independent training is accepted; one on the mocks is refused.
    accepted = joint_shear(info={
        "path": str(tmp_path), "data_file": dataset_copy(tmp_path, archives[True], 796)})
    assert accepted.rows.values.size == 480
    with pytest.raises(ValueError, match="trained on the simulations"):
        joint_shear(info={
            "path": str(tmp_path), "data_file": dataset_copy(tmp_path, archives[False], 796)})

    # Step 4: the Gaussian likelihood (mock_count = 0) makes no such assumption.
    joint_shear(info={
        "path": str(tmp_path), "data_file": dataset_copy(tmp_path, archives[False], 0)})
