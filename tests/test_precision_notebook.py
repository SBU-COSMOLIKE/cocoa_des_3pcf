"""Check the covariance teaching example and external simulation identities."""

import numpy as np
import pytest

import cosmolike_des_3pcf_notebook_wrappers as notebook


def test_controlled_joint_sample_is_singular_without_requiring_its_inverse():
    """Truncation and finite-sample error have separate saved diagnostics."""
    report = notebook.controlled_precision_study(
        ndata=12, training_count=8, validation_count=100, seed=21,
    )
    assert report["sample_rank_in_model_units"] <= 7
    assert report["sample_inverse_attempted"] is False
    assert report["training_stream"] != report["validation_stream"]
    assert report["true_residual_spectral_radius"] < 1.0
    assert np.any(report["system"]["truth"][:6, 6:] != 0.0)
    assert report["diagnostics"]["exact_second_order"]["positive"]
    error = report["diagnostics"]["truth"]["precision_error_eigenvalues"]
    np.testing.assert_allclose(error, 0.0, atol=3e-14)
    exact = report["diagnostics"]["truth"]["exact_variance_over_quoted"]
    np.testing.assert_allclose(exact, 1.0, rtol=3e-14)


def test_external_simulation_rows_and_parent_identity(tmp_path):
    """Patch count cannot silently replace the independent sky count."""
    system = notebook.controlled_joint_covariance(ndata=8)
    path = tmp_path / "joint.npz"
    vectors = np.arange(40, dtype=np.float64).reshape(5, 8)
    labels = system["row_labels"]
    kinds = system["row_kind"]
    parents = np.arange(5)
    np.savez(
        file=path, vectors=vectors, row_labels=labels, row_kind=kinds,
        parent_ids=parents, estimator=np.array("finite_sas"),
    )
    record = notebook.load_joint_simulations(
        path=path, expected_labels=labels, expected_kind=kinds,
        expected_estimator="finite_sas",
    )
    np.testing.assert_array_equal(record["vectors"], vectors)
    assert record["independent_count"] == 5
    with pytest.raises(ValueError, match="overlap"):
        notebook.load_joint_simulations(
            path=path, expected_labels=labels, expected_kind=kinds,
            expected_estimator="finite_sas", excluded_parent_ids=[2],
        )
    with pytest.raises(ValueError, match="rows do not match"):
        notebook.load_joint_simulations(
            path=path, expected_labels=labels[::-1], expected_kind=kinds,
            expected_estimator="finite_sas",
        )
    parents[1] = parents[0]
    np.savez(
        file=path, vectors=vectors, row_labels=labels, row_kind=kinds,
        parent_ids=parents, estimator=np.array("finite_sas"),
    )
    with pytest.raises(ValueError, match="patches are not independent"):
        notebook.load_joint_simulations(
            path=path, expected_labels=labels, expected_kind=kinds,
            expected_estimator="finite_sas",
        )
