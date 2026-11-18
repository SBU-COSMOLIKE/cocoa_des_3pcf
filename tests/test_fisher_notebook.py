"""Bounded checks of the teaching wrapper and its public-input declarations."""

from pathlib import Path

import numpy as np
import pytest

import cosmolike_des_3pcf_fisher_wrappers as notebook


PROJECT = Path(__file__).resolve().parents[1]


def test_public_covariance_keeps_all_rows_and_the_cross_block():
    """No model run, scale selection, or independent-block substitution occurs."""
    source = PROJECT / "data" / "upstream_y3"
    if not source.is_dir():
        pytest.skip("Pinned public inputs have not been downloaded.")
    workflow = notebook.FisherWorkflow(
        configuration=PROJECT / "configs" / "des_y3_tatt_fisher.yaml",
    )
    rows, covariance, counts = workflow.public_covariance()
    assert counts == {"xip": 200, "xim": 200, "map3": 80}
    assert covariance.shape == (480, 480)
    assert rows.labels.size == 480
    assert np.any(covariance[:400, 400:] != 0.0)
    report = workflow.inspect()
    assert report["distinct_mean_evaluations"] == 109
    assert report["gaussian_prior_sigmas"]["dz3"] == 0.011
    verification = notebook.verify_public_inputs()
    assert len(verification["files"]) == 12
    assert all(record["status"] == "matches_pinned_blob"
               for record in verification["files"])


def test_controlled_ia_demonstration_reaches_both_sign_and_power_choices():
    """Tiny physical inputs show real C responses, without forecasting DES errors."""
    nonlinear = notebook.controlled_ia_response(power_mode=2, threads=1)
    linear = notebook.controlled_ia_response(power_mode=0, threads=1)
    repeated = notebook.controlled_ia_response(power_mode=2, threads=1)
    assert nonlinear["signals"].shape == (3, 2, 12)
    assert np.all(np.isfinite(nonlinear["signals"]))
    assert not np.array_equal(nonlinear["signals"][1], nonlinear["signals"][2])
    assert not np.array_equal(nonlinear["signals"][1:], linear["signals"][1:])
    np.testing.assert_array_equal(nonlinear["signals"][0], linear["signals"][0])
    np.testing.assert_array_equal(nonlinear["signals"], repeated["signals"])
