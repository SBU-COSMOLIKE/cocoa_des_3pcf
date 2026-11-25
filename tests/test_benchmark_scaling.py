"""Functional check of the scaling driver on two points and two threads.

tests/benchmark_scaling.py times model.logposterior of a numbered example
in one fresh process per thread count. This check runs it on two saved
points at one and two threads (about two and a half minutes) and asserts
the four things a scaling result needs:
  provider use        = CAMB, the bispectrum block and the likelihood each
                        recompute once at every point;
  recomputation       = the two points give different vectors;
  timing boundaries   = full, camb and other are recorded, and the
                        component intervals add up to the full time;
  output agreement    = both thread counts give the same vector.
It measures no performance; long runs belong on a quiet machine.
"""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
COCOA = PROJECT.parents[1]


def test_summary_rejects_nonfinite_predictions(tmp_path):
    """A NaN can evade a simple greater-than comparison at every thread count."""
    import runpy

    for threads in (1, 2):
        np.savez(tmp_path / f"threads_{threads}.npz", names=["camb"],
                 full=[1., 1.], spent=[[0.5], [0.5]], calls=[[1], [1]],
                 datavector=[[np.nan, 1.], [np.nan, 2.]])
    summarize = runpy.run_path(str(PROJECT / "tests/benchmark_scaling.py"))["summary"]
    with pytest.raises(ValueError, match="nonfinite data vectors"):
        summarize(SimpleNamespace(output=tmp_path, threads=[1, 2], tolerance=1.e-12,
                                  npoints=2))


def test_driver_records_boundaries_recomputation_and_agreement(tmp_path):
    """Two points, one and two threads: every check of the driver passes."""
    result = subprocess.run(
        args=[sys.executable, "-B", str(PROJECT / "tests/benchmark_scaling.py"),
              "--yaml", str(PROJECT / "EXAMPLE_EVALUATE3.yaml"),
              "--output", str(tmp_path), "--threads", "1", "2",
              "--npoints", "2"],
        cwd=COCOA, capture_output=True, text=True, check=False, timeout=900,
    )
    assert 0 == result.returncode, result.stdout + result.stderr

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert {"camb", "camb.transfers", "bihalofit"} <= set(summary["components"])
    assert set(summary["boundaries"]) == {"full", "camb", "other"}
    for threads in ("1", "2"):
        assert summary["max_relative_difference_to_first_thread_count"][threads] <= 1.0e-12

        with np.load(tmp_path / f"threads_{threads}.npz") as record:
            assert np.all(1 == record["calls"])
            assert not np.array_equal(record["datavector"][0], record["datavector"][1])
            np.testing.assert_allclose(np.sum(record["spent"], axis=1),
                                       record["full"], rtol=0.02)
