"""Check exact-file provenance and retained partial accuracy studies.

The files below contain artificial bytes; no C library is loaded and no
cosmology is evaluated. The tests exercise real hashing, archive reading,
and identity refusal with independently changed source and binary files.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))

from des3pcf.provenance import prediction_identity, verify_prediction_identity


def artificial_prediction(tmp_path):
    """Create distinct physical-input and implementation files without imports."""
    project = tmp_path / "Cocoa" / "projects" / "project"
    files = {
        "source.txt": b"source density samples A",
        "tables.npz": b"physical table bytes A",
        "covariance.npz": b"fixed covariance bytes",
        "interface/third.so": b"three-point native library A",
        "interface/second.so": b"two-point native library A",
        "scripts/run_3pcf.py": b"# physical driver A\n",
        "scripts/des3pcf/mean.py": b"# mean implementation A\n",
    }
    for name, contents in files.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    core = project.parents[1] / "external_modules/code/cosmolike_core"
    for relative in (
        "cosmolike_notebook_utils/fisher.py",
        "cosmolike_notebook_utils/covariance/precision.py",
        "theory/bihalofit/model.py",
        "theory/bacco_bispectrum/response.py",
    ):
        filename = core / relative
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text("# Artificial shared implementation A\n")
    config = {
        "physics": {"tables": "tables.npz", "cosmology_mode": "fixed"},
        "source": {"path": "source.txt"},
        "covariance": {"path": "covariance.npz"},
        "rows": {"mode": "generated"},
    }
    arguments = {
        "config": config,
        "configuration_path": project / "configuration.yaml",
        "project": project,
        "three_point_path": project / "interface/third.so",
        "two_point_path": project / "interface/second.so",
    }
    return project, arguments


@pytest.mark.parametrize(
    "changed_file",
    [
        "source.txt",
        "tables.npz",
        "interface/third.so",
        "interface/second.so",
        "scripts/des3pcf/mean.py",
    ],
)
def test_same_path_input_or_implementation_change_is_detected(tmp_path, changed_file):
    """Start/end equality fails when exact bytes change, without renaming files."""
    project, arguments = artificial_prediction(tmp_path=tmp_path)
    beginning = prediction_identity(**arguments)
    unchanged = prediction_identity(**arguments)
    verify_prediction_identity(
        recorded=beginning,
        current=unchanged,
        description="Unchanged test calculation",
    )
    target = project / changed_file
    target.write_bytes(target.read_bytes() + b" changed during evaluation")
    completed = prediction_identity(**arguments)
    with pytest.raises(ValueError, match="files changed"):
        verify_prediction_identity(
            recorded=beginning,
            current=completed,
            description="Test calculation during evaluation",
        )


def test_legacy_baseline_cannot_inherit_current_files(tmp_path):
    """A current fingerprint is not evidence about how an old Fisher was made."""
    _, arguments = artificial_prediction(tmp_path=tmp_path)
    current = prediction_identity(**arguments)
    with pytest.raises(ValueError, match="no recorded physical dependency identity"):
        verify_prediction_identity(
            recorded=None,
            current=current,
            description="Legacy baseline Fisher",
        )


def test_covariance_construction_hashes_inputs_without_reading_output(tmp_path):
    """An absent output matrix must not prevent checking its physical inputs.

    The covariance command consumes power and source tables and creates a
    matrix. Deleting that destination distinguishes its role from a Fisher
    calculation, which must read the existing matrix before differentiating.
    """
    project, arguments = artificial_prediction(tmp_path=tmp_path)
    (project / "covariance.npz").unlink()
    arguments["covariance_output"] = True
    beginning = prediction_identity(**arguments)
    assert "joint_covariance" not in beginning["input_files"]
    assert "physical_tables" in beginning["input_files"]
    assert "source_curves" in beginning["input_files"]
    target = project / "tables.npz"
    target.write_bytes(b"changed input power table")
    completed = prediction_identity(**arguments)
    with pytest.raises(ValueError, match="files changed"):
        verify_prediction_identity(
            recorded=beginning,
            current=completed,
            description="Covariance construction",
        )
