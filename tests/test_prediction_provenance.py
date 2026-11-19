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
sys.path.insert(0, str(PROJECT / "python"))

from des3pcf.provenance import prediction_identity, verify_prediction_identity
from des3pcf.fisher_accuracy import accuracy_cases_for_request
from des3pcf.fisher_accuracy import load_retained_noise_case


def artificial_prediction(tmp_path):
    """Create distinct physical-input and implementation files without imports."""
    project = tmp_path / "project"
    files = {
        "source.txt": b"source density samples A",
        "tables.npz": b"physical table bytes A",
        "covariance.npz": b"fixed covariance bytes",
        "interface/third.so": b"three-point native library A",
        "interface/second.so": b"two-point native library A",
        "scripts/run_3pcf.py": b"# physical driver A\n",
        "python/des3pcf/mean.py": b"# mean implementation A\n",
    }
    for name, contents in files.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
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
        "python/des3pcf/mean.py",
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


def save_noise_case(path, identity, addition):
    """Write one small completed archive with its actual expected identity."""
    np.savez(
        file=path,
        a=np.asarray([addition]),
        sigma=np.asarray([0.1]),
        stencil_values=np.arange(10.0).reshape(1, 5, 2),
        evaluation_count=np.asarray(5),
        identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
    )


def test_partial_request_retains_and_validates_completed_cases(tmp_path):
    """Requesting a third case keeps earlier archives but never trusts a summary.

    These helpers are the actual runner's selection and archive-read path.
    No summary file is needed to recover the two completed cases, and a
    changed runtime identity prevents an unrequested case from being reused.
    """
    identities = {}
    for name in ("first", "second", "third"):
        identities[name] = {
            "baseline_result_sha256": "fixed baseline bytes",
            "configuration_sha256": name + " settings",
            "dependency_sha256": {"binary": "same native bytes"},
        }
    for name, addition in (("first", 0.2), ("second", 0.3)):
        save_noise_case(
            path=tmp_path / f"{name}_noise.npz",
            identity=identities[name],
            addition=addition,
        )
    candidates = [
        {"name": "first"},
        {"name": "second"},
        {"name": "third"},
        {"name": "unrequested_missing"},
    ]
    visited = accuracy_cases_for_request(
        candidates=candidates, only=["third"], output=tmp_path,
    )
    assert [item["name"] for item in visited] == ["first", "second", "third"]
    retained = []
    for name in ("first", "second"):
        saved = load_retained_noise_case(
            path=tmp_path / f"{name}_noise.npz", identity=identities[name],
        )
        retained.append(float(saved["a"][0]))
        assert saved["evaluation_count"] == 5
    assert retained == [0.2, 0.3]
    changed = dict(identities["first"])
    changed["dependency_sha256"] = {"binary": "changed native bytes"}
    with pytest.raises(ValueError, match="different inputs"):
        load_retained_noise_case(path=tmp_path / "first_noise.npz", identity=changed)
