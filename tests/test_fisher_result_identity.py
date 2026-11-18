"""Reject changed files around saved means and accuracy-case calculations.

All predictions in these tests are artificial saved arrays. File hashing,
archive loading, and the public notebook reuse method are exercised without
constructing a physical model, importing a native extension, or running CAMB.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

import cosmolike_des_3pcf_fisher_wrappers as notebook
from des3pcf.fisher_accuracy import current_noise_case_identity
from des3pcf.provenance import prediction_identity, verify_prediction_identity
from test_prediction_provenance import artificial_prediction


@pytest.mark.parametrize(
    "changed_file",
    [
        "source.txt",
        "interface/third.so",
        "baseline.npz",
        "configuration.yaml",
        "accuracy_plan.yaml",
        "shared_fisher.py",
    ],
)
def test_candidate_end_check_rejects_changed_dependency(tmp_path, changed_file):
    """A completed candidate cannot be saved under an obsolete beginning hash."""
    project, arguments = artificial_prediction(tmp_path=tmp_path)
    extra = {
        "baseline.npz": b"baseline Fisher archive A",
        "configuration.yaml": b"candidate numerical settings A",
        "accuracy_plan.yaml": b"selected scan plan A",
        "shared_fisher.py": b"# shared stencil estimator A\n",
    }
    for name, contents in extra.items():
        (project / name).write_bytes(contents)
    arguments.update(
        baseline_path=project / "baseline.npz",
        dependency_paths=[project / "accuracy_plan.yaml", project / "shared_fisher.py"],
        selected_columns=[0, 2],
        steps=np.asarray([0.02, 0.03, 0.04]),
    )
    beginning = current_noise_case_identity(**arguments)
    unchanged = current_noise_case_identity(**arguments)
    verify_prediction_identity(
        recorded=beginning,
        current=unchanged,
        description="Unchanged accuracy case",
    )
    target = project / changed_file
    target.write_bytes(target.read_bytes() + b" replaced during calculation")
    completed = current_noise_case_identity(**arguments)
    with pytest.raises(ValueError, match="files changed"):
        verify_prediction_identity(
            recorded=beginning,
            current=completed,
            description="Accuracy case during evaluation",
        )


def artificial_saved_workflow(tmp_path, command, legacy=False, covariance=True):
    """Save labeled array fixtures through the ordinary result-file writer.

    The source and power-table files intentionally contain only bytes: the
    reuse path must hash them without trying to calculate a model. Current
    implementation files are fingerprinted by reading them, never imported
    as native libraries. The generated rows need no measurement FITS files.
    """
    project, _ = artificial_prediction(tmp_path=tmp_path)
    template = notebook.PROJECT / "configs/des_y3_nla_demo_fisher.yaml"
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["threads"] = 1
    config["source"]["path"] = str(project / "source.txt")
    config["physics"]["tables"] = str(project / "tables.npz")
    config["physics"]["cosmology_mode"] = "fixed_tables"
    # A fixed-table fixture can vary IA, but cannot request cosmological
    # derivatives. No physical evaluation is needed for archive checks.
    config["fisher"] = {"steps": {"a1": 0.01}, "priors": {}}
    config["mean"]["two_point"] = {"mode": "isolated_nla"}
    config["rows"] = {
        "mode": "forecast",
        "aperture_radii_arcmin": [14.0],
        "xi_edges_arcmin": {
            "lower": 10.0,
            "upper": 20.0,
            "count": 2,
        },
    }
    config["covariance"]["path"] = str(project / "covariance.npz")
    config["outputs"]["evaluate"] = str(project / "saved_mean.npz")
    config["outputs"]["fisher"] = str(project / "saved_fisher.npz")
    if not covariance:
        config.pop("covariance")
    path = project / "workflow.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    config, path, digest = notebook.driver.load_configuration(path=path, command=command)
    rows = notebook.driver.rows_for_run(config=config, path=path)
    matrix = np.eye(rows.values.size)
    np.savez(
        file=project / "covariance.npz",
        covariance=matrix,
        row_labels=rows.labels,
    )
    metadata = {
        "configuration_sha256": digest,
        "configuration": config,
    }
    if not legacy:
        metadata["prediction_identity"] = prediction_identity(
            config=config,
            configuration_path=path,
            project=notebook.PROJECT,
            three_point_path=(
                notebook.PROJECT / "interface/cosmolike_des_3pcf_interface.so"
            ),
        )
    arrays = {"mean": np.zeros(rows.values.size)}
    if command == "fisher":
        arrays["fiducial_covariance"] = matrix
    notebook.driver.save_result(
        path=Path(config["outputs"][command]),
        arrays=arrays,
        metadata=metadata,
        rows=rows,
        overwrite=False,
    )
    return notebook.FisherWorkflow(configuration=path), project / "source.txt"


@pytest.mark.parametrize("command", ["evaluate", "fisher"])
def test_execute_or_reuse_refuses_changed_source_but_history_is_readable(
    tmp_path, command,
):
    """Same YAML is insufficient for reuse; an explicitly historical read remains."""
    workflow, source = artificial_saved_workflow(tmp_path=tmp_path, command=command)
    first, _ = workflow.ensure_result(command=command)
    source.write_bytes(source.read_bytes() + b" revised source curve")
    archived, _ = workflow.load_result(command=command)
    np.testing.assert_array_equal(archived["mean"], first["mean"])
    with pytest.raises(ValueError, match="files changed"):
        workflow.ensure_result(command=command)


@pytest.mark.parametrize("command", ["evaluate", "fisher"])
def test_execute_or_reuse_refuses_missing_legacy_identity(tmp_path, command):
    """Old outputs remain inspectable but cannot stand in for a current calculation."""
    workflow, _ = artificial_saved_workflow(
        tmp_path=tmp_path,
        command=command,
        legacy=True,
    )
    archived, _ = workflow.load_result(command=command)
    assert archived["mean"].size > 0
    with pytest.raises(ValueError, match="no recorded physical dependency identity"):
        workflow.ensure_result(command=command)


def test_mean_identity_allows_omitted_covariance(tmp_path):
    """A mean-only saved result can be reused without an irrelevant covariance."""
    workflow, _ = artificial_saved_workflow(
        tmp_path=tmp_path,
        command="evaluate",
        covariance=False,
    )
    arrays, metadata = workflow.ensure_result(command="evaluate")
    assert arrays["mean"].size > 0
    assert "joint_covariance" not in metadata["prediction_identity"]["input_files"]
