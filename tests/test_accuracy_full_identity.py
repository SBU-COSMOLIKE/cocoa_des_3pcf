"""Keep historical accuracy comparisons separate from executable reuse.

Only small labeled archives and file hashes are used. The fixtures reuse the
real saved-result writer and notebook readers; they never evaluate a model.
Changed source bytes exercise the refusal before the child Fisher launch.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import cosmolike_des_3pcf_accuracy_wrappers as accuracy_notebook
import cosmolike_des_3pcf_fisher_wrappers as fisher_notebook
from des3pcf.provenance import file_digest
from test_fisher_result_identity import artificial_saved_workflow


def saved_accuracy_study(tmp_path, cosmological=False):
    """Create one baseline, one selected-column archive and one full candidate.

    The candidate has a doubled two-parameter Fisher matrix, so its exact
    FoM ratio is two. All physical inputs and implementation identities are
    initially shared; only its numerical resolution and output path differ.
    """
    workflow, source = artificial_saved_workflow(tmp_path=tmp_path, command="fisher")
    driver = fisher_notebook.driver
    config, path, _ = driver.load_configuration(
        path=workflow.configuration, command="fisher",
    )
    rows = driver.rows_for_run(config=config, path=path)
    arrays, metadata = workflow.load_result(command="fisher")
    # This fixed-table toy can vary its two IA parameters. Its parameter
    # labels, configured spacings and archived arrays must describe the
    # same experiment even though no physical mean is evaluated here.
    config["fisher"]["steps"] = {"a1": 0.02, "eta1": 0.04}
    pair = ["a1", "eta1"]
    if cosmological:
        # These are historical array records, not requests for a CAMB
        # evaluation. Four columns separate the varied subset from its FoM.
        config["physics"]["cosmology_mode"] = "camb"
        config["fisher"]["steps"] = {
            "omega_m": 0.003,
            "h": 0.005,
            "w0": 0.02,
            "wa": 0.04,
        }
        pair = ["w0", "wa"]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _, _, baseline_configuration_digest = driver.load_configuration(
        path=path, command="fisher",
    )
    metadata["configuration"] = config
    metadata["configuration_sha256"] = baseline_configuration_digest
    arrays.pop("observed_values")
    for key in ("row_labels", "row_kind", "row_bins", "row_angles_radians"):
        arrays.pop(key)
    names = list(config["fisher"]["steps"])
    nparameter = len(names)
    arrays["parameter_names"] = np.asarray(names)
    arrays["fisher_h"] = np.eye(nparameter)
    arrays["fisher_h2"] = np.eye(nparameter)
    # A simple linear response gives unambiguous saved node identities.
    # The artificial Fisher/noise scores below test plotting denominators,
    # not a scientific reconstruction from this toy's derivatives.
    for label, divisor in (("h", 1.0), ("h2", 2.0)):
        values = np.empty((nparameter, 5, rows.values.size))
        for column, name in enumerate(names):
            step = config["fisher"]["steps"][name]/divisor
            for index, offset in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
                values[column, index] = (column + 1)*offset*step
        arrays[f"stencil_values_{label}"] = values
    baseline = Path(config["outputs"]["fisher"])
    driver.save_result(
        path=baseline,
        arrays=arrays,
        metadata=metadata,
        rows=rows,
        overwrite=True,
    )
    output = path.parent / "accuracy"
    output.mkdir()
    candidate = deepcopy(config)
    candidate["accuracy"]["chi_boost"] = 2.0
    candidate["outputs"]["fisher"] = str(output / "chi_2_full_fisher.npz")
    case_path = output / "chi_2.yaml"
    case_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    _, _, case_digest = driver.load_configuration(path=case_path, command="fisher")
    case_metadata = deepcopy(metadata)
    case_metadata["configuration_sha256"] = case_digest
    case_metadata["configuration"] = candidate
    case_arrays = dict(arrays)
    case_arrays["fisher_h"] = 2.0 * np.eye(nparameter)
    case_arrays["fisher_h2"] = 2.0 * np.eye(nparameter)
    driver.save_result(
        path=Path(candidate["outputs"]["fisher"]),
        arrays=case_arrays,
        metadata=case_metadata,
        rows=rows,
        overwrite=False,
    )
    identity = {
        "baseline_result_sha256": file_digest(path=baseline),
        "configuration_sha256": case_digest,
        "prediction_identity": metadata["prediction_identity"],
        "dependency_sha256": {},
        "selected_columns": [0, 1],
        "absolute_steps": [config["fisher"]["steps"][name]/2 for name in names],
    }
    np.savez(
        file=output / "chi_2_noise.npz",
        identity_json=np.asarray(json.dumps(identity)),
        a=np.zeros(nparameter),
    )
    report = {
        "format": "des3pcf-fisher-accuracy-result-v1",
        "configuration_sha256": metadata["configuration_sha256"],
        "baseline_result": str(baseline),
        "baseline_result_sha256": file_digest(path=baseline),
        "prediction_identity": metadata["prediction_identity"],
        "dependency_sha256": {},
        "parameter_names": names,
        "selected_columns": [0, 1],
        "absolute_steps": identity["absolute_steps"],
        "pair": pair,
        "baseline_fom": 1.0,
        "cases": [
            {
                "name": "chi_2",
                "configuration": str(case_path),
                "a": np.zeros(nparameter).tolist(),
            },
        ],
    }
    (output / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    study = accuracy_notebook.AccuracyStudy(
        configuration=path,
        output=output,
        plan=fisher_notebook.PROJECT / "configs/des_y3_fisher_accuracy.yaml",
    )
    return study, source, Path(candidate["outputs"]["fisher"])


def rewrite_metadata(path, change):
    """Replace only recorded metadata while retaining every numerical array."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if change == "missing":
        metadata.pop("prediction_identity")
    else:
        metadata["prediction_identity"]["input_files"]["source_curves"]["sha256"] = (
            "different recorded physical source"
        )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(file=path, **arrays)


def test_historical_full_comparison_survives_current_source_change(tmp_path):
    """Historical records can agree even when today's input file has changed."""
    study, source, _ = saved_accuracy_study(tmp_path=tmp_path)
    before = study.load_full_checks()
    np.testing.assert_allclose(before["ratios"], [[1.0, 2.0], [1.0, 2.0]])
    source.write_bytes(source.read_bytes() + b" a later source reconstruction")
    after = study.load_full_checks()
    np.testing.assert_array_equal(after["ratios"], before["ratios"])


@pytest.mark.parametrize("change", ["missing", "different"])
def test_full_comparison_refuses_different_archived_prediction(tmp_path, change):
    """Identical rows and covariance cannot certify identical physical code/input."""
    study, _, full_path = saved_accuracy_study(tmp_path=tmp_path)
    rewrite_metadata(path=full_path, change=change)
    with pytest.raises(ValueError, match="identity|files changed"):
        study.load_full_checks()


def test_full_execution_refuses_source_change_before_child_launch(tmp_path):
    """Executable reuse must refuse a changed file before starting the driver."""
    study, source, full_path = saved_accuracy_study(tmp_path=tmp_path)
    # Removing the full output makes this an executable request, rather than
    # a refusal caused merely by an existing destination.
    full_path.unlink()
    source.write_bytes(source.read_bytes() + b" revised after selected scan")
    with pytest.raises(ValueError, match="Current full-check inputs.*files changed"):
        study.run_full_check(case="chi_2")
    assert not full_path.exists()
