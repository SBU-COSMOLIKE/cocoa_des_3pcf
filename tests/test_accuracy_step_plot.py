"""Verify historical step identity and same-subset noise normalization."""

from copy import deepcopy
import json

import numpy as np
import pytest

from test_accuracy_full_identity import saved_accuracy_study


def saved_step_fixture(study):
    """Save artificial two-column diagnostics with a deliberately nonunit h2 score."""
    report, baseline, metadata = study.load()
    selected = baseline["parameter_names"].tolist()
    base_steps = []
    for name in selected:
        base_steps.append(metadata["configuration"]["fisher"]["steps"][name])
    base_steps = np.asarray(a=base_steps)
    cases = []
    choices = [
        ("h4", 0.25, 1.2, 1.05),
        ("h2", 0.5, 1.5, 1.0),
        ("h", 1.0, 1.8, 0.96),
        ("two_h", 2.0, 2.1, 0.98),
    ]
    for label, factor, noise, partial in choices:
        cases.append({
            "step_label": label,
            "absolute_steps": (base_steps*factor).tolist(),
            "diagonal_noise_inflation_ratio": noise,
            "selected_derivative_replacement_fom_ratio": partial,
            "selected_derivative_replacement_status": "resolved",
        })
    points = []
    for name in selected:
        for unit in (-1, 1, -16, 16):
            points.append({"name": name, "quarter_step_unit": unit})
    summary = {
        "format": "des3pcf-cosmology-step-study-v1",
        "configuration": str(study.configuration),
        "baseline": report["baseline_result"],
        "identity": {
            "baseline_sha256": report["baseline_result_sha256"],
            "configuration_sha256": report["configuration_sha256"],
            "prediction_identity": deepcopy(report["prediction_identity"]),
        },
        "selected_parameters": selected,
        "new_mean_count": len(points),
        "new_point_plan": points,
        "executed": True,
        "cases": list(reversed(cases)),
    }
    arrays = {
        "parameter_names": baseline["parameter_names"],
        "selected_columns": np.asarray(a=[0, 1], dtype=np.int64),
        "absolute_base_steps": base_steps,
        "row_labels": baseline["row_labels"],
    }
    directory = study.output.parent/"steps"
    directory.mkdir()
    write_step_fixture(directory=directory, summary=summary, arrays=arrays)
    return directory, summary, arrays


def write_step_fixture(directory, summary, arrays):
    """Save matching summary and NPZ metadata without a model or binary import."""
    text = json.dumps(summary)
    (directory/"summary.json").write_text(text, encoding="utf-8")
    np.savez(
        file=directory/"stencils.npz",
        metadata_json=np.asarray(a=text),
        **arrays,
    )


def test_step_plot_ratios_use_the_same_two_column_h2_denominator(tmp_path):
    """A current source edit cannot relabel or invalidate matching historical records."""
    study, source, _ = saved_accuracy_study(tmp_path=tmp_path)
    directory, _, _ = saved_step_fixture(study=study)
    source.write_bytes(source.read_bytes()+b" later physical source revision")
    loaded = study.load_step_study(directory=directory)
    np.testing.assert_allclose(
        actual=loaded["ratios"],
        desired=[[0.8, 1.0, 1.2, 1.4], [1.05, 1.0, 0.96, 0.98]],
    )
    assert loaded["labels"] == [
        r"$\Delta/4$",
        r"$\Delta/2$",
        r"$\Delta$",
        r"$2\Delta$",
    ]
    assert loaded["summary"]["new_mean_count"] == 8


@pytest.mark.parametrize("mismatch", ["summary", "baseline", "physical_identity"])
def test_step_plot_refuses_inconsistent_historical_records(tmp_path, mismatch):
    """Matching filenames or plot shapes cannot substitute for recorded provenance."""
    study, _, _ = saved_accuracy_study(tmp_path=tmp_path)
    directory, summary, arrays = saved_step_fixture(study=study)
    if mismatch == "summary":
        summary["cases"][0]["diagonal_noise_inflation_ratio"] = 9.0
        (directory/"summary.json").write_text(json.dumps(summary), encoding="utf-8")
        message = "summary differs"
    else:
        if mismatch == "baseline":
            summary["identity"]["baseline_sha256"] = "different archived baseline"
            message = "different baseline"
        else:
            physical = summary["identity"]["prediction_identity"]
            physical["input_files"]["source_curves"]["sha256"] = "different source"
            message = "files changed"
        write_step_fixture(directory=directory, summary=summary, arrays=arrays)
    with pytest.raises(ValueError, match=message):
        study.load_step_study(directory=directory)
