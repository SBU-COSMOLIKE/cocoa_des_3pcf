"""Check the full h/4 study using declared polynomial vectors, without native calls.

Dyadic coordinates make the node identities exact. Cubic and quartic
observables test the fourth-order stencil, while a fifth-degree observable
retains a calculable finite-step error. File checks test whether reused
means retain their source identity; no physical engine is constructed.
"""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from cosmolike_notebook_utils.fisher import differentiate
from des3pcf.provenance import file_digest


HERE = Path(__file__).resolve().parent
STUDIES = HERE / "studies"
sys.path.insert(0, str(STUDIES))
SCRIPT = STUDIES / "full_smaller_step_study.py"
SPEC = importlib.util.spec_from_file_location(
    name="full_smaller_step_study", location=SCRIPT,
)
study = importlib.util.module_from_spec(spec=SPEC)
SPEC.loader.exec_module(module=study)


def polynomial_mean(parameters):
    """Return cubic, quartic and quintic toy observables at the supplied point."""
    matter = parameters["omega_m"]
    hubble = parameters["h"]
    offset = parameters["offset"]
    return np.asarray(a=[
        matter**3 + 2.0 * hubble + offset,
        matter**4 + hubble**3 + 3.0 * offset,
        matter**5 + hubble + offset**3,
    ], dtype=np.float64)


def reference_inputs():
    """Use positive, negative and zero fiducials with absolute dyadic steps."""
    reference = {
        "omega_m": 0.25,
        "h": -0.75,
        "offset": -0.0,
    }
    steps = {
        "omega_m": 0.125,
        "h": 0.0625,
        "offset": 0.125,
    }
    bounds = {
        "omega_m": (-1.0, 1.5),
        "h": (-1.5, 0.0),
        "offset": (-1.0, 1.0),
    }
    return (
        reference,
        steps,
        bounds,
    )


def direct_stencil(reference, name, step):
    """Evaluate five explicit toy points, preserving the signed-zero center."""
    values = []
    for multiplier in (
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
    ):
        point = dict(reference)
        if multiplier != 0.0:
            point[name] = float(reference[name]) + multiplier * step
        values.append(polynomial_mean(parameters=point))
    return np.asarray(a=values, dtype=np.float64)


def baseline_arrays():
    """Produce ordinary h/h2 derivative arrays and retained toy stencil samples."""
    reference, steps, bounds = reference_inputs()
    names, mean, derivative, derivative_half = differentiate(
        predict=polynomial_mean, parameters=reference, steps=steps, bounds=bounds,
    )
    stencils = []
    half_stencils = []
    for name in names:
        stencils.append(direct_stencil(reference=reference, name=name, step=steps[name]))
        half_stencils.append(direct_stencil(
            reference=reference, name=name, step=steps[name] / 2.0,
        ))
    return {
        "parameter_names": np.asarray(a=names),
        "mean": mean,
        "derivatives_h": derivative,
        "derivatives_h2": derivative_half,
        "stencil_values_h": np.asarray(a=stencils),
        "stencil_values_h2": np.asarray(a=half_stencils),
        "row_labels": np.asarray(a=[
            "toy_cubic",
            "toy_quartic",
            "toy_quintic",
        ]),
    }


def complete_bank(arrays):
    """Add only the six new toy means to the already checked archived bank."""
    reference, steps, bounds = reference_inputs()
    bank = study.baseline_bank(arrays=arrays, reference=reference, steps=steps)
    plan = study.quarter_point_plan(reference=reference, steps=steps, bounds=bounds)
    for record in plan:
        values = polynomial_mean(parameters=record["parameters"])
        study.recorded_vector(
            bank=bank, parameters=record["parameters"], values=values, nrow=3,
        )
    return bank


def test_half_control_and_quarter_stencil_match_independent_toy():
    """Reconstruction preserves old bits and yields the expected smaller-step error."""
    arrays = baseline_arrays()
    reference, steps, bounds = reference_inputs()
    bank = complete_bank(arrays=arrays)
    half, quarter, stencils, coordinates = study.reconstruct_derivatives(
        bank=bank, reference=reference, steps=steps, bounds=bounds, arrays=arrays,
    )
    np.testing.assert_array_equal(
        x=half.view(np.uint64), y=arrays["derivatives_h2"].view(np.uint64),
    )
    for column, name in enumerate(steps):
        expected = direct_stencil(reference=reference, name=name, step=steps[name] / 4)
        np.testing.assert_array_equal(x=stencils[column], y=expected)
        assert coordinates[column, 2].hex() == float(reference[name]).hex()
    exact_first_two = np.asarray(a=[3 * reference["omega_m"]**2,
                                       4 * reference["omega_m"]**3])
    np.testing.assert_array_equal(x=quarter[:2, 0], y=exact_first_two)
    exact_quintic = 5 * reference["omega_m"]**4
    old_error = half[2, 0] - exact_quintic
    new_error = quarter[2, 0] - exact_quintic
    assert old_error / new_error == 16.0


def test_plan_preserves_absolute_steps_and_signed_zero():
    """Negative and zero fiducials use the same absolute-step convention."""
    reference, steps, bounds = reference_inputs()
    plan = study.quarter_point_plan(reference=reference, steps=steps, bounds=bounds)
    assert len(plan) == 2 * len(steps)
    for record in plan:
        name = record["parameter"]
        unit = record["quarter_step_unit"]
        expected = reference[name] + unit * (steps[name] / 4)
        assert record["parameters"][name].hex() == expected.hex()
        for fixed in reference:
            if fixed != name:
                assert record["parameters"][fixed].hex() == reference[fixed].hex()
        assert record["parameter_key"] != study.point_key(parameters=reference)


@pytest.mark.parametrize("failure", [
    "collapsed",
    "bounds",
    "underflow",
])
def test_coordinate_refusal_before_any_predictor(failure):
    """A valid half-step does not authorize a collapsed or out-of-bounds quarter-step."""
    reference, steps, bounds = reference_inputs()
    # NumPy ufunc inputs are positional-only: value, then target direction.
    if failure == "collapsed":
        reference["omega_m"] = 1.0
        steps["omega_m"] = 2 * np.spacing(1.0)
    elif failure == "bounds":
        bounds["omega_m"] = (0.2, 0.3)
    else:
        steps["omega_m"] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError):
        study.quarter_point_plan(reference=reference, steps=steps, bounds=bounds)


@pytest.mark.parametrize("failure", [
    "shared_node",
    "missing_point",
    "derivative",
])
def test_changed_or_incomplete_archived_values_are_refused(failure):
    """The control derivative catches bad assembly without a physical fallback."""
    arrays = baseline_arrays()
    reference, steps, bounds = reference_inputs()
    if failure == "shared_node":
        value = arrays["stencil_values_h2"][0, 0, 0]
        arrays["stencil_values_h2"][0, 0, 0] = np.nextafter(value, np.inf)
        with pytest.raises(ValueError, match="differs bitwise"):
            study.baseline_bank(arrays=arrays, reference=reference, steps=steps)
        return
    bank = complete_bank(arrays=arrays)
    if failure == "missing_point":
        plan = study.quarter_point_plan(reference=reference, steps=steps, bounds=bounds)
        del bank[plan[0]["parameter_key"]]
    else:
        value = arrays["derivatives_h2"][0, 0]
        arrays["derivatives_h2"][0, 0] = np.nextafter(value, np.inf)
    with pytest.raises(ValueError):
        study.reconstruct_derivatives(
            bank=bank, reference=reference, steps=steps, bounds=bounds, arrays=arrays,
        )


def write_step_fixture(directory, failure=None):
    """Write a small declared toy counterpart of the existing two-column archive."""
    from cosmology_step_study import new_point_plan

    arrays = baseline_arrays()
    reference, steps, bounds = reference_inputs()
    producer = STUDIES / "cosmology_step_study.py"
    identity = {
        "baseline_sha256": "controlled-baseline",
        "configuration_sha256": "controlled-configuration",
        "prediction_identity": {"physical_inputs": "controlled-toy"},
        "helper_sha256": {str(producer): file_digest(path=producer)},
    }
    recorded_identity = dict(identity)
    recorded_identity["study_sha256"] = identity["helper_sha256"][str(producer)]
    labels = [
        "h4",
        "h2",
        "h",
    ]
    factors = {
        "h4": 0.25,
        "h2": 0.5,
        "h": 1.0,
    }
    cases = []
    saved = {
        "parameter_names": arrays["parameter_names"],
        "row_labels": arrays["row_labels"],
        "selected_columns": np.asarray(a=[0, 1]),
        "absolute_base_steps": np.asarray(a=[steps["omega_m"], steps["h"]]),
    }
    for label in labels:
        cases.append({"step_label": label})
        values = []
        for name in ("omega_m", "h"):
            values.append(direct_stencil(
                reference=reference, name=name, step=steps[name] * factors[label],
            ))
        saved["stencil_values_" + label] = np.asarray(a=values)
        # The reader checks these diagnostic arrays' finite shape only. The
        # means above, independently evaluated from the toy, are reused.
        saved["selected_derivatives_" + label] = arrays["derivatives_h2"][:, :2]
    summary = {
        "format": "des3pcf-cosmology-step-study-v1",
        "executed": True,
        "identity": recorded_identity,
        "selected_parameters": ["omega_m", "h"],
        "cases": cases,
        "new_point_plan": new_point_plan(
            parameters=reference, steps=steps, bounds=bounds, larger_step=False,
        ),
    }
    if failure == "missing_h4":
        summary["cases"] = [{"step_label": "h"}, {"step_label": "h2"}]
    elif failure == "producer":
        recorded_identity["study_sha256"] = "different-producer"
    elif failure == "baseline":
        recorded_identity["baseline_sha256"] = "different-baseline"
    elif failure == "coordinate":
        summary["new_point_plan"][0]["point"]["omega_m"] += 0.001
    elif failure == "shared_node":
        saved["stencil_values_h4"][0, 0, 0] += 0.001
    metadata = study.json_text(record=summary)
    (directory / "summary.json").write_text(data=metadata, encoding="utf-8")
    saved["metadata_json"] = np.asarray(a=metadata)
    np.savez_compressed(file=directory / "stencils.npz", **saved)
    config = {"parameters": reference, "ia": {}}
    return (
        identity,
        arrays,
        config,
        reference,
        steps,
    )


def test_valid_two_column_reuse_leaves_only_other_parameter_points(tmp_path):
    """Four source-verified means remove four evaluations from the six-point toy."""
    identity, arrays, config, reference, steps = write_step_fixture(directory=tmp_path)
    reused, _ = study.load_step_reuse(
        directory=tmp_path, identity=identity, arrays=arrays, config=config,
        reference=reference, steps=steps,
    )
    _, _, bounds = reference_inputs()
    plan = study.quarter_point_plan(reference=reference, steps=steps, bounds=bounds)
    missing = 0
    for record in plan:
        if record["parameter_key"] not in reused:
            missing += 1
    assert len(reused) == 4
    assert missing == 2


@pytest.mark.parametrize("failure", [
    "missing_h4",
    "producer",
    "baseline",
    "coordinate",
    "shared_node",
])
def test_two_column_reuse_refuses_inconsistent_provenance_or_nodes(tmp_path, failure):
    """A matching filename never substitutes for the expected coordinates and producer."""
    identity, arrays, config, reference, steps = write_step_fixture(
        directory=tmp_path, failure=failure,
    )
    with pytest.raises(ValueError):
        study.load_step_reuse(
            directory=tmp_path, identity=identity, arrays=arrays, config=config,
            reference=reference, steps=steps,
        )


@pytest.mark.parametrize("failure", [
    None,
    "known_mean",
    "receipt",
    "absent_receipt",
])
def test_existing_point_requires_receipt_and_source_bits(tmp_path, failure):
    """A valid self-consistent receipt cannot overrule a known reused mean."""
    reference, steps, bounds = reference_inputs()
    record = study.quarter_point_plan(reference=reference, steps=steps, bounds=bounds)[0]
    record["origin"] = "two_column_step_study"
    expected = {"manifest_sha256": "controlled-manifest", "point": record}
    mean = polynomial_mean(parameters=record["parameters"])
    reused = {record["parameter_key"]: mean.copy()}
    if failure == "known_mean":
        mean[0] = np.nextafter(mean[0], np.inf)
    path = tmp_path / "point.npz"
    receipt_path = tmp_path / "point.json"
    np.savez_compressed(
        file=path, mean=mean, metadata_json=study.json_text(record=expected),
    )
    receipt = {"point": record, "archive_sha256": file_digest(path=path)}
    if failure == "receipt":
        receipt["archive_sha256"] = "changed-hash"
    if failure != "absent_receipt":
        study.checked_text(
            path=receipt_path, content=study.json_text(record=receipt), write=True,
        )
    if failure is None:
        actual = study.verified_point(
            path=path, receipt_path=receipt_path, expected=expected,
            nrow=3, reused=reused,
        )
        np.testing.assert_array_equal(x=actual.view(np.uint64), y=mean.view(np.uint64))
    else:
        exceptions = (
            ValueError,
            FileExistsError,
            FileNotFoundError,
        )
        with pytest.raises(exceptions):
            study.verified_point(
                path=path, receipt_path=receipt_path, expected=expected,
                nrow=3, reused=reused,
            )
