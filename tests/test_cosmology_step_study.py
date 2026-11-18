"""Check exact stencil reuse and new-evaluation counts without a physical model."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "studies/cosmology_step_study.py"
SPEC = importlib.util.spec_from_file_location("cosmology_step_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def polynomial_mean(parameters):
    """Two coupled cubic observables have exact derivatives at dyadic points."""
    matter = parameters["omega_m"]
    hubble = parameters["h"]
    fixed = parameters["fixed"]
    return np.asarray([
        fixed + matter**3 + 2.0*hubble,
        matter - 3.0*hubble**3 + matter*hubble,
    ])


def controlled_reference():
    """Return exactly representable values, absolute steps and explicit bounds."""
    return (
        {"omega_m": 0.25, "h": 0.75, "fixed": 1.5},
        {"omega_m": 0.03125, "h": 0.0625},
        {"omega_m": (0.0, 1.5), "h": (0.0, 1.5)},
    )


def direct_stencil(reference, name, spacing):
    """Independently evaluate the five physical coordinates for a small toy."""
    result = []
    for multiplier in (-2.0, -1.0, 0.0, 1.0, 2.0):
        point = dict(reference)
        point[name] += multiplier*spacing
        result.append(polynomial_mean(parameters=point))
    return np.asarray(result)


@pytest.mark.parametrize("larger_step,expected_count", [(False, 4), (True, 8)])
def test_new_evaluations_and_saved_node_assembly(larger_step, expected_count):
    """Only four/eight new calls reproduce independent full toy stencils."""
    reference, steps, bounds = controlled_reference()
    plan = study.new_point_plan(
        parameters=reference, steps=steps, bounds=bounds, larger_step=larger_step,
    )
    calls = []

    def record_prediction(parameters):
        """Record ordinary predictor calls; do not replace any imported API."""
        calls.append(dict(parameters))
        return polynomial_mean(parameters=parameters)

    new_values = study.evaluate_missing(plan=plan, predict=record_prediction, ndata=2)
    assert len(calls) == len(plan) == expected_count
    assert all(point["fixed"] == reference["fixed"] for point in calls)
    assert all(point != reference for point in calls)
    labels = ["h4", "h2", "h"]
    if larger_step:
        labels.append("two_h")
    for name in study.COLUMNS:
        old = direct_stencil(reference=reference, name=name, spacing=steps[name])
        half = direct_stencil(reference=reference, name=name, spacing=steps[name]/2)
        saved = study.reuse_nodes(stencil_h=old, stencil_h2=half)
        for label in labels:
            spacing = steps[name]*study.STEP_FACTORS[label]
            actual = study.assemble_stencil(
                nodes=saved, new_values=new_values[name], label=label,
            )
            expected = direct_stencil(reference=reference, name=name, spacing=spacing)
            np.testing.assert_array_equal(actual, expected)
            derivative = (actual[0]-8*actual[1]+8*actual[3]-actual[4])/(12*spacing)
            if name == "omega_m":
                exact = [3*reference[name]**2, 1+reference["h"]]
            else:
                exact = [2.0, -9*reference[name]**2+reference["omega_m"]]
            np.testing.assert_array_equal(derivative, exact)
    assert len(calls) == expected_count


def test_conflicting_saved_shared_coordinate_is_refused():
    """Repeated physical coordinates must contain identical recorded vectors."""
    reference, steps, _ = controlled_reference()
    old = direct_stencil(reference=reference, name="h", spacing=steps["h"])
    half = direct_stencil(reference=reference, name="h", spacing=steps["h"]/2)
    half[0, 0] = np.nextafter(half[0, 0], np.inf)
    with pytest.raises(ValueError, match="differs bitwise"):
        study.reuse_nodes(stencil_h=old, stencil_h2=half)


def test_larger_stencil_bounds_fail_before_prediction():
    """An allowed small-step request does not authorize an out-of-bounds outer one."""
    reference, steps, bounds = controlled_reference()
    bounds["omega_m"] = (0.15, 0.35)
    small = study.new_point_plan(parameters=reference, steps=steps, bounds=bounds)
    assert len(small) == 4
    with pytest.raises(ValueError, match="leaves its bounds"):
        study.new_point_plan(
            parameters=reference, steps=steps, bounds=bounds, larger_step=True,
        )
