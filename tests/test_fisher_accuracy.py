"""Check the thin adapters and the stated limits of the shared diagnostic.

These small polynomial and matrix examples require no physical C extension,
CAMB run or forecast. The tests import the actual shared core functions;
they never replace them. Known polynomial cancellations and an independent
Schur complement supply answers beyond a comparison of two adapter calls.
"""

import numpy as np
import pytest

from cosmolike_notebook_utils import fisher as shared_fisher
from des3pcf.fisher_accuracy import (
    get_fom_noise_weights,
    get_stencil_noise,
    make_offset_predictor,
    predict_fom_noise,
    retained_stencil_noise,
)
from des3pcf.forecast import ExactMeanCache, differentiate, fisher_matrix


def saved_values(predict, reference, names, steps):
    """Collect controlled means in the project's retained archive ordering."""
    columns = []
    for column, name in enumerate(names):
        vectors = []
        for multiple in (-2.0, -1.0, 0.0, 1.0, 2.0):
            point = dict(reference)
            if multiple != 0.0:
                point[name] = float(reference[name]) + multiple * steps[column]
            vectors.append(np.array(predict(point), dtype=np.float64, copy=True))
        columns.append(vectors)
    return np.asarray(columns, dtype=np.float64)


def test_estimators_are_the_shared_core_functions():
    """Keep one implementation of the fourth difference and FoM mathematics."""
    assert get_stencil_noise is shared_fisher.get_stencil_noise
    assert get_fom_noise_weights is shared_fisher.get_fom_noise_weights
    assert predict_fom_noise is shared_fisher.predict_fom_noise


def test_cubic_saved_stencils_need_no_new_predictions():
    """A fourth difference annihilates a cubic, including coupled data rows."""
    calls = []

    def mean(parameters):
        """Record controlled evaluations before the archive-only score."""
        calls.append(dict(parameters))
        x = parameters["x"]
        y = parameters["y"]
        return np.array([3.0*x + 2.0*y + x**3 + y**2,
                         -x + 4.0*y + 2.0*x**2 - y**3,
                         5.0*x - 2.0*y])

    steps = np.array([0.125, 0.25])
    values = saved_values(predict=mean, reference={"x": 0.0, "y": 0.0},
                          names=["x", "y"], steps=steps)
    retained = values.copy()
    evaluations = len(calls)
    values.setflags(write=False)
    score = retained_stencil_noise(stencil_values=values, steps=steps, invcov=np.eye(3))
    assert len(calls) == evaluations
    np.testing.assert_array_equal(x=values, y=retained)
    np.testing.assert_array_equal(x=score["sigma"], y=np.zeros(2))
    np.testing.assert_array_equal(x=score["a"], y=np.zeros(2))
    np.testing.assert_array_equal(x=score["eps"], y=steps)
    np.testing.assert_allclose(score["ddv"], [[3.0, 2.0], [-1.0, 4.0], [5.0, -2.0]],
                               rtol=2.0e-15, atol=2.0e-15)


def test_smooth_quartic_response_is_not_independent_noise():
    """A known smooth fourth derivative leaves 24*7*h^4 in the first row.

    Halving h lowers this residual by sixteen, although no random error
    exists. Its derivative at the zero fiducial remains exactly zero.
    This demonstrates why the jitter interpretation needs smooth-curvature
    control; the returned score alone cannot identify the residual's cause.
    """
    def mean(parameters):
        """Return a smooth quartic and an independent linear observable."""
        x = parameters["x"]
        return np.array([7.0*x**4, 3.0*x])

    precision = np.array([[2.0, 0.25], [0.25, 1.0]])
    scores = []
    for step in (0.125, 0.0625):
        values = saved_values(predict=mean, reference={"x": 0.0}, names=["x"],
                              steps=[step])
        score = retained_stencil_noise(stencil_values=values, steps=[step],
                                       invcov=precision)
        expected_sigma = 24.0 * 7.0 * step**4 * np.sqrt(2.0 / 70.0)
        np.testing.assert_allclose(score["sigma"], [expected_sigma], rtol=3.0e-15)
        np.testing.assert_allclose(score["ddv"][:, 0], [0.0, 3.0], atol=2.0e-15)
        scores.append(score)
    np.testing.assert_allclose(scores[1]["sigma"] / scores[0]["sigma"], [1.0 / 16.0])
    np.testing.assert_allclose(scores[1]["a"] / scores[0]["a"], [1.0 / 64.0])


def test_odd_correlated_error_can_bias_fom_with_zero_jitter_score():
    """A shared linear numerical slope biases both columns but cancels in Delta4.

    The true two-row model is (x,y). Its computed version adds the same
    vector times (x+y), producing coherent column errors and a nonzero
    off-diagonal Fisher change. The exact computed FoM is 1.375 times the
    true one; the independent-jitter estimator correctly has no evidence
    for this separate error family and returns a zero diagonal addition.
    """
    def calculated(parameters):
        """Add an odd coherent error to a unit-response physical toy."""
        x = parameters["x"]
        y = parameters["y"]
        return np.array([x, y]) + (x + y) * np.array([0.25, 0.125])

    values = saved_values(predict=calculated, reference={"x": 0.0, "y": 0.0},
                          names=["x", "y"], steps=[0.125, 0.125])
    score = retained_stencil_noise(stencil_values=values, steps=[0.125, 0.125],
                                   invcov=np.eye(2))
    np.testing.assert_array_equal(x=score["a"], y=np.zeros(2))
    predicted = predict_fom_noise(F=np.eye(2), i=0, j=1, a=score["a"])
    assert predicted["ratio"] == 1.0
    actual_fisher = score["ddv"].T @ score["ddv"]
    np.testing.assert_allclose(np.sqrt(np.linalg.det(actual_fisher)), 1.375,
                               rtol=3.0e-15)
    assert abs(actual_fisher[0, 1]) > 0.1


def test_absolute_offsets_reuse_existing_h_and_half_h_model_keys():
    """Negative and zero fiducials reach the exact saved physical coordinates.

    The expensive baseline's calls populate ExactMeanCache first. Both
    shared stencils must then read those keys, including the signed-zero
    center, without another model evaluation. Mutating the caller's
    reference and names after adapter construction cannot move the center.
    """
    calls = []

    def mean(parameters):
        """Record dictionary values and produce a small smooth model."""
        calls.append(dict(parameters))
        return np.array([parameters["w0"]**2 + parameters["wa"],
                         parameters["w0"] - 2.0*parameters["wa"]])

    reference = {"w0": -0.7, "wa": -0.0, "fixed": 0.83}
    steps = {"w0": 0.013, "wa": 0.021}
    cached = ExactMeanCache(predict=mean)
    names, _, _, _ = differentiate(predict=cached, parameters=reference, steps=steps)
    assert len(calls) == 13
    adapter = make_offset_predictor(predict=cached, reference=reference, names=names)
    reference["w0"] = -9.0
    names.append("unrelated")
    for divisor in (1.0, 2.0):
        for index, name in enumerate(steps):
            step = steps[name] / divisor
            score = get_stencil_noise(dv=adapter, index=index, h=step,
                                      CV=np.zeros(2), invcov=np.eye(2))
            assert score["eps"] == step
    assert len(calls) == 13
    assert float(calls[0]["wa"]).hex() == (-0.0).hex()
    assert all(point["fixed"] == 0.83 for point in calls)
    with pytest.raises(ValueError, match="Configure candidate accuracy"):
        adapter(param=np.zeros(2), AccuracyBoost=2.0)
    assert len(calls) == 13


def test_exact_diagonal_fom_rescoring_keeps_priors_and_marginalization():
    """Compare the shared exact ratio with an independent Schur complement.

    A nuisance prior changes how much its numerical noise can affect the
    selected pair. Eliminating the nuisance block independently checks
    that the forecast is marginalized, and finite diagonal additions test
    the exact ratio beyond the small-addition expansion.
    """
    data_fisher = np.array([[5.0, 4.0, 1.0], [4.0, 5.0, 2.0], [1.0, 2.0, 3.0]])
    derivatives = np.linalg.cholesky(data_fisher).T
    fisher = fisher_matrix(derivatives=derivatives, names=["w0", "wa", "n"],
                           precision=np.eye(3), priors={"n": 0.5})
    additions = np.array([0.4, 0.2, 1.3])
    changed = fisher + np.diag(additions)
    original_pair = fisher[:2, :2] - np.outer(fisher[:2, 2], fisher[2, :2]) / fisher[2, 2]
    changed_pair = changed[:2, :2] - (
        np.outer(changed[:2, 2], changed[2, :2]) / changed[2, 2]
    )
    expected = np.sqrt(np.linalg.det(changed_pair) / np.linalg.det(original_pair))
    result = predict_fom_noise(F=fisher, i=0, j=1, a=additions)
    np.testing.assert_allclose(result["ratio"], expected, rtol=3.0e-15)
    assert not np.isclose(result["ratio"], np.exp(result["dlnFoM"]), rtol=1.0e-4)
    weights = get_fom_noise_weights(F=fisher, i=0, j=1, eps=[0.1, 0.2, 0.3])
    np.testing.assert_allclose(weights["R2"][:2], [1.0, 1.0], rtol=3.0e-15)
    without_prior = get_fom_noise_weights(F=data_fisher, i=0, j=1)
    assert weights["R2"][2] * weights["sigma2"][2] < (
        without_prior["R2"][2] * without_prior["sigma2"][2]
    )


@pytest.mark.parametrize("invalid", ["centers", "steps", "precision", "nonfinite"])
def test_retained_archive_refuses_inconsistent_inputs(invalid):
    """A mixed-center archive cannot represent one Fisher reference point."""
    values = np.zeros(shape=(2, 5, 3))
    steps = np.array([0.1, 0.2])
    precision = np.eye(3)
    if invalid == "centers":
        values[1, 2, 0] = 1.0
    elif invalid == "steps":
        steps[1] = 0.0
    elif invalid == "precision":
        precision = np.eye(2)
    else:
        values[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        retained_stencil_noise(stencil_values=values, steps=steps, invcov=precision)
