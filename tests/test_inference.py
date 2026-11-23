"""Check statistical selection, prior accounting and example configuration.

These tests use a small explicit linear mean, not CAMB or a compiled shear
model. The covariance has nonzero two-point/three-point cross terms, making
selection-before-inversion and joint-vs-independent mistakes detectable.
The examples' --check commands separately exercise the physical mean.
"""

from pathlib import Path
import json
import operator
import sys

import numpy as np
import pytest
import yaml


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))

from likelihood_example import prior_distributions
from likelihood_example import check_likelihood
from likelihood_example import make_likelihood
from cobaya.likelihoods.des_3pcf.survey import SurveyRows


def rows():
    """Define three labeled measurements with distinguishable numerical units."""
    return SurveyRows(
        kind=np.array(["xip", "xim", "map3"]),
        bins=np.array([[0, 0, -1], [0, 0, -1], [0, 0, 0]]),
        angles=np.ones((3, 3)),
        values=np.array([4.0, 5.0, 6.0]),
        labels=np.array(["xip:first", "xim:first", "map3:first"]),
    )


def linear_mean(parameters):
    """Return a linear response whose cross-correlations affect chi squared."""
    return parameters["amplitude"] * np.array([1.0, 2.0, -1.0])


def covariance():
    """An explicitly positive matrix with a nonzero xi/Map3 cross-covariance."""
    return np.array([
        [2.0, 0.3, 0.7],
        [0.3, 1.5, -0.2],
        [0.7, -0.2, 1.0],
    ])


def discrete_model_mean(parameters):
    """Select one of two physical responses using an actual integer selector.

    operator.index accepts integer scalars, including NumPy integers, but
    refuses floats. This has the same type requirement as the native IA
    selector. The amplitude remains a continuous sampled parameter.
    """
    mode = operator.index(parameters["power_mode"])
    patterns = {
        1: np.array(object=[1.0, 2.0, -1.0]),
        2: np.array(object=[1.0, -2.0, 3.0]),
    }
    return parameters["amplitude"] * patterns[mode]


@pytest.mark.parametrize("selector", [1, np.int64(1)])
def test_full_parameter_points_preserve_discrete_model_types(selector):
    """A continuous displacement must preserve fixed integer model choices."""
    initial = {"amplitude": 1, "power_mode": selector}
    likelihood = make_likelihood(
        predict=discrete_model_mean,
        rows=rows(),
        covariance=covariance(),
        parameters=initial,
    )
    # The YAML may spell a continuous reference amplitude as an integer.
    # Its displaced value must still accept a fractional change, while
    # the separate model selector retains the integer type it was given.
    displaced = {"amplitude": 1.25, "power_mode": selector}
    np.testing.assert_array_equal(
        x=likelihood.mean(parameters=displaced),
        y=discrete_model_mean(parameters=displaced),
    )
    changed_mode = {"amplitude": 1.25, "power_mode": 2}
    assert not np.array_equal(
        a1=likelihood.mean(parameters=displaced),
        a2=likelihood.mean(parameters=changed_mode),
    )
    np.testing.assert_array_equal(
        x=likelihood.mean(parameters=initial),
        y=likelihood.data,
    )


def test_selection_before_inverse_and_gaussian_normalization():
    """A two-row marginal must use its covariance, not a full-inverse slice."""
    matrix = covariance()
    selection = np.array([2, 0])
    data = np.array([0.2, 0.5, -0.1])
    likelihood = make_likelihood(
        predict=linear_mean, rows=rows(), covariance=matrix,
        parameters={"amplitude": 0.0}, data=data, indices=selection,
    )
    reduced = matrix[np.ix_(selection, selection)]
    residual = linear_mean({"amplitude": 0.7})[selection] - data[selection]
    expected_chi2 = float(residual @ np.linalg.solve(reduced, residual))
    expected_log = -0.5 * (
        expected_chi2 + 2.0 * np.log(2.0 * np.pi) + np.linalg.slogdet(reduced)[1]
    )
    score = likelihood(parameters={"amplitude": 0.7})
    np.testing.assert_allclose(score, expected_log, rtol=3.0e-15, atol=0.0)
    wrong_precision = np.linalg.inv(matrix)[np.ix_(selection, selection)]
    wrong_chi2 = float(residual @ wrong_precision @ residual)
    assert abs(wrong_chi2 - expected_chi2) > 0.01
    np.testing.assert_array_equal(likelihood.rows.labels, ["map3:first", "xip:first"])


def test_cross_covariance_is_retained_and_synthetic_data_are_fiducial():
    """The synthetic datum is the model, while the joint score retains C23."""
    matrix = covariance()
    likelihood = make_likelihood(
        predict=linear_mean, rows=rows(), covariance=matrix,
        parameters={"amplitude": 1.3},
    )
    np.testing.assert_array_equal(likelihood.data, linear_mean({"amplitude": 1.3}))
    assert not np.array_equal(likelihood.data, rows().values)
    _, fiducial_chi2 = likelihood.score_mean(mean=likelihood.data)
    assert fiducial_chi2 == 0.0
    residual = linear_mean({"amplitude": 1.4}) - likelihood.data
    expected = float(residual @ np.linalg.solve(matrix, residual))
    _, measured = likelihood.score_mean(mean=linear_mean({"amplitude": 1.4}))
    np.testing.assert_allclose(measured, expected, rtol=3.0e-15, atol=0.0)
    independent = matrix.copy()
    independent[:2, 2] = 0.0
    independent[2, :2] = 0.0
    wrong = float(residual @ np.linalg.solve(independent, residual))
    assert abs(measured - wrong) > 0.001


def test_input_ownership_and_restored_check():
    """Caller edits cannot change the fixed data/covariance after preparation."""
    data = np.array([0.1, 0.2, 0.3])
    matrix = covariance()
    initial = {"amplitude": 0.5}
    likelihood = make_likelihood(
        predict=linear_mean, rows=rows(), covariance=matrix,
        parameters=initial, data=data,
    )
    reference = likelihood(parameters={"amplitude": 0.5})
    data[:] = 40.0
    matrix[:] = 0.0
    initial["amplitude"] = 100.0
    assert likelihood(parameters={"amplitude": 0.5}) == reference
    priors = prior_distributions(priors={
        "amplitude": {
            "distribution": "normal",
            "mean": 0.0,
            "sigma": 2.0,
        },
    })
    result = check_likelihood(
        likelihood=likelihood, distributions=priors,
        specification={"check": {"parameter": "amplitude", "delta": 0.1}},
    )
    assert result["restored_bitwise"] is True
    assert result["maximum_absolute_mean_change"] > 0.0


def test_normalized_prior_is_not_a_likelihood_penalty():
    """Changing a prior changes its density, not the fixed likelihood value."""
    priors = prior_distributions(priors={
        "amplitude": {
            "distribution": "normal",
            "mean": 0.0,
            "sigma": 0.5,
            "minimum": -0.8,
            "maximum": 0.8,
        },
    })
    distribution = priors["amplitude"]
    assert distribution.cdf(0.8) - distribution.cdf(-0.8) == 1.0
    likelihood = make_likelihood(
        predict=linear_mean, rows=rows(), covariance=covariance(),
        parameters={"amplitude": 0.0},
    )
    point = {"amplitude": 0.3}
    log_like = likelihood(parameters=point)
    residual = linear_mean(point)
    expected = likelihood.log_normalization - 0.5 * (
        residual @ np.linalg.solve(covariance(), residual)
    )
    np.testing.assert_allclose(log_like, expected, rtol=3.0e-15, atol=0.0)
    assert abs(distribution.logpdf(point["amplitude"])) > 0.01


@pytest.mark.parametrize("indices", [[0, 0], [0.0, 1.0], [4], []])
def test_malformed_selection_is_rejected(indices):
    """Repeated, noninteger, out-of-range and empty selections are not datasets."""
    with pytest.raises(ValueError):
        make_likelihood(
            predict=linear_mean, rows=rows(), covariance=covariance(),
            parameters={"amplitude": 0.0}, indices=np.asarray(indices),
        )
